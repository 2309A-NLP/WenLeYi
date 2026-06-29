"""
PartialFC（部分全连接）训练模块
本模块实现了PartialFC算法，用于在单机上训练超大规模（千万级）人脸身份。

核心思想：
- 不需要将所有类别的softmax权重都加载到GPU显存中
- 每次训练只采样部分正负样本的类别中心
- 通过内存银行（memory bank）维护完整的softmax权重
- 大幅减少GPU显存占用，同时保持训练精度

参考论文：
Partial FC: Training 10 Million Identities on a Single Machine
https://arxiv.org/abs/2010.05222
作者：Xiang An, Yang Xiao, XuHan Zhu (DeepGlint)
"""
import logging
import os

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.functional import normalize, linear
from torch.nn.parameter import Parameter


class PartialFC(Module):
    """
    PartialFC模块：部分全连接层
    
    将softmax分类层的权重分布到多个GPU上，每个GPU只维护一部分类别中心。
    通过采样机制减少每次前向传播需要计算的类别数量。
    
    支持分布式训练，通过all_gather收集所有GPU的标签和特征。
    """

    @torch.no_grad()
    def __init__(self, rank, local_rank, world_size, batch_size, resume,
                 margin_softmax, num_classes, sample_rate=1.0, embedding_size=512, prefix="./"):
        """
        初始化PartialFC模块
        
        参数:
            rank: int
                当前进程的全局唯一ID（从0到world_size-1）
            local_rank: int
                当前进程在本机内的GPU编号（从0到7）
            world_size: int
                GPU总数
            batch_size: int
                当前rank（GPU）上的批量大小
            resume: bool
                是否从检查点恢复softmax权重
            margin_softmax: callable
                角间隔softmax函数（如cosface、arcface）
            num_classes: int
                存储在当前rank上的类别中心数量，通常为总类别数除以GPU数
            sample_rate: float
                PartialFC的采样率，当类别数超过200万时，
                采样可以大幅加速训练并减少显存占用，默认1.0
            embedding_size: int
                特征维度，默认512
            prefix: str
                检查点保存路径，默认'./'
        """
        super(PartialFC, self).__init__()
        #
        self.num_classes: int = num_classes  # 总类别数
        self.rank: int = rank  # 全局进程ID
        self.local_rank: int = local_rank  # 本地GPU编号
        self.device: torch.device = torch.device("cuda:{}".format(self.local_rank))  # 计算设备
        self.world_size: int = world_size  # GPU总数
        self.batch_size: int = batch_size  # 批量大小
        self.margin_softmax: callable = margin_softmax  # 角间隔softmax函数
        self.sample_rate: float = sample_rate  # 采样率
        self.embedding_size: int = embedding_size  # 特征维度
        self.prefix: str = prefix  # 保存路径
        # 计算当前rank负责的类别范围
        self.num_local: int = num_classes // world_size + int(rank < num_classes % world_size)
        self.class_start: int = num_classes // world_size * rank + min(rank, num_classes % world_size)
        self.num_sample: int = int(self.sample_rate * self.num_local)  # 采样数量

        # 权重文件路径（用于检查点保存和恢复）
        self.weight_name = os.path.join(self.prefix, "rank_{}_softmax_weight.pt".format(self.rank))
        self.weight_mom_name = os.path.join(self.prefix, "rank_{}_softmax_weight_mom.pt".format(self.rank))

        if resume:
            try:
                # 从检查点恢复权重
                self.weight: torch.Tensor = torch.load(self.weight_name)
                self.weight_mom: torch.Tensor = torch.load(self.weight_mom_name)
                if self.weight.shape[0] != self.num_local or self.weight_mom.shape[0] != self.num_local:
                    raise IndexError  # 形状不匹配
                logging.info("softmax weight resume successfully!")
                logging.info("softmax weight mom resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                # 恢复失败，重新初始化
                self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
                self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
                logging.info("softmax weight init!")
                logging.info("softmax weight mom init!")
        else:
            # 全新初始化
            self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
            self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
            logging.info("softmax weight init successfully!")
            logging.info("softmax weight mom init successfully!")
        self.stream: torch.cuda.Stream = torch.cuda.Stream(local_rank)  # 异步CUDA流

        self.index = None
        if int(self.sample_rate) == 1:
            # 采样率为1时，不进行采样，直接使用全部权重
            self.update = lambda: 0
            self.sub_weight = Parameter(self.weight)
            self.sub_weight_mom = self.weight_mom
        else:
            # 采样率小于1时，需要动态采样
            self.sub_weight = Parameter(torch.empty((0, 0)).cuda(local_rank))

    def save_params(self):
        """
        保存softmax权重到文件（用于检查点保存）
        """
        torch.save(self.weight.data, self.weight_name)
        torch.save(self.weight_mom, self.weight_mom_name)

    @torch.no_grad()
    def sample(self, total_label):
        """
        采样类别中心
        
        从当前rank负责的类别中，保留所有正样本类别中心，
        并随机采样负样本类别中心，凑满num_sample个。
        
        参数:
            total_label: 所有GPU聚合后的标签张量
        """
        # 筛选出属于当前rank的正样本
        index_positive = (self.class_start <= total_label) & (total_label < self.class_start + self.num_local)
        total_label[~index_positive] = -1  # 不属于当前rank的标签设为-1
        total_label[index_positive] -= self.class_start  # 转换为本地索引
        if int(self.sample_rate) != 1:
            # 部分采样
            positive = torch.unique(total_label[index_positive], sorted=True)  # 获取所有正样本类别
            if self.num_sample - positive.size(0) >= 0:
                # 采样数量足够，随机选择负样本填充
                perm = torch.rand(size=[self.num_local], device=self.device)
                perm[positive] = 2.0  # 正样本优先选择
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]  # 排序以保持顺序
            else:
                # 采样数量不足，只使用正样本
                index = positive
            self.index = index
            total_label[index_positive] = torch.searchsorted(index, total_label[index_positive])
            self.sub_weight = Parameter(self.weight[index])  # 提取采样的权重
            self.sub_weight_mom = self.weight_mom[index]

    def forward(self, total_features, norm_weight):
        """
        PartialFC前向传播：logits = X * W（特征与归一化权重的线性变换）
        
        参数:
            total_features: 所有GPU聚合后的特征张量
            norm_weight: 归一化后的softmax权重
        返回:
            logits: 分类logits
        """
        torch.cuda.current_stream().wait_stream(self.stream)  # 等待异步流完成
        logits = linear(total_features, norm_weight)  # 线性变换
        return logits

    @torch.no_grad()
    def update(self):
        """
        将更新后的权重和动量写回内存银行
        """
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, label, optimizer):
        """
        准备阶段：收集所有GPU的标签，执行采样，设置优化器
        
        参数:
            label: 当前rank的标签张量
            optimizer: PartialFC的优化器（需要更新momentum_buffer）
        返回:
            total_label: 所有GPU聚合后的标签
            norm_weight: 归一化后的softmax权重
        """
        with torch.cuda.stream(self.stream):
            # 收集所有GPU的标签
            total_label = torch.zeros(
                size=[self.batch_size * self.world_size], device=self.device, dtype=torch.long)
            dist.all_gather(list(total_label.chunk(self.world_size, dim=0)), label)
            self.sample(total_label)  # 执行采样
            # 更新优化器状态中的权重引用
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            norm_weight = normalize(self.sub_weight)  # L2归一化
            return total_label, norm_weight

    def forward_backward(self, label, features, optimizer):
        """
        PartialFC的完整前向和反向传播（模型并行）
        
        实现了分布式训练中的交叉熵损失计算和梯度传播。
        通过手动计算softmax和交叉熵的梯度，避免显存中存储完整的logits矩阵。
        
        参数:
            label: 当前rank的标签张量
            features: 当前rank的特征张量
            optimizer: PartialFC的优化器
        返回:
            x_grad: 特征的梯度（需要传播回backbone）
            loss_v: 交叉熵损失值
        """
        # 准备阶段：收集标签并采样
        total_label, norm_weight = self.prepare(label, optimizer)
        # 收集所有GPU的特征
        total_features = torch.zeros(
            size=[self.batch_size * self.world_size, self.embedding_size], device=self.device)
        dist.all_gather(list(total_features.chunk(self.world_size, dim=0)), features.data)
        total_features.requires_grad = True  # 需要计算梯度

        # 前向传播
        logits = self.forward(total_features, norm_weight)
        logits = self.margin_softmax(logits, total_label)  # 应用角间隔

        with torch.no_grad():
            # 数值稳定的softmax计算
            max_fc = torch.max(logits, dim=1, keepdim=True)[0]
            dist.all_reduce(max_fc, dist.ReduceOp.MAX)  # 全局最大值

            # 计算exp(logits - max_fc)并全局求和
            logits_exp = torch.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)
            dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)

            # 计算概率分布
            logits_exp.div_(logits_sum_exp)

            # 获取one-hot编码
            grad = logits_exp
            index = torch.where(total_label != -1)[0]
            one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
            one_hot.scatter_(1, total_label[index, None], 1)

            # 计算交叉熵损失
            loss = torch.zeros(grad.size()[0], 1, device=grad.device)
            loss[index] = grad[index].gather(1, total_label[index, None])
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            loss_v = loss.clamp_min_(1e-30).log_().mean() * (-1)  # 负对数似然

            # 计算梯度：softmax概率 - one_hot
            grad[index] -= one_hot
            grad.div_(self.batch_size * self.world_size)  # 归一化

        # 反向传播：计算logits的梯度
        logits.backward(grad)
        if total_features.grad is not None:
            total_features.grad.detach_()
        # 通过reduce_scatter将特征梯度分发到各GPU
        x_grad: torch.Tensor = torch.zeros_like(features, requires_grad=True)
        dist.reduce_scatter(x_grad, list(total_features.grad.chunk(self.world_size, dim=0)))
        x_grad = x_grad * self.world_size  # 缩放梯度
        # 返回梯度和损失值，用于backbone的反向传播
        return x_grad, loss_v
