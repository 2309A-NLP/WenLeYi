"""
训练回调函数模块

该模块实现了训练过程中的各种回调函数，用于：
1. 验证评估回调（CallBackVerification）：定期在验证集上测试模型精度
2. 训练日志回调（CallBackLogging）：定期记录训练速度、损失、学习率等信息
3. 模型检查点回调（CallBackModelCheckpoint）：定期保存模型权重

这些回调函数在训练循环中被周期性调用，实现训练过程的监控和模型保存。
"""

import logging
import os
import time
from typing import List

import torch

from eval import verification
from utils.utils_logging import AverageMeter


class CallBackVerification(object):
    """
    验证评估回调类
    
    在训练过程中定期执行人脸验证评估，监控模型在标准数据集
    （如 LFW、CFP-FP、AgeDB-30）上的精度变化。
    
    仅在主进程（rank=0）上执行验证，避免多 GPU 重复计算。
    
    参数:
        frequent (int): 验证频率（每隔多少个 step 验证一次）
        rank (int): 进程 rank（0 为主进程）
        val_targets (list): 验证数据集名称列表
        rec_prefix (str): 数据集根路径前缀
        image_size (tuple): 输入图像尺寸，默认 (112, 112)
    """
    def __init__(self, frequent, rank, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent          # 验证频率
        self.rank: int = rank                  # 当前进程 rank
        self.highest_acc: float = 0.0          # 全局最高精度
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)  # 各数据集最高精度
        self.ver_list: List[object] = []       # 验证数据集列表
        self.ver_name_list: List[str] = []     # 验证数据集名称列表
        if self.rank is 0:
            # 仅主进程加载验证数据集
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        """
        执行验证测试
        
        对所有验证数据集进行测试，记录精度和最高精度。
        
        参数:
            backbone: 人脸验证骨干网络
            global_step: 当前训练步数
        """
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        """
        初始化验证数据集
        
        从指定目录加载各验证数据集的二进制文件。
        
        参数:
            val_targets: 验证数据集名称列表
            data_dir: 数据集根目录
            image_size: 图像尺寸
        """
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        """
        回调函数入口
        
        按照指定频率触发验证测试。
        
        参数:
            num_update: 当前训练步数
            backbone: 骨干网络模型
        """
        if self.rank is 0 and num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()           # 切换到评估模式
            self.ver_test(backbone, num_update)
            backbone.train()          # 切换回训练模式


class CallBackLogging(object):
    """
    训练日志回调类
    
    定期记录训练过程中的关键指标，包括：
    - 训练速度（samples/sec）
    - 平均损失
    - 学习率
    - 当前 epoch 和全局步数
    - 预计剩余训练时间
    
    支持 TensorBoard 写入和日志输出。
    
    参数:
        frequent (int): 日志记录频率
        rank (int): 进程 rank
        total_step (int): 总训练步数
        batch_size (int): 批量大小
        world_size (int): 总 GPU 数量
        writer: TensorBoard SummaryWriter（可选）
    """
    def __init__(self, frequent, rank, total_step, batch_size, world_size, writer=None):
        self.frequent: int = frequent          # 日志记录频率
        self.rank: int = rank                  # 当前进程 rank
        self.time_start = time.time()          # 训练开始时间
        self.total_step: int = total_step      # 总训练步数
        self.batch_size: int = batch_size      # 批量大小
        self.world_size: int = world_size      # GPU 总数
        self.writer = writer                   # TensorBoard writer

        self.init = False
        self.tic = 0                           # 上次记录的时间戳

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        """
        回调函数入口
        
        按照指定频率记录训练日志信息。
        
        参数:
            global_step: 全局训练步数
            loss: 损失平均值计算器
            epoch: 当前 epoch
            fp16: 是否使用混合精度训练
            learning_rate: 当前学习率
            grad_scaler: 梯度缩放器（FP16 训练时使用）
        """
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    # 计算训练速度（样本/秒）
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size  # 所有 GPU 的总速度
                except ZeroDivisionError:
                    speed_total = float('inf')

                # 计算预计剩余训练时间
                time_now = (time.time() - self.time_start) / 3600  # 已训练时间（小时）
                time_total = time_now / ((global_step + 1) / self.total_step)  # 预计总时间
                time_for_end = time_total - time_now  # 剩余时间
                
                # 写入 TensorBoard（如果启用了 writer）
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                
                # 格式化日志消息
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.4f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.4f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step, time_for_end
                          )
                logging.info(msg)
                loss.reset()       # 重置损失统计
                self.tic = time.time()  # 更新时间戳
            else:
                # 首次调用：初始化时间戳
                self.init = True
                self.tic = time.time()


class CallBackModelCheckpoint(object):
    """
    模型检查点保存回调类
    
    定期保存模型权重到文件，用于：
    1. 训练中断后的恢复（resume）
    2. 选择最佳模型进行部署
    
    仅在主进程（rank=0）上执行保存操作。
    
    参数:
        rank (int): 进程 rank
        output (str): 模型保存目录，默认 "./"
    """
    def __init__(self, rank, output="./"):
        self.rank: int = rank
        self.output: str = output

    def __call__(self, global_step, backbone, partial_fc, ):
        """
        回调函数入口
        
        在 global_step > 100 后开始保存模型，避免保存初始随机权重。
        
        参数:
            global_step: 全局训练步数
            backbone: 骨干网络模型
            partial_fc: Partial FC 层（可选，用于大类别数训练）
        """
        if global_step > 100 and self.rank == 0:
            # 保存骨干网络权重
            path_module = os.path.join(self.output, "backbone.pth")
            torch.save(backbone.module.state_dict(), path_module)
            logging.info("Pytorch Model Saved in '{}'".format(path_module))

        # 保存 Partial FC 层参数（如果存在）
        if global_step > 100 and partial_fc is not None:
            partial_fc.save_params()
