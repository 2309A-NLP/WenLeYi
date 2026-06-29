"""
ArcFace分布式训练主脚本
本模块实现了基于PartialFC的人脸识别模型分布式训练流程。

主要功能：
1. 支持多GPU分布式训练（基于NCCL后端）
2. 支持混合精度训练（FP16）
3. 支持学习率预热和阶梯衰减
4. 支持模型检查点保存和恢复
5. 定期在验证集上评估模型性能

使用方法：
    torchrun --nproc_per_node=8 train.py config.py

训练流程：
1. 初始化分布式环境
2. 加载数据集和骨干网络
3. 创建PartialFC模块和优化器
4. 循环训练：前向传播 -> PartialFC前向反向 -> 反向传播 -> 参数更新
5. 定期验证和保存检查点
"""
import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

import losses
from backbones import get_model
from dataset import MXFaceDataset, SyntheticDataset, DataLoaderX
from partial_fc import PartialFC
from utils.utils_amp import MaxClipGradScaler
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging


def main(args):
    """
    训练主函数
    
    参数:
        args: 命令行参数，包含config文件路径和local_rank
    """
    cfg = get_config(args.config)  # 加载训练配置

    # ========== 初始化分布式训练环境 ==========
    try:
        world_size = int(os.environ['WORLD_SIZE'])  # 从环境变量获取GPU总数
        rank = int(os.environ['RANK'])               # 从环境变量获取全局rank
        dist.init_process_group('nccl')  # 初始化NCCL分布式后端
    except KeyError:
        # 单GPU模式（用于调试）
        world_size = 1
        rank = 0
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:12584", rank=rank, world_size=world_size)

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)  # 设置当前GPU
    os.makedirs(cfg.output, exist_ok=True)  # 创建输出目录
    init_logging(rank, cfg.output)  # 初始化日志

    # ========== 加载数据集 ==========
    if cfg.rec == "synthetic":
        train_set = SyntheticDataset(local_rank=local_rank)  # 合成数据（用于调试）
    else:
        train_set = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)  # MXNet格式数据集

    # 分布式采样器（确保每个GPU看到不同的数据子集）
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    # 使用增强版DataLoader（异步GPU数据传输）
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)

    # ========== 创建骨干网络 ==========
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(local_rank)

    # ========== 恢复检查点（可选） ==========
    if cfg.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")  # 骨干网络权重路径
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank == 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            if rank == 0:
                logging.info("resume fail, backbone init successfully!")

    # 包装为分布式数据并行（DDP）
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()  # 设置为训练模式

    # ========== 创建损失函数和PartialFC模块 ==========
    margin_softmax = losses.get_loss(cfg.loss)  # 创建角间隔损失函数
    module_partial_fc = PartialFC(
        rank=rank, local_rank=local_rank, world_size=world_size, resume=cfg.resume,
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

    # ========== 创建优化器 ==========
    # 骨干网络优化器（学习率按批量大小线性缩放）
    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,  # 线性缩放学习率
        momentum=0.9, weight_decay=cfg.weight_decay)
    # PartialFC优化器
    opt_pfc = torch.optim.SGD(
        params=[{'params': module_partial_fc.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    # ========== 计算训练步数和学习率调度 ==========
    num_image = len(train_set)  # 训练集大小
    total_batch_size = cfg.batch_size * world_size  # 总批量大小
    cfg.warmup_step = num_image // total_batch_size * cfg.warmup_epoch  # 预热步数
    cfg.total_step = num_image // total_batch_size * cfg.num_epoch  # 总训练步数

    def lr_step_func(current_step):
        """
        学习率调度函数
        
        预热阶段：学习率线性增长
        训练阶段：在指定轮数进行阶梯衰减（每次衰减为原来的0.1倍）
        """
        cfg.decay_step = [x * num_image // total_batch_size for x in cfg.decay_epoch]
        if current_step < cfg.warmup_step:
            return current_step / cfg.warmup_step  # 线性预热
        else:
            return 0.1 ** len([m for m in cfg.decay_step if m <= current_step])  # 阶梯衰减

    # 为骨干网络和PartialFC分别创建学习率调度器
    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=lr_step_func)
    scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_pfc, lr_lambda=lr_step_func)

    # 打印配置信息
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # ========== 创建回调函数 ==========
    val_target = cfg.val_targets  # 验证集列表
    callback_verification = CallBackVerification(2000, rank, val_target, cfg.rec)  # 验证回调
    callback_logging = CallBackLogging(50, rank, cfg.total_step, cfg.batch_size, world_size, None)  # 日志回调
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)  # 检查点回调

    # ========== 训练循环 ==========
    loss = AverageMeter()  # 损失累加器
    start_epoch = 0
    global_step = 0
    # 混合精度训练的梯度缩放器
    grad_amp = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None

    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)  # 设置epoch（影响分布式采样的shuffle）
        for step, (img, label) in enumerate(train_loader):
            global_step += 1

            # ===== 前向传播 =====
            features = F.normalize(backbone(img))  # 提取特征并L2归一化

            # ===== PartialFC前向反向传播（计算分类损失和特征梯度） =====
            x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)

            # ===== 骨干网络反向传播 =====
            if cfg.fp16:
                # 混合精度训练
                features.backward(grad_amp.scale(x_grad))  # 使用缩放的梯度
                grad_amp.unscale_(opt_backbone)  # 反缩放
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)  # 梯度裁剪
                grad_amp.step(opt_backbone)  # 更新参数
                grad_amp.update()  # 更新缩放因子
            else:
                # 标准精度训练
                features.backward(x_grad)  # 反向传播
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)  # 梯度裁剪
                opt_backbone.step()  # 更新参数

            # ===== 更新PartialFC权重和清零梯度 =====
            opt_pfc.step()  # 更新PartialFC参数
            module_partial_fc.update()  # 将更新后的权重写回内存银行
            opt_backbone.zero_grad()  # 清零骨干网络梯度
            opt_pfc.zero_grad()  # 清零PartialFC梯度

            # ===== 日志记录和验证 =====
            loss.update(loss_v, 1)  # 更新损失
            callback_logging(global_step, loss, epoch, cfg.fp16, scheduler_backbone.get_last_lr()[0], grad_amp)
            callback_verification(global_step, backbone)  # 定期验证

            # ===== 更新学习率 =====
            scheduler_backbone.step()
            scheduler_pfc.step()

        # 每个epoch结束保存检查点
        callback_checkpoint(global_step, backbone, module_partial_fc)

    # 训练结束，销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')  # 配置文件路径
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')  # 本地GPU编号
    main(parser.parse_args())
