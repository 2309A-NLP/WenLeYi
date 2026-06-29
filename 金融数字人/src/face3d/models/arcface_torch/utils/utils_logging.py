"""
日志工具模块

该模块提供了训练过程中的日志记录功能，包括：
1. AverageMeter 类：用于计算和存储指标的平均值（如损失、速度等）
2. init_logging 函数：初始化全局日志系统

日志系统支持同时输出到控制台和文件，便于训练过程的监控和问题排查。
"""

import logging
import os
import sys


class AverageMeter(object):
    """
    平均值计算器
    
    用于实时计算和存储某个指标的当前值、累计值、平均值和计数。
    常用于跟踪训练损失、推理速度等指标。
    
    使用示例：
        meter = AverageMeter()
        meter.update(loss.item(), n=batch_size)
        print(meter.avg)  # 打印平均损失
        
        meter.reset()  # 重置统计
    """

    def __init__(self):
        self.val = None      # 最近一次更新的值
        self.avg = None      # 累计平均值
        self.sum = None      # 累计总和
        self.count = None    # 累计计数
        self.reset()

    def reset(self):
        """重置所有统计值为 0"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        更新统计值
        
        参数:
            val: 新的值
            n (int): 该值对应的样本数（默认为 1）
        """
        self.val = val
        self.sum += val * n      # 累加总和
        self.count += n          # 累加计数
        self.avg = self.sum / self.count  # 更新平均值


def init_logging(rank, models_root):
    """
    初始化全局日志系统
    
    仅在主进程（rank=0）上初始化日志，避免多进程重复输出。
    日志同时输出到：
    1. 控制台（stdout）：实时查看训练状态
    2. 训练日志文件（training.log）：持久化保存
    
    参数:
        rank (int): 当前进程 rank（0 为主进程）
        models_root (str): 模型保存目录，日志文件保存在此目录下
    """
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        # 日志格式：Training: 时间戳-消息内容
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        
        # 文件处理器：写入 training.log
        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        # 控制台处理器：输出到标准输出
        handler_stream = logging.StreamHandler(sys.stdout)
        
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank_id: %d' % rank)
