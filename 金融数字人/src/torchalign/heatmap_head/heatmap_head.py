# 热图预测头（Heatmap Head）模块
# 负责将骨干网络输出的特征图转换为人脸关键点坐标
# 整体流程：特征图 -> 网络块处理 -> 热图 -> 坐标解码

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入子模块：blocks（网络块）和 transforms（坐标变换）
from . import blocks, transforms


# 定义模块公开接口
__all__ = [ 'HeatmapHead' ]


class HeatmapHead(nn.Module):
    """热图预测头
    
    该模块是关键点检测的"头部"网络，包含两个核心组件：
    1. head（网络块）：将骨干网络特征映射到热图空间
       - 输入：骨干网络输出的特征图（如270通道）
       - 输出：二值热图（前景/背景概率图）
    2. decoder（解码器）：将热图解码为关键点坐标
       - 输入：二值热图
       - 输出：关键点的(x, y)坐标
    """
    def __init__(self, cfg, **kwargs):
        """初始化热图预测头
        
        参数:
            cfg: 配置对象，包含热图相关的所有超参数
        """
        super(HeatmapHead, self).__init__()
        # 创建解码器：将二值热图转换为坐标
        # topk: 解码时选取的最大值点数量（用于加权平均）
        # stride: 热图到坐标的缩放步长
        self.decoder = transforms.__dict__[cfg.HEATMAP.DECODER](
            topk=cfg.HEATMAP.TOPK,
            stride=cfg.HEATMAP.STRIDE,
        )
        # 创建网络块：特征图到热图的映射
        # in_channels: 输入通道数（骨干网络输出）
        # proj_channels: 投影层通道数（中间层）
        # out_channels: 输出通道数（关键点数量 × 2，前景/背景各一个通道）
        self.head = blocks.__dict__[cfg.HEATMAP.BLOCK](
            in_channels=cfg.HEATMAP.IN_CHANNEL,
            proj_channels=cfg.HEATMAP.PROJ_CHANNEL,
            out_channels=cfg.HEATMAP.OUT_CHANNEL,
        )
        
    def forward(self, input):
        """前向传播
        
        流程：特征图 -> head处理得到二值热图 -> decoder解码得到坐标
        
        参数:
            input: 骨干网络输出的特征图 [N, C, H, W]
        返回:
            关键点坐标 [N, K, 2]，K为关键点数量
        """
        # head将特征图转换为二值热图，decoder将热图解码为坐标
        return self.decoder(self.head(input))
        
