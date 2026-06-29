# 头部网络块（Head Block）模块
# 定义 BinaryHeadBlock：将骨干网络特征转换为二值热图
# 二值热图包含两个通道：前景（关键点区域）和背景（非关键点区域）

import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义模块公开接口
__all__ = [ 'BinaryHeadBlock' ]


class BinaryHeadBlock(nn.Module):
    """二值头部网络块
    
    该块将骨干网络提取的高维特征图映射为二值热图。
    
    网络结构：
    - 1x1卷积：通道投影（降维或保持维度）
    - 批量归一化：稳定训练
    - ReLU激活：引入非线性
    - 1x1卷积：输出二值热图（2 × 关键点数量 个通道）
    
    输出的热图包含两类通道：
    - 偶数索引通道：背景概率图（非关键点区域）
    - 奇数索引通道：前景概率图（关键点区域）
    """
    def __init__(self, in_channels, proj_channels, out_channels, **kwargs):
        """初始化二值头部网络块
        
        参数:
            in_channels: 输入通道数（来自骨干网络的特征维度，如270）
            proj_channels: 投影层通道数（中间层维度）
            out_channels: 关键点数量（如98）
        """
        super(BinaryHeadBlock, self).__init__()
        # 构建网络层序列
        self.layers = nn.Sequential(
            # 1x1卷积：将输入特征投影到指定通道数
            nn.Conv2d(in_channels, proj_channels, 1, bias=False),
            # 批量归一化：加速训练收敛
            nn.BatchNorm2d(proj_channels),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 1x1卷积：输出2倍关键点数量的通道（前景+背景）
            nn.Conv2d(proj_channels, out_channels*2, 1, bias=False),
        )
        
    def forward(self, input):
        """前向传播
        
        将特征图转换为二值热图，并reshape为 [N, 2, K, H, W] 格式
        其中 2 表示前景/背景两类，K 表示关键点数量
        
        参数:
            input: 骨干网络特征图 [N, C, H, W]
        返回:
            二值热图 [N, 2, K, H, W]
            - 第1维=0：背景概率图
            - 第1维=1：前景概率图
        """
        N, C, H, W = input.shape
        # 输出通道数 = 2 × 关键点数量，reshape为 [N, 2, K, H, W]
        return self.layers(input).view(N, 2, -1, H, W)
