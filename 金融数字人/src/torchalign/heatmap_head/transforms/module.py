# 热图到坐标转换的模块化实现
# 提供 BinaryHeatmap2Coordinate 类，将二值热图解码为关键点坐标
# 二值热图包含前景和背景两个通道，解码时只使用前景通道

import torch
import torch.nn as nn

# 导入函数式工具模块
from . import functional as F


# 定义模块公开接口
__all__ = [ 'BinaryHeatmap2Coordinate' ]


class BinaryHeatmap2Coordinate(nn.Module):
    """二值热图到坐标转换模块
    
    该模块将 BinaryHeadBlock 输出的二值热图转换为关键点坐标。
    
    二值热图格式：[N, 2, K, H, W]
    - 第1维=0：背景概率图（忽略）
    - 第1维=1：前景概率图（关键点区域）
    
    解码流程：
    1. 取出前景概率图（input[:,1,...]）
    2. 使用 heatmap2coord 进行 soft-argmax 解码
    3. 乘以步长因子将热图坐标转换为输入图像坐标
    """
    def __init__(self, stride=4.0, topk=5, **kwargs):
        """初始化二值热图到坐标转换模块
        
        参数:
            stride: 步长因子，热图坐标 × stride = 输入图像坐标
                    由于骨干网络通常进行4倍下采样，所以默认为4.0
            topk: 解码时选取的最大值点数量，默认为5
        """
        super(BinaryHeatmap2Coordinate, self).__init__()
        self.topk = topk  # soft-argmax 使用的 top-k 值
        self.stride = stride  # 坐标缩放步长
        
    def forward(self, input):
        """前向传播
        
        从二值热图的前景通道提取关键点坐标
        
        参数:
            input: 二值热图 [N, 2, K, H, W]
                   input[:,0,...] = 背景概率图
                   input[:,1,...] = 前景概率图
        返回:
            关键点坐标 [N, K, 2]，已缩放到输入图像坐标系
        """
        # 取前景通道（索引1），使用 soft-argmax 解码，再乘以步长
        return self.stride * F.heatmap2coord(input[:,1,...], self.topk)
        
    def __repr__(self):
        """返回模块的字符串表示，便于调试和日志记录"""
        format_string = self.__class__.__name__ + '('
        format_string += 'topk={}, '.format(self.topk)
        format_string += 'stride={}'.format(self.stride)
        format_string += ')'
        return format_string
