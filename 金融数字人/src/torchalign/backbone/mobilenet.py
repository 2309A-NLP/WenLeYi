# MobileNetV2 - 轻量级骨干网络
# 适用于移动端和实时应用场景
# 相比HRNet，参数量和计算量大幅减少，但精度略有下降
# 核心设计：倒残差结构（Inverted Residual）+ 线性瓶颈（Linear Bottleneck）

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


# 定义模块公开接口
__all__ = ['MobileNetV2', 'mobilenetv2']


class Block(nn.Module):
    """倒残差瓶颈块（Inverted Residual Block）
    
    MobileNetV2的核心构建块，与传统瓶颈块相反：
    传统瓶颈块：降维 -> 3x3卷积 -> 升维（窄-宽-窄）
    倒残差块：升维 -> 3x3深度可分离卷积 -> 降维（宽-窄-宽）
    
    优势：
    1. 在高维空间进行特征提取，信息损失更少
    2. 深度可分离卷积大幅减少参数量和计算量
    3. 残差连接在低维瓶颈层之间进行
    """
    def __init__(self, in_channels, out_channels, expansion=1, stride=1):
        """初始化倒残差瓶颈块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            expansion: 通道扩展因子（控制中间层宽度）
            stride: 深度卷积的步长（>1时进行空间下采样）
        """
        super(Block, self).__init__()
        if expansion == 1:
            # 扩展因子为1时不进行通道扩展
            # 结构：深度卷积 -> 逐点卷积（无ReLU）
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # 扩展因子 > 1 时，先升维再降维
            channels = expansion * in_channels  # 中间层通道数
            self.conv = nn.Sequential(
                # 1x1逐点卷积：升维（扩展通道数）
                nn.Conv2d(in_channels, channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                # 3x3深度可分离卷积：在高维空间进行空间特征提取
                # groups=channels 表示每个通道独立卷积，大幅减少参数量
                nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                # 1x1逐点卷积：降维（恢复通道数）
                nn.Conv2d(channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        # 残差连接条件：步长为1 且 输入输出通道数相同
        self.residual = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        """前向传播
        
        当满足残差连接条件时，输出 = F(x) + x
        否则直接输出卷积结果
        """
        out = self.conv(x)
        if self.residual:
            out = out + x
        return out


class MobileNetV2(nn.Module):
    """MobileNetV2 网络模型
    
    用于人脸关键点检测的骨干网络，提取多尺度特征。
    与HRNet不同，MobileNetV2采用逐级降采样的方式，
    但通过多尺度特征融合来弥补分辨率损失。
    
    特征提取层次：
    - C2（浅层特征）：丰富的细节信息，如边缘、纹理
    - C3（中层特征）：中等语义信息
    - C4（深层特征）：高级语义信息，如面部结构
    """
    def __init__(self, config):
        """初始化MobileNetV2
        
        参数:
            config: 网络配置列表，每个元素为 (expansion, out_channels, blocks, stride)
        """
        super(MobileNetV2, self).__init__()
        in_channels = config[0][1]
        # 初始卷积层：3通道输入 -> 配置的初始通道数
        features = [nn.Sequential(
            nn.Conv2d(3, in_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )]
        # 根据配置逐层构建倒残差瓶颈块
        for expansion, out_channels, blocks, stride in config[1:]:
            for i in range(blocks):
                # 只在每组的第一个块使用配置的步长，其余步长为1
                features.append(Block(in_channels, out_channels, expansion, stride if i == 0 else 1))
                in_channels = out_channels
        self.features = nn.Sequential(*features)

    def forward(self, x):
        """前向传播 - 提取多尺度特征
        
        将网络分为三个阶段，分别提取不同层次的特征，
        然后上采样到相同尺寸后沿通道维度拼接。
        
        参数:
            x: 输入图像 [N, 3, H, W]
        返回:
            多尺度融合特征 [N, C2+C3+C4, H/4, W/4]
        """
        # 浅层特征（前4层）
        c2 = self.features[:4](x)
        # 中层特征（第4-7层）
        c3 = self.features[4:7](c2)
        # 深层特征（第7-14层）
        c4 = self.features[7:14](c3)
        # 将所有特征上采样到c2的尺寸后拼接
        kwargs = {'size': c2.shape[-2:],'mode': 'bilinear','align_corners': False}
        return torch.cat([F.interpolate(xx,**kwargs) for xx in [c2,c3,c4]], 1)


def mobilenetv2(pretrained=False, **kwargs):
    """构建 MobileNetV2 模型
    
    Args:
        pretrained (bool): 如果为 True，加载 ImageNet 预训练权重
        
    网络配置说明：
    - (1, 32, 1, 1): 无扩展，32通道，1个块，步长1
    - (6, 24, 2, 2): 6倍扩展，24通道，2个块，第一个块步长2
    - (6, 32, 3, 2): 6倍扩展，32通道，3个块
    - (6, 64, 4, 2): 6倍扩展，64通道，4个块
    - (6, 96, 3, 1): 6倍扩展，96通道，3个块
    """
    config = [
        (1,  32, 1, 1),   # 初始层：无扩展，32通道
        (1,  16, 1, 1),   # 第一层：无扩展，16通道
        (6,  24, 2, 2),   # 第二层：6倍扩展，24通道，下采样
        (6,  32, 3, 2),   # 第三层：6倍扩展，32通道，下采样
        (6,  64, 4, 2),   # 第四层：6倍扩展，64通道，下采样
        (6,  96, 3, 1),   # 第五层：6倍扩展，96通道
    ]
    model = MobileNetV2(config, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['mobilenetv2']), strict=False)
    return model

