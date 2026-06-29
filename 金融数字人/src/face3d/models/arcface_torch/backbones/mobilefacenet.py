"""
MobileFaceNet轻量级人脸特征提取网络
本模块实现了MobileFaceNet网络架构，专为移动端和嵌入式设备的人脸识别设计。

主要特点：
1. 轻量级设计，参数量少，推理速度快
2. 使用深度可分离卷积减少计算量
3. 使用全局深度卷积GDC代替全连接层进行特征聚合
4. 适合移动端部署和实时人脸识别

原始代码参考：https://github.com/cavalleria/cavaface.pytorch/blob/master/backbone/mobilefacenet.py
原始作者：cavalleria
"""

import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch


class Flatten(Module):
    """展平层，将多维特征图展平为一维向量"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(Module):
    """
    卷积模块：Conv2d -> BatchNorm2d -> PReLU
    用于特征提取的基本卷积块
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        """初始化卷积模块"""
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_c),
            PReLU(num_parameters=out_c)
        )

    def forward(self, x):
        """前向传播"""
        return self.layers(x)


class LinearBlock(Module):
    """
    线性卷积模块：Conv2d -> BatchNorm2d（不带激活函数）
    用于维度变换，不改变特征的非线性
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        """初始化线性卷积模块"""
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        """前向传播"""
        return self.layers(x)


class DepthWise(Module):
    """
    深度可分离卷积模块
    
    结构：1x1逐点卷积 -> 3x3深度卷积 -> 1x1逐点卷积（线性）
    可选残差连接
    """
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        """初始化深度可分离卷积模块"""
        super(DepthWise, self).__init__()
        self.residual = residual  # 是否使用残差连接
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),  # 1x1逐点卷积
            ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),  # 3x3深度卷积
            LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))  # 1x1逐点卷积
        )

    def forward(self, x):
        """前向传播（可选残差连接）"""
        short_cut = None
        if self.residual:
            short_cut = x  # 保存输入用于残差连接
        x = self.layers(x)
        if self.residual:
            output = short_cut + x  # 残差连接
        else:
            output = x
        return output


class Residual(Module):
    """
    残差块：由多个DepthWise模块堆叠而成
    每个DepthWise模块都带有残差连接
    """
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        """初始化残差块"""
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise(c, c, True, kernel, stride, padding, groups))
        self.layers = Sequential(*modules)

    def forward(self, x):
        """前向传播"""
        return self.layers(x)


class GDC(Module):
    """
    全局深度卷积（Global Depthwise Convolution）
    
    用深度卷积替代传统的全连接层进行全局特征聚合，
    优点是参数量少、效率高，适合移动端部署。
    """
    def __init__(self, embedding_size):
        """初始化GDC模块"""
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),  # 全局深度卷积
            Flatten(),  # 展平
            Linear(512, embedding_size, bias=False),  # 全连接层
            BatchNorm1d(embedding_size))  # 特征归一化

    def forward(self, x):
        """前向传播"""
        return self.layers(x)


class MobileFaceNet(Module):
    """
    MobileFaceNet主网络
    
    网络结构：
    - 多个ConvBlock、DepthWise、Residual模块的堆叠
    - 1x1卷积进行通道压缩
    - GDC（全局深度卷积）进行特征聚合
    - 输出固定维度的人脸特征向量
    
    网络层次逐步降低空间分辨率，增加通道数，提取多层次特征。
    """
    def __init__(self, fp16=False, num_features=512):
        """初始化MobileFaceNet"""
        super(MobileFaceNet, self).__init__()
        scale = 2  # 通道缩放因子，使网络有更大的表示能力
        self.fp16 = fp16  # 是否使用半精度
        # 主干网络（逐步降采样并增加通道数）
        self.layers = nn.Sequential(
            ConvBlock(3, 64 * scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1)),  # 输入层
            ConvBlock(64 * scale, 64 * scale, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64),  # 深度卷积
            DepthWise(64 * scale, 64 * scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128),  # 下采样
            Residual(64 * scale, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),  # 4个残差块
            DepthWise(64 * scale, 128 * scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256),  # 下采样
            Residual(128 * scale, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),  # 6个残差块
            DepthWise(128 * scale, 128 * scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512),  # 下采样
            Residual(128 * scale, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),  # 2个残差块
        )
        self.conv_sep = ConvBlock(128 * scale, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))  # 1x1通道压缩
        self.features = GDC(num_features)  # 全局深度卷积特征聚合
        self._initialize_weights()  # 权重初始化

    def _initialize_weights(self):
        """权重初始化：卷积层使用Kaiming初始化，BN层初始化为标准值"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  # BN权重初始化为1
                m.bias.data.zero_()     # BN偏置初始化为0
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像张量，形状为 (N, 3, 112, 112)
        返回:
            人脸特征向量，形状为 (N, num_features)
        """
        with torch.cuda.amp.autocast(self.fp16):  # 主干网络使用半精度
            x = self.layers(x)
        x = self.conv_sep(x.float() if self.fp16 else x)  # 通道压缩（转为float32）
        x = self.features(x)  # GDC特征聚合
        return x


def get_mbf(fp16, num_features):
    """创建MobileFaceNet模型实例
    
    参数:
        fp16: 是否使用半精度训练
        num_features: 输出特征维度
    返回:
        MobileFaceNet模型
    """
    return MobileFaceNet(fp16, num_features)
