# networks.py - 音频姿态模型的基础网络组件定义
# 该模块定义了多个可复用的神经网络基础组件，主要包括：
# - ResidualConv: 残差卷积块，用于特征提取
# - Upsample: 转置卷积上采样模块
# - Squeeze_Excite_Block: 通道注意力机制（SE-Net）
# - ASPP: 空洞空间金字塔池化（Atrous Spatial Pyramid Pooling）
# - Upsample_: 双线性插值上采样模块
# - AttentionBlock: 注意力门控模块（用于U-Net的跳跃连接）

import torch.nn as nn
import torch


class ResidualConv(nn.Module):
    """
    残差卷积块：包含两条并行路径——主卷积路径和跳跃连接路径。
    
    主路径：BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3
    跳跃路径：Conv3x3（调整通道数和空间尺寸）
    
    最终输出 = 主路径输出 + 跳跃路径输出
    这种设计有助于缓解深层网络的梯度消失问题。
    
    参数说明：
        input_dim: 输入通道数
        output_dim: 输出通道数
        stride: 卷积步长（用于控制空间尺寸变化）
        padding: 填充大小
    """
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        # 主卷积路径：两个3x3卷积层 + BN + ReLU
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),          # 批归一化
            nn.ReLU(),                          # 激活函数
            nn.Conv2d(                           # 第一个3x3卷积
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),          # 批归一化
            nn.ReLU(),                          # 激活函数
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),  # 第二个3x3卷积
        )
        # 跳跃连接路径：1x1或3x3卷积调整通道数和空间尺寸
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        # 主路径输出 + 跳跃连接输出
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    """
    转置卷积上采样模块。
    
    使用转置卷积（反卷积）将特征图的空间尺寸放大。
    常用于U-Net解码器中，逐步恢复特征图的分辨率。
    
    参数说明：
        input_dim: 输入通道数
        output_dim: 输出通道数
        kernel: 转置卷积核大小
        stride: 转置卷积步长（决定上采样倍率）
    """
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        # 转置卷积层
        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    """
    通道注意力模块（Squeeze-and-Excitation Block）。
    
    通过自适应地重新校准通道间的特征响应来提升网络表达能力。
    
    工作流程：
    1. Squeeze（压缩）：用全局平均池化将每个通道压缩为一个标量
    2. Excitation（激励）：通过两层全连接网络学习通道间的关系
    3. Scale（缩放）：用Sigmoid输出的权重对原始特征进行逐通道缩放
    
    参数说明：
        channel: 输入特征图的通道数
        reduction: 降维比例，默认16（即中间层通道数 = channel/16）
    """
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        # 全局平均池化：将每个通道的空间维度压缩为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接网络：学习通道间的注意力权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),                                  # 激活
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid(),                                           # 输出[0,1]的权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()         # 获取batch size和通道数
        # Squeeze：全局平均池化 [b, c, h, w] -> [b, c]
        y = self.avg_pool(x).view(b, c)
        # Excitation：通过FC网络生成通道权重 [b, c] -> [b, c, 1, 1]
        y = self.fc(y).view(b, c, 1, 1)
        # Scale：逐通道加权
        return x * y.expand_as(x)


class ASPP(nn.Module):
    """
    空洞空间金字塔池化（Atrous Spatial Pyramid Pooling）。
    
    使用不同膨胀率（dilation rate）的空洞卷积并行处理输入，
    以捕获不同尺度的上下文信息。
    
    空洞卷积（Atrous/Dilated Convolution）通过在卷积核中插入"空洞"来
    扩大感受野，而不增加参数量或降低分辨率。
    
    参数说明：
        in_dims: 输入通道数
        out_dims: 输出通道数
        rate: 膨胀率列表，默认[6, 12, 18]，对应不同尺度的感受野
    """
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        # 三个不同膨胀率的空洞卷积分支
        # 膨胀率为6的空洞卷积
        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        # 膨胀率为12的空洞卷积
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        # 膨胀率为18的空洞卷积
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        # 1x1卷积：将三个分支的输出拼接后整合为输出通道数
        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        # 初始化权重
        self._init_weights()

    def forward(self, x):
        # 三个分支分别处理
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        # 在通道维度上拼接三个分支的结果
        out = torch.cat([x1, x2, x3], dim=1)
        # 1x1卷积整合
        return self.output(out)

    def _init_weights(self):
        """使用Kaiming初始化卷积权重，BatchNorm初始化为标准值"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    """
    双线性插值上采样模块。
    
    使用双线性插值将特征图放大指定倍率。
    相比转置卷积，双线性插值不会引入棋盘伪影（checkerboard artifacts）。
    
    参数说明：
        scale: 上采样倍率，默认2
    """
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        # 双线性插值上采样
        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    """
    注意力门控模块（Attention Gate）。
    
    用于U-Net的跳跃连接中，通过注意力机制让解码器关注
    编码器特征中最相关的区域，抑制不相关区域的特征。
    
    工作流程：
    1. 编码器特征经过卷积+池化得到注意力线索
    2. 解码器特征经过卷积得到门控信号
    3. 两者相加后通过1x1卷积生成注意力权重图
    4. 用注意力权重图对解码器特征进行加权
    
    参数说明：
        input_encoder: 编码器特征的通道数
        input_decoder: 解码器特征的通道数
        output_dim: 注意力特征的输出通道数
    """
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        # 编码器特征处理：BN -> ReLU -> Conv -> MaxPool
        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),             # 池化以匹配解码器特征的空间尺寸
        )

        # 解码器特征处理：BN -> ReLU -> Conv
        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        # 注意力权重生成：BN -> ReLU -> 1x1 Conv -> 输出单通道权重图
        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),    # 输出1通道的注意力图
        )

    def forward(self, x1, x2):
        """
        参数：
            x1: 编码器特征（跳跃连接传来的）
            x2: 解码器特征（上采样传来的）
        
        返回：
            注意力加权后的解码器特征
        """
        # 编码器特征和解码器特征相加
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        # 生成注意力权重图
        out = self.conv_attn(out)
        # 用注意力权重图对解码器特征进行逐元素加权
        return out * x2
