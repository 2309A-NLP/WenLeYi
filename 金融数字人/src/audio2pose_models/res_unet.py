# res_unet.py - 残差U-Net网络定义
# 该模块定义了一个基于残差连接的U-Net网络（ResUnet）
# ResUnet结合了U-Net的编码器-解码器结构和残差连接的优势
# 在SadTalker中用于姿态运动特征的编码和解码

import torch
import torch.nn as nn
from src.audio2pose_models.networks import ResidualConv, Upsample


class ResUnet(nn.Module):
    """
    残差U-Net（Residual U-Net）网络。
    
    U-Net是一种经典的编码器-解码器网络结构，最初用于医学图像分割。
    在本项目中，ResUnet被用于姿态运动特征的编码和解码。
    
    网络结构：
    - 编码器（下采样路径）：逐层减小空间尺寸，增加通道数
    - 桥接层（Bridge）：连接编码器和解码器的中间层
    - 解码器（上采样路径）：逐层恢复空间尺寸，减少通道数
    - 跳跃连接：编码器特征与解码器特征拼接，保留空间细节信息
    
    特殊设计：
    - 时间维度使用stride=(2,1)进行下采样，频率维度不变
    - 这种非对称下采样适合处理时间序列数据
    - 最终输出通过Sigmoid激活，将值限制在[0,1]范围
    
    参数说明：
        channel: 输入通道数（默认1）
        filters: 各层的通道数列表（默认[32, 64, 128, 256]）
    """
    def __init__(self, channel=1, filters=[32, 64, 128, 256]):
        super(ResUnet, self).__init__()

        # ===== 输入层 =====
        # 两个3x3卷积 + BN + ReLU + 残差连接
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        # 输入跳跃连接（1x1卷积调整通道数）
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        # ===== 编码器路径（下采样） =====
        # 第一个残差卷积块：32 -> 64通道，时间维度下采样2倍
        self.residual_conv_1 = ResidualConv(filters[0], filters[1], stride=(2,1), padding=1)
        # 第二个残差卷积块：64 -> 128通道，时间维度下采样2倍
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], stride=(2,1), padding=1)

        # ===== 桥接层 =====
        # 连接编码器和解码器：128 -> 256通道，时间维度下采样2倍
        self.bridge = ResidualConv(filters[2], filters[3], stride=(2,1), padding=1)

        # ===== 解码器路径（上采样） =====
        # 第一个上采样：转置卷积恢复时间维度
        self.upsample_1 = Upsample(filters[3], filters[3], kernel=(2,1), stride=(2,1))
        # 第一个残差卷积：融合上采样特征和编码器跳跃连接特征
        # 输入通道 = 256(上采样) + 128(跳跃连接) = 384
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], stride=1, padding=1)

        # 第二个上采样
        self.upsample_2 = Upsample(filters[2], filters[2], kernel=(2,1), stride=(2,1))
        # 第二个残差卷积
        # 输入通道 = 128 + 64 = 192
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], stride=1, padding=1)

        # 第三个上采样
        self.upsample_3 = Upsample(filters[1], filters[1], kernel=(2,1), stride=(2,1))
        # 第三个残差卷积
        # 输入通道 = 64 + 32 = 96
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], stride=1, padding=1)

        # ===== 输出层 =====
        # 1x1卷积将通道数降为1 + Sigmoid激活
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),    # 32通道 -> 1通道
            nn.Sigmoid(),                       # 输出范围[0,1]
        )

    def forward(self, x):
        """
        前向传播。
        
        参数：
            x: 输入张量，形状 [bs, 1, seq_len, feature_dim]
        
        返回：
            output: 输出张量，形状 [bs, 1, seq_len, feature_dim]
        """
        # ===== 编码阶段 =====
        # 输入层 + 跳跃连接（残差）
        x1 = self.input_layer(x) + self.input_skip(x)    # [bs, 32, seq_len, 6]
        # 第一次下采样
        x2 = self.residual_conv_1(x1)                     # [bs, 64, seq_len/2, 6]
        # 第二次下采样
        x3 = self.residual_conv_2(x2)                     # [bs, 128, seq_len/4, 6]
        
        # ===== 桥接阶段 =====
        # 第三次下采样（编码器最深层）
        x4 = self.bridge(x3)                              # [bs, 256, seq_len/8, 6]

        # ===== 解码阶段 =====
        # 第一次上采样 + 跳跃连接
        x4 = self.upsample_1(x4)                          # [bs, 256, seq_len/4, 6]
        x5 = torch.cat([x4, x3], dim=1)                  # [bs, 384, seq_len/4, 6]

        # 第一个上采样残差卷积
        x6 = self.up_residual_conv1(x5)                   # [bs, 128, seq_len/4, 6]

        # 第二次上采样 + 跳跃连接
        x6 = self.upsample_2(x6)                          # [bs, 128, seq_len/2, 6]
        x7 = torch.cat([x6, x2], dim=1)                  # [bs, 192, seq_len/2, 6]

        # 第二个上采样残差卷积
        x8 = self.up_residual_conv2(x7)                   # [bs, 64, seq_len/2, 6]

        # 第三次上采样 + 跳跃连接
        x8 = self.upsample_3(x8)                          # [bs, 64, seq_len, 6]
        x9 = torch.cat([x8, x1], dim=1)                  # [bs, 96, seq_len, 6]

        # 第三个上采样残差卷积
        x10 = self.up_residual_conv3(x9)                  # [bs, 32, seq_len, 6]

        # 输出层：32通道 -> 1通道 + Sigmoid
        output = self.output_layer(x10)                   # [bs, 1, seq_len, 6]

        return output
