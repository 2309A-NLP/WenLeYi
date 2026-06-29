# discriminator.py - 姿态序列判别器网络定义
# 该模块定义了用于判断姿态运动序列真实性的判别器
# 判别器接收姿态运动序列，输出每个时间步的真实性评分
# 用于CVAE的对抗训练（adversarial training），提升生成姿态的自然性和连贯性

import torch
import torch.nn.functional as F
from torch import nn

class ConvNormRelu(nn.Module):
    """
    卷积 + 归一化 + 激活函数的组合模块。
    
    支持1D和2D两种卷积类型，适用于不同维度的输入数据。
    1D卷积用于处理序列数据（如姿态运动序列），
    2D卷积用于处理图像数据（如梅尔频谱图）。
    
    参数说明：
        conv_type: 卷积类型，'1d'或'2d'
        in_channels: 输入通道数
        out_channels: 输出通道数
        downsample: 是否使用下采样（stride=2）
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        norm: 归一化类型，'BN'（批归一化）或'IN'（实例归一化）
        leaky: 是否使用LeakyReLU（True）或ReLU（False）
    """
    def __init__(self, conv_type='1d', in_channels=3, out_channels=64, downsample=False,
                 kernel_size=None, stride=None, padding=None, norm='BN', leaky=False):
        super().__init__()
        # 设置默认的卷积核大小、步长和填充
        if kernel_size is None:
            if downsample:
                # 下采样模式：使用较大的步长
                kernel_size, stride, padding = 4, 2, 1
            else:
                # 非下采样模式：保持空间尺寸不变
                kernel_size, stride, padding = 3, 1, 1

        if conv_type == '2d':
            # 2D卷积层（用于图像数据）
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
            # 2D归一化层
            if norm == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == 'IN':
                self.norm = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError
        elif conv_type == '1d':
            # 1D卷积层（用于序列数据）
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
            # 1D归一化层
            if norm == 'BN':
                self.norm = nn.BatchNorm1d(out_channels)
            elif norm == 'IN':
                self.norm = nn.InstanceNorm1d(out_channels)
            else:
                raise NotImplementedError
        
        # Kaiming初始化卷积权重，适合ReLU激活函数
        nn.init.kaiming_normal_(self.conv.weight)

        # 选择激活函数：LeakyReLU或ReLU
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False) if leaky else nn.ReLU(inplace=True)

    def forward(self, x):
        # 卷积操作
        x = self.conv(x)
        # 对于InstanceNorm1d，需要调整维度以正确应用归一化
        # InstanceNorm1d期望输入维度为 [N, C, L]，但1D卷积输出也是这个格式
        # 这里permute是为了在通道维度上进行归一化
        if isinstance(self.norm, nn.InstanceNorm1d):
            x = self.norm(x.permute((0, 2, 1))).permute((0, 2, 1))  # normalize on [C]
        else:
            x = self.norm(x)
        # 激活函数
        x = self.act(x)
        return x


class PoseSequenceDiscriminator(nn.Module):
    """
    姿态序列判别器：判断输入的姿态运动序列是真实的还是生成的。
    
    判别器采用1D卷积网络，对姿态运动序列进行逐时间步的真伪判断。
    输入为姿态运动序列，输出为每个时间步的判别分数（真/假概率）。
    
    网络结构：
    - 两次下采样：将序列长度减半两次
    - 一次保持尺寸的卷积：提取高维特征
    - 最后用1x1卷积映射为单通道输出
    
    判别器在GAN训练中与生成器对抗：
    - 生成器努力生成"骗过"判别器的姿态运动
    - 判别器努力区分真实和生成的姿态运动
    - 两者交替训练，最终生成器能生成逼真的姿态运动
    
    参数说明：
        cfg: 配置文件对象
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 获取LeakyReLU的配置参数
        leaky = self.cfg.MODEL.DISCRIMINATOR.LEAKY_RELU

        # 判别器网络：3层1D卷积 + 1层输出卷积
        self.seq = nn.Sequential(
            # 第一层：下采样，输入通道 -> 256通道，序列长度减半
            # B, 256, 64
            ConvNormRelu('1d', cfg.MODEL.DISCRIMINATOR.INPUT_CHANNELS, 256, downsample=True, leaky=leaky),
            # 第二层：下采样，256通道 -> 512通道，序列长度再减半
            # B, 512, 32
            ConvNormRelu('1d', 256, 512, downsample=True, leaky=leaky),
            # 第三层：保持尺寸，512通道 -> 1024通道
            # B, 1024, 16
            ConvNormRelu('1d', 512, 1024, kernel_size=3, stride=1, padding=1, leaky=leaky),
            # 输出层：1024通道 -> 1通道，输出每个时间步的判别分数
            # B, 1, 16
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        """
        判别器的前向传播。
        
        参数：
            x: 输入姿态运动序列，形状 [bs, seq_len, 6]
        
        返回：
            判别分数，形状 [bs, seq_len]，每个时间步一个分数
        """
        # 将输入reshape为1D卷积所需的格式：[bs, seq_len, 6] -> [bs, 6, seq_len]
        # 先展平batch和seq_len维度，再转置
        x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
        # 通过判别器网络
        x = self.seq(x)
        # 去掉通道维度：[bs, 1, seq_len] -> [bs, seq_len]
        x = x.squeeze(1)
        return x
