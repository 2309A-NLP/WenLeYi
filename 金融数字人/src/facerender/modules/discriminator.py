"""
discriminator.py - 判别器网络模块
该模块实现了用于对抗训练的判别器网络，类似于 Pix2Pix 的 PatchGAN 架构。
判别器的任务是区分真实图像和生成器生成的假图像，
在训练过程中帮助生成器产生更逼真的面部图像。
支持多尺度判别，可在不同分辨率下评估图像真实性。
"""
from torch import nn
import torch.nn.functional as F
# 导入关键点转高斯热图的工具函数（用于可视化或特征提取）
from facerender.modules.util import kp2gaussian
import torch


class DownBlock2d(nn.Module):
    """
    2D 下采样块，用于判别器的编码器部分。
    
    每个下采样块包含：卷积 -> [可选的实例归一化] -> LeakyReLU 激活 -> [可选的平均池化]
    通过逐步降低空间分辨率、增加通道数来提取图像特征。
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        """
        参数:
            in_features (int): 输入特征通道数
            out_features (int): 输出特征通道数
            norm (bool): 是否使用实例归一化
            kernel_size (int): 卷积核大小，默认 4
            pool (bool): 是否在卷积后进行平均池化下采样
            sn (bool): 是否使用谱归一化（用于稳定对抗训练）
        """
        super(DownBlock2d, self).__init__()
        # 2D 卷积层
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        # 可选的谱归一化：约束权重矩阵的谱范数，防止判别器过强
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        # 可选的实例归一化层
        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        """
        前向传播。
        参数:
            x (Tensor): 输入特征图，形状 (bs, in_features, H, W)
        返回:
            Tensor: 输出特征图，经过卷积、归一化、激活和可选池化
        """
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        # LeakyReLU 激活函数，斜率 0.2，防止梯度消失
        out = F.leaky_relu(out, 0.2)
        # 可选的 2x2 平均池化下采样
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out


class Discriminator(nn.Module):
    """
    判别器网络，类似于 Pix2Pix 的 PatchGAN 判别器。
    
    结构：
    - 多个 DownBlock2d 组成的编码器，逐步下采样
    - 最终 1x1 卷积输出单通道预测图（每个位置对应一个 patch 的真/假判断）
    
    PatchGAN 的优点是只关注局部图像块的真实性，可以更好地保持局部细节。
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, **kwargs):
        """
        参数:
            num_channels (int): 输入图像通道数，默认 3（RGB）
            block_expansion (int): 第一层的通道扩展数
            num_blocks (int): 下采样块的数量
            max_features (int): 最大特征通道数限制
            sn (bool): 是否使用谱归一化
        """
        super(Discriminator, self).__init__()

        # 构建多个下采样块
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = nn.ModuleList(down_blocks)
        # 最终 1x1 卷积将多通道特征映射为单通道预测图
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        """
        前向传播。
        
        参数:
            x (Tensor): 输入图像，形状 (bs, C, H, W)
        
        返回:
            tuple: (feature_maps, prediction_map)
                - feature_maps: 各层的特征图列表（用于计算感知损失）
                - prediction_map: 最终的真/假预测图
        """
        feature_maps = []
        out = x

        # 逐层提取特征，记录每层输出用于特征匹配损失
        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        # 最终 1x1 卷积输出判别结果
        prediction_map = self.conv(out)

        return feature_maps, prediction_map


class MultiScaleDiscriminator(nn.Module):
    """
    多尺度判别器。
    
    在多个图像分辨率尺度上分别使用独立的判别器，
    使判别器能够同时关注全局结构和局部细节。
    这种设计在图像生成质量上通常优于单尺度判别器。
    """

    def __init__(self, scales=(), **kwargs):
        """
        参数:
            scales (tuple): 每个尺度的缩放因子，如 (1.0, 0.5, 0.25)
            **kwargs: 传递给每个子判别器的参数
        """
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        # 为每个尺度创建一个独立的判别器
        for scale in scales:
            # 将缩放因子中的 '.' 替换为 '-' 作为字典键名
            discs[str(scale).replace('.', '-')] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x):
        """
        前向传播：对每个尺度的输入分别进行判别。
        
        参数:
            x (dict): 包含不同尺度图像的字典，键为 'prediction_1.0', 'prediction_0.5' 等
        
        返回:
            dict: 包含各尺度的特征图和预测图
                - 'feature_maps_scale': 各尺度的中间特征图
                - 'prediction_map_scale': 各尺度的判别预测图
        """
        out_dict = {}
        for scale, disc in self.discs.items():
            # 将键名中的 '-' 还原为 '.'
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            # 使用对应尺度的判别器处理图像
            feature_maps, prediction_map = disc(x[key])
            # 保存结果，键名包含尺度信息
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict
