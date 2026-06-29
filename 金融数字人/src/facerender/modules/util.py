"""
util.py - 面部渲染工具模块
该模块提供了面部渲染系统中常用的各种基础组件，包括：
1. 关键点相关工具：关键点转高斯热图、坐标网格生成
2. 网络构建块：残差块、上下采样块、相同维度块（2D 和 3D 版本）
3. 沙漏网络（Hourglass）：编码器-解码器结构，用于特征提取和预测
4. 抗锯齿下采样：使用高斯核进行带限下采样
5. SPADE 归一化：空间自适应归一化层
6. 音频到图像的端到端模型

这些组件是整个面部渲染系统的基础设施，
被生成器、判别器、关键点检测器等多个模块广泛使用。
"""
from torch import nn

import torch.nn.functional as F
import torch

# 导入同步批归一化（支持多 GPU 训练时的统计量同步）
from src.facerender.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from src.facerender.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d

# 导入谱归一化（用于稳定对抗训练）
import torch.nn.utils.spectral_norm as spectral_norm


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    将关键点转换为高斯热力图表示。
    
    每个关键点在其位置周围生成一个高斯分布的热力图，
    距离关键点越近的像素值越高。这用于：
    1. 在稠密运动网络中表示关键点的影响范围
    2. 作为注意力权重指导特征变形
    
    参数:
        kp (dict): 关键点字典，包含 'value' 键，值为 (bs, num_kp, 3) 的坐标
        spatial_size (tuple): 输出热力图的空间尺寸 (d, h, w)
        kp_variance (float): 高斯核的方差，控制热力图的扩散程度
    
    返回:
        Tensor: 高斯热力图，形状 (bs, num_kp, d, h, w)
    """
    # 提取关键点坐标作为高斯均值
    mean = kp['value']

    # 创建 3D 坐标网格，范围 [-1, 1]
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    # 根据关键点的维度数量添加前导维度
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    # 扩展坐标网格以匹配 batch 和关键点维度
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    # 将关键点坐标重塑为 (bs, num_kp, 1, 1, 1, 3)
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    # 计算每个位置到关键点的欧氏距离的平方
    mean_sub = (coordinate_grid - mean)

    # 应用高斯函数：exp(-0.5 * ||x - mean||^2 / variance)
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def make_coordinate_grid_2d(spatial_size, type):
    """
    创建 2D 坐标网格，范围 [-1, 1] x [-1, 1]。
    
    参数:
        spatial_size (tuple): 网格尺寸 (h, w)
        type: 数据类型（如 torch.float32）
    
    返回:
        Tensor: 2D 坐标网格，形状 (h, w, 2)
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    # 将坐标归一化到 [-1, 1] 范围
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    # 扩展为 2D 网格
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    # 拼接 x 和 y 坐标
    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


def make_coordinate_grid(spatial_size, type):
    """
    创建 3D 坐标网格，范围 [-1, 1] x [-1, 1] x [-1, 1]。
    
    用于在 3D 空间中表示位置，为稠密运动网络和关键点检测器提供坐标参考。
    
    参数:
        spatial_size (tuple): 网格尺寸 (d, h, w)
        type: 数据类型
    
    返回:
        Tensor: 3D 坐标网格，形状 (d, h, w, 3)
    """
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)

    # 将坐标归一化到 [-1, 1] 范围
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
   
    # 扩展为 3D 网格
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)
    # 拼接 x, y, z 坐标
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed


class ResBottleneck(nn.Module):
    """
    残差瓶颈块（Residual Bottleneck Block）。
    
    用于头部姿态估计器（HEEstimator）的骨干网络。
    结构：1x1 降维 -> 3x3 卷积（可选步长）-> 1x1 升维 + 残差连接
    
    当步长不为 1 时，跳跃连接使用 1x1 卷积调整维度。
    """

    def __init__(self, in_features, stride):
        """
        参数:
            in_features (int): 输入特征通道数
            stride (int): 3x3 卷积的步长（用于下采样）
        """
        super(ResBottleneck, self).__init__()
        # 瓶颈结构：先降维到 1/4，再升维回原始通道数
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features//4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(in_channels=in_features//4, out_channels=in_features, kernel_size=1)
        # 三个卷积层各带一个批归一化
        self.norm1 = BatchNorm2d(in_features//4, affine=True)
        self.norm2 = BatchNorm2d(in_features//4, affine=True)
        self.norm3 = BatchNorm2d(in_features, affine=True)

        self.stride = stride
        # 当步长不为 1 时，需要对跳跃连接进行下采样和维度匹配
        if self.stride != 1:
            self.skip = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=stride)
            self.norm4 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        """前向传播。"""
        # 主路径：conv1 -> BN -> ReLU -> conv2 -> BN -> ReLU -> conv3 -> BN
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        # 跳跃连接（如果步长不为 1，需要下采样）
        if self.stride != 1:
            x = self.skip(x)
            x = self.norm4(x)
        # 残差相加 + ReLU 激活
        out += x
        out = F.relu(out)
        return out


class ResBlock2d(nn.Module):
    """
    2D 残差块，保持空间分辨率不变。
    
    结构：BN -> ReLU -> Conv -> BN -> ReLU -> Conv + 残差连接
    使用预激活（pre-activation）结构，先归一化再卷积。
    """

    def __init__(self, in_features, kernel_size, padding):
        """
        参数:
            in_features (int): 输入/输出特征通道数
            kernel_size (int): 卷积核大小
            padding (int): 填充大小
        """
        super(ResBlock2d, self).__init__()
        # 两个 2D 卷积层
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        # 两个批归一化层
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        """前向传播：预激活残差块。"""
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        # 残差连接
        out += x
        return out


class ResBlock3d(nn.Module):
    """
    3D 残差块，保持空间分辨率不变。
    
    结构与 ResBlock2d 类似，但使用 3D 卷积和 3D 批归一化，
    用于处理带有深度维度的 3D 特征图。
    """

    def __init__(self, in_features, kernel_size, padding):
        """
        参数:
            in_features (int): 输入/输出特征通道数
            kernel_size (int): 卷积核大小
            padding (int): 填充大小
        """
        super(ResBlock3d, self).__init__()
        # 两个 3D 卷积层
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        # 两个 3D 批归一化层
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        """前向传播：预激活 3D 残差块。"""
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        # 残差连接
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    2D 上采样块，用于解码器中恢复空间分辨率。
    
    结构：双线性上采样(2x) -> 卷积 -> BN -> ReLU
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        """
        参数:
            in_features (int): 输入特征通道数
            out_features (int): 输出特征通道数
            kernel_size (int): 卷积核大小
            padding (int): 填充大小
            groups (int): 分组卷积的组数
        """
        super(UpBlock2d, self).__init__()

        # 2D 卷积层
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        """前向传播：上采样 -> 卷积 -> 归一化 -> 激活。"""
        # 双线性插值上采样 2 倍
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class UpBlock3d(nn.Module):
    """
    3D 上采样块，用于 3D 解码器中恢复空间分辨率。
    
    结构：三线性上采样(深度不变，空间 2x) -> 3D 卷积 -> BN -> ReLU
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        """
        参数:
            in_features (int): 输入特征通道数
            out_features (int): 输出特征通道数
            kernel_size (int): 卷积核大小
            padding (int): 填充大小
            groups (int): 分组卷积的组数
        """
        super(UpBlock3d, self).__init__()

        # 3D 卷积层
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        """前向传播：3D 上采样 -> 卷积 -> 归一化 -> 激活。"""
        # 三线性插值：深度维度不变，空间维度 2 倍上采样
        # out = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear')
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    2D 下采样块，用于编码器中降低空间分辨率。
    
    结构：卷积 -> BN -> ReLU -> 平均池化(2x)
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        """
        参数:
            in_features (int): 输入特征通道数
            out_features (int): 输出特征通道数
            kernel_size (int): 卷积核大小
            padding (int): 填充大小
            groups (int): 分组卷积的组数
        """
        super(DownBlock2d, self).__init__()
        # 2D 卷积层
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        # 2x2 平均池化进行下采样
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        """前向传播：卷积 -> 归一化 -> 激活 -> 池化。"""
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class DownBlock3d(nn.Module):
    """
    3D 下采样块，用于 3D 编码器中降低空间分辨率。
    
    结构：3D 卷积 -> BN -> ReLU -> 3D 平均池化(空间 2x 下采样，深度不变)
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        """
        参数:
            in_features (int): 输入特征通道数
            out_features (int): 输出特征通道数
            kernel_size (int): 卷积核大小
            padding (int): 填充大小
            groups (int): 分组卷积的组数
        """
        super(DownBlock3d, self).__init__()
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups, stride=(1, 2, 2))
        '''
        # 3D 卷积层（不使用步长下采样）
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)
        # 3D 平均池化：深度维度不变，空间维度 2 倍下采样
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        """前向传播：卷积 -> 归一化 -> 激活 -> 池化。"""
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    2D 同维度块，保持空间分辨率不变。
    
    结构：卷积 -> BN -> 激活函数
    用于编码器的第一层和解码器的中间层。
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, lrelu=False):
        """
        参数:
            in_features (int): 输入特征通道数
            out_features (int): 输出特征通道数
            groups (int): 分组卷积的组数
            kernel_size (int): 卷积核大小
            padding (int): 填充大小
            lrelu (bool): 是否使用 LeakyReLU（否则使用 ReLU）
        """
        super(SameBlock2d, self).__init__()
        # 2D 卷积层
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        # 激活函数选择
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        """前向传播：卷积 -> 归一化 -> 激活。"""
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out


class Encoder(nn.Module):
    """
    沙漏网络的编码器部分。
    
    由多个 3D 下采样块组成，逐步降低空间分辨率、增加通道数，
    提取多尺度特征用于后续的解码器和跳跃连接。
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        """
        参数:
            block_expansion (int): 通道扩展因子
            in_features (int): 输入特征通道数
            num_blocks (int): 下采样块的数量
            max_features (int): 最大特征通道数
        """
        super(Encoder, self).__init__()

        # 构建多个 3D 下采样块
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        """
        前向传播：逐层编码，保存每层输出用于跳跃连接。
        
        参数:
            x (Tensor): 输入 3D 特征
        
        返回:
            list: 每层的输出特征列表（从粗到细）
        """
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    沙漏网络的解码器部分。
    
    由多个 3D 上采样块组成，逐步恢复空间分辨率。
    每层与编码器的对应层进行跳跃连接（拼接），融合多尺度特征。
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        """
        参数:
            block_expansion (int): 通道扩展因子
            in_features (int): 输入特征通道数
            num_blocks (int): 上采样块的数量
            max_features (int): 最大特征通道数
        """
        super(Decoder, self).__init__()

        up_blocks = []

        # 逆序构建上采样块，从最深层开始
        for i in range(num_blocks)[::-1]:
            # 输入通道数 = 上采样块的输出 + 编码器对应层的跳跃连接通道数
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        # 输出通道数 = 最终上采样块的输出 + 原始输入通道数
        # self.out_filters = block_expansion
        self.out_filters = block_expansion + in_features

        # 最终的 3D 卷积和归一化
        self.conv = nn.Conv3d(in_channels=self.out_filters, out_channels=self.out_filters, kernel_size=3, padding=1)
        self.norm = BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x):
        """
        前向传播：逐层解码并与编码器特征进行跳跃连接。
        
        参数:
            x (list): 编码器输出的多尺度特征列表
        
        返回:
            Tensor: 解码后的 3D 特征
        """
        out = x.pop()  # 取出最深层的特征
        # 逐层上采样并与跳跃连接拼接
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()  # 取出编码器对应层的特征
            out = torch.cat([out, skip], dim=1)  # 通道维度拼接
        # out = self.up_blocks[-1](out)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Hourglass(nn.Module):
    """
    沙漏网络（Hourglass Network）。
    
    一种对称的编码器-解码器结构，通过跳跃连接融合多尺度特征。
    广泛用于关键点检测、姿态估计等任务中。
    
    在稠密运动网络中用于从变形特征预测运动权重和遮挡图。
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        """
        参数:
            block_expansion (int): 通道扩展因子
            in_features (int): 输入特征通道数
            num_blocks (int): 编码/解码层数
            max_features (int): 最大特征通道数
        """
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        """
        前向传播：编码 -> 解码。
        
        参数:
            x (Tensor): 输入特征
        
        返回:
            Tensor: 输出特征
        """
        return self.decoder(self.encoder(x))


class KPHourglass(nn.Module):
    """
    关键点专用沙漏网络。
    
    与标准沙漏网络类似，但使用 2D 下采样和 3D 上采样，
    用于将 2D 图像特征转换为 3D 空间特征。
    这是关键点检测器的核心特征提取器。
    """ 

    def __init__(self, block_expansion, in_features, reshape_features, reshape_depth, num_blocks=3, max_features=256):
        """
        参数:
            block_expansion (int): 通道扩展因子
            in_features (int): 输入特征通道数（图像通道数）
            reshape_features (int): 重塑后的特征通道数
            reshape_depth (int): 重塑后的深度维度
            num_blocks (int): 下采样块数量
            max_features (int): 最大特征通道数
        """
        super(KPHourglass, self).__init__()
        
        # 下采样阶段：使用 2D 下采样块
        self.down_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.down_blocks.add_module('down'+ str(i), DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                                                   min(max_features, block_expansion * (2 ** (i + 1))),
                                                                   kernel_size=3, padding=1))

        # 1x1 卷积调整通道数并重塑为 3D 特征
        in_filters = min(max_features, block_expansion * (2 ** num_blocks))
        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=reshape_features, kernel_size=1)

        # 上采样阶段：使用 3D 上采样块
        self.up_blocks = nn.Sequential()
        for i in range(num_blocks):
            in_filters = min(max_features, block_expansion * (2 ** (num_blocks - i)))
            out_filters = min(max_features, block_expansion * (2 ** (num_blocks - i - 1)))
            self.up_blocks.add_module('up'+ str(i), UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.reshape_depth = reshape_depth
        self.out_filters = out_filters

    def forward(self, x):
        """
        前向传播：2D 下采样 -> 重塑为 3D -> 3D 上采样。
        
        参数:
            x (Tensor): 输入 2D 图像特征 (bs, C, H, W)
        
        返回:
            Tensor: 输出 3D 特征 (bs, C', D', H', W')
        """
        # 2D 下采样提取特征
        out = self.down_blocks(x)
        # 1x1 卷积调整通道数
        out = self.conv(out)
        bs, c, h, w = out.shape
        # 将 2D 特征重塑为 3D 特征：通道维度拆分为 (C/depth, depth)
        out = out.view(bs, c//self.reshape_depth, self.reshape_depth, h, w)
        # 3D 上采样恢复空间分辨率
        out = self.up_blocks(out)

        return out
        

class AntiAliasInterpolation2d(nn.Module):
    """
    抗锯齿 2D 下采样模块。
    
    使用高斯核进行带限下采样（band-limited downsampling），
    在下采样前进行低通滤波以避免混叠伪影。
    
    这比简单的双线性插值或最近邻下采样能更好地保留输入信号，
    特别是在处理面部图像时能减少锯齿和模糊。
    """
    def __init__(self, channels, scale):
        """
        参数:
            channels (int): 输入图像通道数
            scale (float): 下采样倍率（如 0.5 表示缩小一半）
        """
        super(AntiAliasInterpolation2d, self).__init__()
        # 根据缩放因子计算高斯核的标准差和大小
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        # 计算填充大小
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # 构建 2D 高斯核
        # 高斯核是每个维度的高斯函数的乘积
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # 归一化使核的元素和为 1
        kernel = kernel / torch.sum(kernel)
        # 重塑为深度卷积权重格式
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        # 注册为缓冲区（不参与梯度计算，但会随模型保存/加载）
        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        """
        前向传播：使用高斯核进行抗锯齿下采样。
        
        参数:
            input (Tensor): 输入特征图，形状 (bs, C, H, W)
        
        返回:
            Tensor: 下采样后的特征图
        """
        if self.scale == 1.0:
            return input

        # 填充以保持输出尺寸
        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        # 使用分组卷积应用高斯核（每通道独立滤波）
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        # 步长下采样（跳过像素）
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


class SPADE(nn.Module):
    """
    SPADE（Spatially-Adaptive Normalization）层。
    
    一种空间自适应的归一化方法，根据语义分割图（条件输入）
    动态计算归一化的缩放（gamma）和偏移（beta）参数。
    
    与传统的 BatchNorm/InstanceNorm 不同，SPADE 的归一化参数
    是空间变化的，能根据输入的语义信息自适应调整。
    
    在 SPADEDecoder 中用于保持生成图像的空间结构和语义一致性。
    """
    def __init__(self, norm_nc, label_nc):
        """
        参数:
            norm_nc (int): 需要归一化的特征通道数
            label_nc (int): 条件标签/语义图的通道数
        """
        super().__init__()

        # 无参数的实例归一化（只做标准化，不学习仿射参数）
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128  # 中间隐藏层通道数

        # 共享的特征提取层：将语义图转换为特征
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        # 预测 gamma（缩放因子）和 beta（偏移量）的卷积层
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        """
        前向传播：使用 SPADE 进行空间自适应归一化。
        
        参数:
            x (Tensor): 输入特征，形状 (bs, norm_nc, H, W)
            segmap (Tensor): 条件语义图，形状 (bs, label_nc, H', W')
        
        返回:
            Tensor: 归一化后的特征，形状与输入相同
        """
        # 先进行标准的实例归一化
        normalized = self.param_free_norm(x)
        # 将语义图插值到与输入特征相同的空间尺寸
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        # 从语义图提取特征
        actv = self.mlp_shared(segmap)
        # 预测空间自适应的 gamma 和 beta
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # 应用自适应归一化：out = normalized * (1 + gamma) + beta
        out = normalized * (1 + gamma) + beta
        return out
    

class SPADEResnetBlock(nn.Module):
    """
    基于 SPADE 的残差块。
    
    结构与标准残差块类似，但使用 SPADE 替代标准的归一化层，
    使归一化过程能根据条件输入（语义图）自适应调整。
    
    支持可选的谱归一化和空洞卷积。
    """
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        """
        参数:
            fin (int): 输入特征通道数
            fout (int): 输出特征通道数
            norm_G (str): 归一化类型（包含 'spectral' 则使用谱归一化）
            label_nc (int): 条件标签通道数
            use_se (bool): 是否使用 SE 注意力模块
            dilation (int): 空洞卷积的膨胀率
        """
        super().__init__()
        # 属性设置
        self.learned_shortcut = (fin != fout)  # 输入输出通道不同时使用学习的跳跃连接
        fmiddle = min(fin, fout)  # 中间层通道数
        self.use_se = use_se
        # 创建卷积层
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        # 如果输入输出通道不同，需要 1x1 卷积调整维度
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # 如果指定使用谱归一化，应用到所有卷积层
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # 定义 SPADE 归一化层
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        """
        前向传播：SPADE 残差块。
        
        参数:
            x (Tensor): 输入特征
            seg1 (Tensor): 条件语义图
        
        返回:
            Tensor: 输出特征 = 跳跃连接 + 残差
        """
        x_s = self.shortcut(x, seg1)
        # 主路径：SPADE -> 激活 -> 卷积 -> SPADE -> 激活 -> 卷积
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        """跳跃连接：如果维度不匹配则使用学习的 1x1 卷积。"""
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        """LeakyReLU 激活函数，斜率 0.2。"""
        return F.leaky_relu(x, 2e-1)

class audio2image(nn.Module):
    """
    音频到图像的端到端模型。
    
    将音频信号直接驱动面部图像生成，不需要显式的语义特征提取。
    结合了视频头部姿态估计器、音频头部姿态估计器、
    关键点检测器和图像生成器。
    
    训练时可以同时优化所有子网络。
    """

    def __init__(self, generator, kp_extractor, he_estimator_video, he_estimator_audio, train_params):
        """
        参数:
            generator: 图像生成器
            kp_extractor: 关键点检测器
            he_estimator_video: 视频头部姿态估计器
            he_estimator_audio: 音频头部姿态估计器
            train_params: 训练参数
        """
        super().__init__()
        # 属性设置
        self.generator = generator
        self.kp_extractor = kp_extractor
        self.he_estimator_video = he_estimator_video
        self.he_estimator_audio = he_estimator_audio
        self.train_params = train_params

    def headpose_pred_to_degree(self, pred):
        """
        将头部姿态的分类预测转换为角度值。
        与 make_animation.py 中的同名函数功能相同。
        """
        device = pred.device
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)
        pred = F.softmax(pred)
        degree = torch.sum(pred*idx_tensor, 1) * 3 - 99

        return degree
    
    def get_rotation_matrix(self, yaw, pitch, roll):
        """
        根据欧拉角生成 3D 旋转矩阵。
        与 make_animation.py 中的同名函数功能相同。
        """
        yaw = yaw / 180 * 3.14
        pitch = pitch / 180 * 3.14
        roll = roll / 180 * 3.14

        roll = roll.unsqueeze(1)
        pitch = pitch.unsqueeze(1)
        yaw = yaw.unsqueeze(1)

        # 绕 Z 轴（翻滚）的旋转矩阵
        roll_mat = torch.cat([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), 
                          torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                          torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1)
        roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

        # 绕 X 轴（俯仰）的旋转矩阵
        pitch_mat = torch.cat([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch), 
                           torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                           -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1)
        pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

        # 绕 Y 轴（偏航）的旋转矩阵
        yaw_mat = torch.cat([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),  
                         torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                         torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
        yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

        # 组合旋转矩阵
        rot_mat = torch.einsum('bij,bjk,bkm->bim', roll_mat, pitch_mat, yaw_mat)

        return rot_mat

    def keypoint_transformation(self, kp_canonical, he):
        """
        对规范关键点应用头部姿态和表情变换。
        与 make_animation.py 中的同名函数功能相同。
        """
        kp = kp_canonical['value']    # (bs, k, 3)
        yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
        t, exp = he['t'], he['exp']
    
        yaw = self.headpose_pred_to_degree(yaw)
        pitch = self.headpose_pred_to_degree(pitch)
        roll = self.headpose_pred_to_degree(roll)

        rot_mat = self.get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)
    
        # 关键点旋转
        kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    
        # 关键点平移
        t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
        kp_t = kp_rotated + t

        # 叠加表情偏移
        exp = exp.view(exp.shape[0], -1, 3)
        kp_transformed = kp_t + exp

        return {'value': kp_transformed}

    def forward(self, source_image, target_audio):
        """
        前向传播：从源图像和目标音频生成面部动画帧。
        
        参数:
            source_image (Tensor): 源面部图像
            target_audio (Tensor): 目标音频特征
        
        返回:
            dict: 生成器输出，包含 'prediction' 等键
        """
        # 使用视频估计器提取源图像的头部姿态
        pose_source = self.he_estimator_video(source_image)
        # 使用音频估计器从目标音频提取驱动姿态
        pose_generated = self.he_estimator_audio(target_audio)
        # 检测源图像的关键点
        kp_canonical = self.kp_extractor(source_image)
        # 对源关键点应用源姿态变换
        kp_source = self.keypoint_transformation(kp_canonical, pose_source)
        # 对源关键点应用驱动姿态变换
        kp_transformed_generated = self.keypoint_transformation(kp_canonical, pose_generated)
        # 使用生成器合成新帧
        generated = self.generator(source_image, kp_source=kp_source, kp_driving=kp_transformed_generated)
        return generated