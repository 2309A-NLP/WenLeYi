"""
keypoint_detector.py - 关键点检测器和头部姿态估计器模块
该模块包含两个核心网络：
1. KPDetector：从输入图像中检测面部关键点的位置和局部雅可比矩阵
2. HEEstimator：估计头部姿态（yaw/pitch/roll）、平移向量和表情参数

这两个网络是 SadTalker 系统的核心组件，
它们将输入图像分解为关键点位置和头部姿态/表情参数，
用于驱动目标面部的动画生成。
"""
from torch import nn
import torch
import torch.nn.functional as F

# 导入同步批归一化（支持多 GPU 训练）
from src.facerender.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
# 导入工具函数：关键点沙漏网络、坐标网格生成、抗锯齿下采样、残差瓶颈块
from src.facerender.modules.util import KPHourglass, make_coordinate_grid, AntiAliasInterpolation2d, ResBottleneck


class KPDetector(nn.Module):
    """
    关键点检测器。
    
    检测输入图像中的规范关键点（canonical keypoints），
    返回每个关键点的位置（3D 坐标）和局部雅可比矩阵。
    
    工作原理：
    1. 使用沙漏网络（KPHourglass）提取多尺度特征
    2. 通过 3D 卷积预测关键点热力图
    3. 对热力图进行 softmax 归一化
    4. 计算热力图的加权平均得到关键点坐标
    5. 可选地估计每个关键点的 3x3 雅可比矩阵（描述局部形变）
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, reshape_channel, reshape_depth,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1, single_jacobian_map=False):
        """
        参数:
            block_expansion (int): 沙漏网络的通道扩展因子
            feature_channel (int): 特征通道数
            num_kp (int): 检测的关键点数量
            image_channel (int): 输入图像通道数
            max_features (int): 最大特征通道数
            reshape_channel (int): 重塑后的通道数
            reshape_depth (int): 重塑后的深度维度
            num_blocks (int): 沙漏网络的块数
            temperature (float): softmax 温度参数，控制热力图的集中程度
            estimate_jacobian (bool): 是否估计雅可比矩阵
            scale_factor (float): 输入图像的缩放因子
            single_jacobian_map (bool): 是否使用单一雅可比图（所有关键点共享）
        """
        super(KPDetector, self).__init__()

        # 核心特征提取器：关键点沙漏网络
        self.predictor = KPHourglass(block_expansion, in_features=image_channel,
                                     max_features=max_features,  reshape_features=reshape_channel, reshape_depth=reshape_depth, num_blocks=num_blocks)

        # 3D 卷积层：将特征图映射为关键点热力图
        # 每个关键点对应一个通道
        # self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=7, padding=3)
        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=3, padding=1)

        # 雅可比矩阵估计层（可选）
        if estimate_jacobian:
            # 每个关键点对应一个 3x3 雅可比矩阵（9个值）
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            # self.jacobian = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=9 * self.num_jacobian_maps, kernel_size=7, padding=3)
            self.jacobian = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=9 * self.num_jacobian_maps, kernel_size=3, padding=1)
            '''
            初始雅可比矩阵为单位矩阵：
            [[1 0 0]
             [0 1 0]
             [0 0 1]]
            '''
            # 将权重初始化为零，偏置初始化为单位矩阵
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        # 如果缩放因子不为 1，使用抗锯齿下采样
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(image_channel, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        从高斯热力图中提取关键点坐标。
        
        原理：计算热力图的加权平均位置作为关键点坐标。
        热力图中每个位置的坐标值乘以其对应的热力图值，然后求和。
        
        参数:
            heatmap (Tensor): 关键点热力图，形状 (bs, num_kp, d, h, w)
        
        返回:
            dict: 包含 'value' 键，值为关键点坐标张量 (bs, num_kp, 3)
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        # 创建坐标网格并扩展维度
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        # 加权求和：热力图值作为权重，坐标作为值
        value = (heatmap * grid).sum(dim=(2, 3, 4))
        kp = {'value': value}

        return kp

    def forward(self, x):
        """
        前向传播：从输入图像检测关键点。
        
        参数:
            x (Tensor): 输入图像，形状 (bs, C, H, W)
        
        返回:
            dict: 包含以下键值对：
                - 'value': 关键点坐标 (bs, num_kp, 3)
                - 'jacobian': 雅可比矩阵 (bs, num_kp, 3, 3)（如果启用）
        """
        # 如果需要下采样，先进行抗锯齿缩放
        if self.scale_factor != 1:
            x = self.down(x)

        # 通过沙漏网络提取特征
        feature_map = self.predictor(x)
        # 预测关键点热力图
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        # 将热力图展平，应用 softmax 归一化（温度参数控制分布的尖锐程度）
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        # 从热力图中提取关键点坐标
        out = self.gaussian2kp(heatmap)

        # 如果启用了雅可比矩阵估计
        if self.jacobian is not None:
            # 从特征图预测雅可比矩阵
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 9, final_shape[2],
                                                final_shape[3], final_shape[4])
            heatmap = heatmap.unsqueeze(2)

            # 使用热力图加权雅可比矩阵（只在关键点附近有意义）
            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 9, -1)
            jacobian = jacobian.sum(dim=-1)
            # 将 9 个值重塑为 3x3 矩阵
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 3, 3)
            out['jacobian'] = jacobian

        return out


class HEEstimator(nn.Module):
    """
    头部姿态和表情估计器（Head and Expression Estimator）。
    
    使用类 ResNet 架构从输入图像中估计：
    - 头部姿态：yaw（偏航）、pitch（俯仰）、roll（翻滚）角度
    - 平移向量 t：头部在 3D 空间中的位置
    - 表情参数 exp：面部表情的偏移量
    
    输出的角度使用分类方式（66 个 bin）表示，
    后续通过 headpose_pred_to_degree 函数转换为角度值。
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, num_bins=66, estimate_jacobian=True):
        """
        参数:
            block_expansion (int): 第一层通道扩展数
            feature_channel (int): 特征通道数
            num_kp (int): 关键点数量（用于计算表情参数维度）
            image_channel (int): 输入图像通道数
            max_features (int): 最大特征通道数
            num_bins (int): 角度分类的 bin 数量，默认 66
            estimate_jacobian (bool): 是否估计雅可比矩阵（此处未使用）
        """
        super(HEEstimator, self).__init__()

        # ====== 类 ResNet 骨干网络 ======
        # 第一层：7x7 卷积 + BN + ReLU + 最大池化
        self.conv1 = nn.Conv2d(in_channels=image_channel, out_channels=block_expansion, kernel_size=7, padding=3, stride=2)
        self.norm1 = BatchNorm2d(block_expansion, affine=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 阶段 1：256 通道
        self.conv2 = nn.Conv2d(in_channels=block_expansion, out_channels=256, kernel_size=1)
        self.norm2 = BatchNorm2d(256, affine=True)
        self.block1 = nn.Sequential()
        for i in range(3):
            self.block1.add_module('b1_'+ str(i), ResBottleneck(in_features=256, stride=1))

        # 阶段 2：512 通道
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.norm3 = BatchNorm2d(512, affine=True)
        self.block2 = ResBottleneck(in_features=512, stride=2)  # 下采样

        self.block3 = nn.Sequential()
        for i in range(3):
            self.block3.add_module('b3_'+ str(i), ResBottleneck(in_features=512, stride=1))

        # 阶段 3：1024 通道
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.norm4 = BatchNorm2d(1024, affine=True)
        self.block4 = ResBottleneck(in_features=1024, stride=2)  # 下采样

        self.block5 = nn.Sequential()
        for i in range(5):
            self.block5.add_module('b5_'+ str(i), ResBottleneck(in_features=1024, stride=1))

        # 阶段 4：2048 通道
        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        self.norm5 = BatchNorm2d(2048, affine=True)
        self.block6 = ResBottleneck(in_features=2048, stride=2)  # 下采样

        self.block7 = nn.Sequential()
        for i in range(2):
            self.block7.add_module('b7_'+ str(i), ResBottleneck(in_features=2048, stride=1))

        # ====== 输出头：分别预测各参数 ======
        self.fc_roll = nn.Linear(2048, num_bins)   # 偏航角分类
        self.fc_pitch = nn.Linear(2048, num_bins)   # 俯仰角分类
        self.fc_yaw = nn.Linear(2048, num_bins)     # 翻滚角分类

        self.fc_t = nn.Linear(2048, 3)              # 平移向量（3D）

        self.fc_exp = nn.Linear(2048, 3*num_kp)     # 表情参数（每个关键点 3 个值）

    def forward(self, x):
        """
        前向传播：从图像估计头部姿态和表情。
        
        参数:
            x (Tensor): 输入图像，形状 (bs, C, H, W)
        
        返回:
            dict: 包含以下键值对：
                - 'yaw': 偏航角预测 (bs, 66)
                - 'pitch': 俯仰角预测 (bs, 66)
                - 'roll': 翻滚角预测 (bs, 66)
                - 't': 平移向量 (bs, 3)
                - 'exp': 表情参数 (bs, 3*num_kp)
        """
        # 前向卷积 + BN + ReLU
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)

        out = self.block1(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.block2(out)

        out = self.block3(out)

        out = self.conv4(out)
        out = self.norm4(out)
        out = F.relu(out)
        out = self.block4(out)

        out = self.block5(out)

        out = self.conv5(out)
        out = self.norm5(out)
        out = F.relu(out)
        out = self.block6(out)

        out = self.block7(out)

        # 全局平均池化，将空间维度压缩为 1x1
        out = F.adaptive_avg_pool2d(out, 1)
        # 展平为一维向量
        out = out.view(out.shape[0], -1)

        # 通过各全连接层分别预测参数
        yaw = self.fc_roll(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_yaw(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
