"""
dense_motion.py - 稠密运动估计网络模块
该模块负责从稀疏关键点运动（由关键点检测器检测到的源和驱动面部关键点）推断出稠密的光流场。
核心思想：利用稀疏的关键点位移，通过神经网络预测出每个像素的运动向量，
从而实现面部区域的精确变形（warping）。
"""
from torch import nn
import torch.nn.functional as F
import torch
# 导入工具函数：Hourglass（沙漏网络）、坐标网格生成、关键点转高斯热图
from src.facerender.modules.util import Hourglass, make_coordinate_grid, kp2gaussian

# 导入同步批归一化层（支持多GPU训练）
from src.facerender.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class DenseMotionNetwork(nn.Module):
    """
    稠密运动估计网络。
    
    该模块根据源面部（kp_source）和驱动面部（kp_driving）的稀疏关键点，
    预测稠密的运动场（deformation field）和遮挡图（occlusion map）。
    
    主要步骤：
    1. 创建稀疏运动场：基于关键点位移和雅可比矩阵计算每个关键点对应的运动
    2. 创建变形特征：利用稀疏运动对特征进行变形
    3. 生成热力图表示：将关键点转为高斯热力图差值
    4. 通过沙漏网络预测遮挡掩码和运动权重
    5. 加权融合各关键点的稀疏运动，得到最终稠密运动场
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=False):
        """
        初始化稠密运动估计网络。
        
        参数:
            block_expansion (int): 沙漏网络中每层通道数的扩展因子
            num_blocks (int): 沙漏网络的编码/解码层数
            max_features (int): 沙漏网络中最大特征通道数
            num_kp (int): 关键点数量
            feature_channel (int): 输入特征通道数
            reshape_depth (int): 3D 特征的深度维度
            compress (int): 特征压缩后的通道数
            estimate_occlusion_map (bool): 是否估计遮挡图
        """
        super(DenseMotionNetwork, self).__init__()
        # 初始化沙漏网络（编码器-解码器结构），用于从变形特征预测运动权重
        # 输入维度为 (num_kp+1) * (compress+1)，+1 是因为包含背景层
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)

        # 3D 卷积层，输出每个关键点（加背景）对应的运动掩码
        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)

        # 1x1 卷积用于压缩特征通道数，减少计算量
        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        # 3D 批归一化层
        self.norm = BatchNorm3d(compress, affine=True)

        # 如果需要估计遮挡图，添加一个 2D 卷积层输出单通道遮挡掩码
        if estimate_occlusion_map:
            # self.occlusion = nn.Conv2d(reshape_channel*reshape_depth, 1, kernel_size=7, padding=3)
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

        self.num_kp = num_kp


    def create_sparse_motions(self, feature, kp_driving, kp_source):
        """
        创建稀疏运动场。
        
        基于源关键点和驱动关键点的位移，为每个关键点生成一个运动场。
        运动场表示每个空间位置到目标位置的映射。
        如果存在雅可比矩阵，还会考虑局部形变（旋转、缩放等仿射变换）。
        
        参数:
            feature (Tensor): 3D 特征图，形状 (bs, c, d, h, w)
            kp_driving (dict): 驱动面部关键点，包含 'value' 和可选的 'jacobian'
            kp_source (dict): 源面部关键点，包含 'value' 和可选的 'jacobian'
        
        返回:
            Tensor: 稀疏运动场，形状 (bs, num_kp+1, d, h, w, 3)
        """
        bs, _, d, h, w = feature.shape
        # 创建标准坐标网格，范围 [-1, 1]，表示无运动的参考坐标
        identity_grid = make_coordinate_grid((d, h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        # 计算驱动关键点相对于标准位置的偏移
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 1, 3)
        
        # 如果存在雅可比矩阵，应用局部仿射变换（旋转、缩放等）
        # 雅可比矩阵描述了关键点附近的局部形变
        if 'jacobian' in kp_driving and kp_driving['jacobian'] is not None:
            # 计算源到驱动的雅可比变换：源雅可比 * 驱动雅可比的逆
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            # 扩展维度以匹配空间坐标网格
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
            # 应用雅可比变换到坐标网格
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)                  

        # 将偏移转换为源关键点坐标系：驱动到源的运动映射
        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        # 添加背景层（无运动的恒等变换）
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        # 拼接背景层和各关键点的运动场
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)                #bs num_kp+1 d h w 3
        
        # sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        """
        创建变形特征图。
        
        利用稀疏运动场对输入特征进行空间变形（warping），
        每个关键点产生一个变形后的特征副本。
        
        参数:
            feature (Tensor): 压缩后的 3D 特征，形状 (bs, c, d, h, w)
            sparse_motions (Tensor): 稀疏运动场，形状 (bs, num_kp+1, d, h, w, 3)
        
        返回:
            Tensor: 变形特征，形状 (bs, num_kp+1, c, d, h, w)
        """
        bs, _, d, h, w = feature.shape
        # 为每个关键点复制一份特征图
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3) !!!!
        # 使用双线性插值进行空间变形（grid_sample 实现光流 warp）
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)
        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        """
        创建热力图表示。
        
        将关键点转换为高斯热力图，并计算驱动与源的差值热力图，
        用于指示哪些区域发生了运动。
        
        参数:
            feature (Tensor): 变形后的特征图
            kp_driving (dict): 驱动面部关键点
            kp_source (dict): 源面部关键点
        
        返回:
            Tensor: 热力图，形状 (bs, num_kp+1, 1, d, h, w)
        """
        spatial_size = feature.shape[3:]
        # 将驱动关键点和源关键点分别转为高斯热力图
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)
        # 计算差值热力图：驱动减源，正值表示运动区域
        heatmap = gaussian_driving - gaussian_source

        # 添加背景通道（全零，表示背景无运动）
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        """
        前向传播：预测稠密运动场。
        
        参数:
            feature (Tensor): 输入 3D 特征图，形状 (bs, c, d, h, w)
            kp_driving (dict): 驱动面部关键点
            kp_source (dict): 源面部关键点
        
        返回:
            dict: 包含以下键值对：
                - 'mask': 运动权重掩码，形状 (bs, num_kp+1, d, h, w)
                - 'deformation': 稠密变形场，形状 (bs, d, h, w, 3)
                - 'occlusion_map': 遮挡图（如果启用），形状 (bs, 1, h, w)
        """
        bs, _, d, h, w = feature.shape

        # 特征压缩：使用 1x1 卷积减少通道数
        feature = self.compress(feature)
        # 批归一化 + ReLU 激活
        feature = self.norm(feature)
        feature = F.relu(feature)

        out_dict = dict()
        # 步骤1：创建稀疏运动场
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        # 步骤2：利用稀疏运动对特征进行变形
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)

        # 步骤3：生成热力图表示
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)

        # 将热力图和变形特征拼接作为沙漏网络的输入
        input_ = torch.cat([heatmap, deformed_feature], dim=2)
        input_ = input_.view(bs, -1, d, h, w)

        # input = deformed_feature.view(bs, -1, d, h, w)      # (bs, num_kp+1 * c, d, h, w)

        # 通过沙漏网络预测运动权重
        prediction = self.hourglass(input_)

        # 生成运动掩码并使用 softmax 归一化，使权重和为 1
        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs, num_kp+1, 1, d, h, w)
        
        # 将极小的权重置零，避免数值问题
        zeros_mask = torch.zeros_like(mask)   
        mask = torch.where(mask < 1e-3, zeros_mask, mask) 

        # 步骤4：加权融合各关键点的稀疏运动，得到最终稠密运动场
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        # 如果启用了遮挡估计，生成遮挡图
        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w)
            # 使用 sigmoid 将输出映射到 [0, 1]，表示遮挡程度
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
