"""
generator.py - 图像生成器网络模块
该模块实现了两种图像生成器架构：
1. OcclusionAwareGenerator：基于遮挡感知的生成器，使用标准编解码器结构
2. OcclusionAwareSPADEGenerator：结合遮挡感知和 SPADE（空间自适应归一化）的生成器
3. SPADEDecoder：基于 SPADE 归一化的解码器，能更好地保留语义信息

这两种生成器都能根据源面部图像和稠密运动场生成新的面部图像，
其中遮挡区域通过遮挡图（occlusion map）进行处理，
避免在被遮挡区域产生伪影。
"""
import torch
from torch import nn
import torch.nn.functional as F
# 导入各种构建块：残差块、相同维度块、上/下采样块、SPADE 残差块
from src.facerender.modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, ResBlock3d, SPADEResnetBlock
# 导入稠密运动估计网络
from src.facerender.modules.dense_motion import DenseMotionNetwork


class OcclusionAwareGenerator(nn.Module):
    """
    基于遮挡感知的图像生成器（NVIDIA 架构变体）。
    
    架构流程：
    1. 编码器：将源图像下采样为多尺度特征
    2. 3D 残差块：在 3D 空间中处理特征
    3. 稠密运动网络：预测运动场和遮挡图
    4. 特征变形：利用运动场对特征进行空间变形
    5. 解码器：将变形后的特征上采样回原始分辨率
    
    遮挡处理：遮挡区域的特征被遮挡图掩码置零，
    使生成器只从可见区域合成新图像。
    """

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        """
        参数:
            image_channel (int): 输入/输出图像通道数（如 3 表示 RGB）
            feature_channel (int): 输入特征通道数
            num_kp (int): 关键点数量
            block_expansion (int): 第一层通道扩展数
            max_features (int): 最大特征通道数
            num_down_blocks (int): 下采样块数量
            reshape_channel (int): 3D 特征的通道数
            reshape_depth (int): 3D 特征的深度维度
            num_resblocks (int): 残差块数量
            estimate_occlusion_map (bool): 是否估计遮挡图
            dense_motion_params (dict): 稠密运动网络的参数
            estimate_jacobian (bool): 是否估计关键点雅可比矩阵
        """
        super(OcclusionAwareGenerator, self).__init__()

        # 初始化稠密运动估计网络（如果提供了参数）
        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        # 编码器部分：将输入图像下采样为特征图
        # 第一层：7x7 卷积，提取初始特征
        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        # 多层下采样块：逐步降低空间分辨率，增加通道数
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        # 1x1 卷积调整通道数到最大特征数
        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        # 3D 残差块：在 3D 空间中增强特征表达
        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        # 运动变形后的处理层
        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        # 2D 残差块：在 2D 空间中进一步处理特征
        self.resblocks_2d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_2d.add_module('2dr' + str(i), ResBlock2d(out_features, kernel_size=3, padding=1))

        # 解码器部分：多层上采样块，逐步恢复空间分辨率
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        # 最终 7x7 卷积输出图像
        self.final = nn.Conv2d(block_expansion, image_channel, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

    def deform_input(self, inp, deformation):
        """
        对输入特征图应用变形场（warping）。
        
        使用三线性插值（trilinear interpolation）根据变形场对 3D 特征进行空间变形。
        如果变形场的空间尺寸与输入不匹配，会先进行插值调整。
        
        参数:
            inp (Tensor): 输入 3D 特征，形状 (bs, c, d, h, w)
            deformation (Tensor): 变形场，形状 (bs, d, h, w, 3)
        
        返回:
            Tensor: 变形后的特征
        """
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            # 调整变形场的空间尺寸以匹配输入
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        # 使用 grid_sample 进行空间变形
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        """
        前向传播：根据源图像和关键点生成新图像。
        
        参数:
            source_image (Tensor): 源面部图像，形状 (bs, C, H, W)
            kp_driving (dict): 驱动面部关键点
            kp_source (dict): 源面部关键点
        
        返回:
            dict: 包含以下键值对：
                - 'prediction': 生成的面部图像
                - 'mask': 运动权重掩码
                - 'occlusion_map': 遮挡图（如果启用）
        """
        # ====== 编码阶段（下采样） ======
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        # 将 2D 特征重塑为 3D 特征，引入深度维度
        # print(out.shape)
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        # 通过 3D 残差块增强特征
        feature_3d = self.resblocks_3d(feature_3d)

        # ====== 根据变形和遮挡变换特征表示 ======
        output_dict = {}
        if self.dense_motion_network is not None:
            # 使用稠密运动网络预测运动场和遮挡图
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            # 利用运动场对 3D 特征进行变形
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            # 将 3D 特征展平为 2D 特征（深度维度与通道维度合并）
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            # 应用遮挡掩码：被遮挡区域的特征置零
            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    # 如果尺寸不匹配，双线性插值调整遮挡图大小
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            # output_dict["deformed"] = self.deform_input(source_image, deformation)  # 3d deformation cannot deform 2d image

        # ====== 解码阶段 ======
        # 通过 2D 残差块处理
        out = self.resblocks_2d(out)
        # 上采样恢复原始分辨率
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        # 最终卷积输出图像，sigmoid 将像素值限制在 [0, 1]
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict


class SPADEDecoder(nn.Module):
    """
    基于 SPADE（Spatially-Adaptive Normalization）的解码器。
    
    SPADE 是一种空间自适应的归一化方法，能根据语义分割图
    动态调整归一化参数（gamma 和 beta），
    从而在生成过程中更好地保留空间结构和语义信息。
    
    该解码器由多层 SPADE 残差块组成，逐步上采样特征并生成图像。
    """

    def __init__(self):
        """初始化 SPADE 解码器。"""
        super().__init__()
        ic = 256  # 输入通道数
        oc = 64   # 输出通道数（最终图像前的通道数）
        norm_G = 'spadespectralinstance'  # 使用 SPADE + 谱归一化 + 实例归一化
        label_nc = 256  # 标签/语义图通道数
        
        # 1x1 卷积将输入通道数翻倍
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        # 中间层：6 个 SPADE 残差块，保持 2*ic 通道数
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        # 上采样层：逐步减少通道数
        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)    # 256 -> 128 通道
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)        # 128 -> 64 通道
        # 最终 3x3 卷积输出 RGB 图像
        self.conv_img = nn.Conv2d(oc, 3, 3, padding=1)
        # 2 倍双线性上采样
        self.up = nn.Upsample(scale_factor=2)
        
    def forward(self, feature):
        """
        前向传播：将特征解码为图像。
        
        参数:
            feature (Tensor): 编码器输出的特征图，形状 (bs, 256, H, W)
        
        返回:
            Tensor: 生成的 RGB 图像，像素值范围 [0, 1]
        """
        # 语义图（使用特征本身作为 SPADE 的条件输入）
        seg = feature
        x = self.fc(feature)
        # 通过 6 层 SPADE 残差块处理
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)
        # 上采样 2 倍 + SPADE 残差块
        x = self.up(x)               
        x = self.up_0(x, seg)         # 256, 128, 128
        x = self.up(x)               
        x = self.up_1(x, seg)         # 64, 256, 256

        # LeakyReLU 激活 + 最终卷积输出 3 通道图像
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = torch.tanh(x)
        # sigmoid 将输出限制在 [0, 1] 范围
        x = F.sigmoid(x)
        
        return x


class OcclusionAwareSPADEGenerator(nn.Module):
    """
    结合遮挡感知和 SPADE 的图像生成器。
    
    与 OcclusionAwareGenerator 类似，但解码器部分使用 SPADEDecoder，
    能够更好地保持生成图像的空间结构和语义一致性。
    
    这是 SadTalker 默认使用的生成器架构，效果最好。
    """

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        """
        初始化函数，参数含义同 OcclusionAwareGenerator。
        """
        super(OcclusionAwareSPADEGenerator, self).__init__()

        # 初始化稠密运动估计网络
        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        # 编码器：3x3 卷积（注意这里用 3x3 而不是 7x7）
        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        # 多层下采样块
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        # 1x1 卷积调整通道数
        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        # 3D 残差块
        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        # 运动变形后的处理层
        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

        # 使用 SPADE 解码器（替代原始的上采样块 + 残差块）
        self.decoder = SPADEDecoder()

    def deform_input(self, inp, deformation):
        """
        对输入特征图应用变形场（warping），与 OcclusionAwareGenerator 中相同。
        """
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        """
        前向传播：根据源图像和关键点生成新图像。
        
        与 OcclusionAwareGenerator 的区别在于解码器部分使用 SPADEDecoder，
        能更好地保留空间结构和语义信息。
        
        参数:
            source_image (Tensor): 源面部图像，形状 (bs, C, H, W)
            kp_driving (dict): 驱动面部关键点
            kp_source (dict): 源面部关键点
        
        返回:
            dict: 包含 'prediction'（生成图像）、'mask'（运动掩码）和可选的 'occlusion_map'
        """
        # ====== 编码阶段（下采样） ======
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        # 将 2D 特征重塑为 3D 特征
        # print(out.shape)
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)

        # ====== 根据变形和遮挡变换特征表示 ======
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            # import pdb; pdb.set_trace()

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            # occlusion_map = torch.where(occlusion_map < 0.95, 0, occlusion_map)
            
            # 应用遮挡掩码
            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

        # ====== 解码阶段：使用 SPADE 解码器 ======
        out = self.decoder(out)

        output_dict["prediction"] = out
        
        return output_dict