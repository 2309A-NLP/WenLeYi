# -*- coding: utf-8 -*-
# face_model.py - 人脸生成模型
# 本文件定义了PIRender的人脸生成网络（FaceGenerator），
# 包含映射网络（MappingNet）、变形网络（WarpingNet）和编辑网络（EditingNet），
# 实现基于姿态条件的人脸图像驱动和生成。

import functools
import torch
import torch.nn as nn
from .base_function import LayerNorm2d, ADAINHourglass, FineEncoder, FineDecoder

def convert_flow_to_deformation(flow):
    """将光流场转换为变形场
    光流场表示像素的位移向量，将其转换为网格采样所需的变形坐标。
    变形场的值范围为(-1, 1)，可直接用于grid_sample函数。
    
    参数:
        flow (tensor): 模型输出的光流场，形状为 (B, 2, H, W)
    返回:
        deformation (tensor): 用于图像变形的网格坐标，形状为 (B, H, W, 2)
    """
    b,c,h,w = flow.shape
    # 将光流归一化到(-1, 1)范围
    flow_norm = 2 * torch.cat([flow[:,:1,...]/(w-1),flow[:,1:,...]/(h-1)], 1)
    # 生成基础坐标网格
    grid = make_coordinate_grid(flow)
    # 基础网格加上光流偏移得到变形场
    deformation = grid + flow_norm.permute(0,2,3,1)
    return deformation

def make_coordinate_grid(flow):
    """生成与光流场相同尺寸的坐标网格
    创建一个归一化到(-1, 1)范围的2D坐标网格，
    用于后续的图像变形操作。
    
    参数:
        flow (tensor): 光流场，用于确定网格的尺寸
    返回:
        grid (tensor): 归一化的2D坐标网格，形状为 (B, H, W, 2)
    """    
    b,c,h,w = flow.shape

    # 生成x方向的坐标，归一化到(-1, 1)
    x = torch.arange(w).to(flow)
    y = torch.arange(h).to(flow)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    # 通过广播生成2D网格
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    # 拼接为(B, H, W, 2)格式的坐标网格
    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    meshed = meshed.expand(b, -1, -1, -1)
    return meshed    

    
def warp_image(source_image, deformation):
    """根据变形场对输入图像进行变形（warping）
    使用双线性插值的grid_sample函数，根据变形场的坐标对源图像进行采样，
    实现图像的空间变换。
    
    参数:
        source_image (tensor): 待变形的源图像，形状为 (B, C, H, W)
        deformation (tensor): 变形场坐标，形状为 (B, H, W, 2)，值范围(-1, 1)
    返回:
        output (tensor): 变形后的图像，形状与source_image相同
    """ 
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = source_image.shape
    # 如果变形场和源图像尺寸不匹配，对变形场进行双线性插值缩放
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear')
        deformation = deformation.permute(0, 2, 3, 1)
    # 使用grid_sample进行双线性插值采样
    return torch.nn.functional.grid_sample(source_image, deformation) 


class FaceGenerator(nn.Module):
    """人脸生成器主网络
    由三个子网络组成：
    1. MappingNet: 将3DMM姿态参数映射为姿态描述符
    2. WarpingNet: 根据姿态描述符对源图像进行变形
    3. EditingNet: 对变形后的图像进行精细编辑，生成最终输出
    
    参数:
        mapping_net: 映射网络的配置字典
        warpping_net: 变形网络的配置字典
        editing_net: 编辑网络的配置字典
        common: 共享配置（如归一化层、激活函数等）
    """
    def __init__(
        self, 
        mapping_net, 
        warpping_net, 
        editing_net, 
        common
        ):  
        super(FaceGenerator, self).__init__()
        # 初始化三个子网络
        self.mapping_net = MappingNet(**mapping_net)
        self.warpping_net = WarpingNet(**warpping_net, **common)
        self.editing_net = EditingNet(**editing_net, **common)
 
    def forward(
        self, 
        input_image, 
        driving_source, 
        stage=None
        ):
        """
        参数:
            input_image: 源人脸图像
            driving_source: 驱动姿态的3DMM参数
            stage: 控制运行阶段
                - 'warp': 仅执行映射和变形（不执行编辑）
                - None/其他: 执行完整的映射、变形和编辑流程
        返回:
            output: 包含变形结果和生成图像的字典
        """
        if stage == 'warp':
            # 仅执行映射和变形阶段
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
        else:
            # 完整流程：映射 -> 变形 -> 编辑
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
            # 编辑网络接收源图像、变形图像和姿态描述符，生成最终图像
            output['fake_image'] = self.editing_net(input_image, output['warp_image'], descriptor)
        return output

class MappingNet(nn.Module):
    """映射网络
    将3DMM姿态系数（如exp、pose等）映射为紧凑的姿态描述符。
    通过1D卷积和全局平均池化，将变长的3DMM参数压缩为固定长度的描述符向量。
    
    参数:
        coeff_nc: 3DMM系数的通道数（输入维度）
        descriptor_nc: 描述符的通道数（输出维度）
        layer: 1D卷积层的数量
    """
    def __init__(self, coeff_nc, descriptor_nc, layer):
        super( MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        # 第一层1D卷积，将输入系数映射到描述符空间
        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        # 多层1D卷积块，使用膨胀卷积扩大感受野
        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)   

        # 全局平均池化，将序列压缩为单一向量
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        """
        参数:
            input_3dmm: 3DMM姿态系数，形状为 (B, coeff_nc, T)
        返回:
            out: 姿态描述符，形状为 (B, descriptor_nc, 1)
        """
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            # 残差连接：卷积输出 + 裁剪后的输入
            out = model(out) + out[:,:,3:-3]
        # 全局平均池化，得到固定长度的描述符
        out = self.pooling(out)
        return out   

class WarpingNet(nn.Module):
    """变形网络
    使用ADAIN沙漏网络将姿态描述符注入到图像特征中，
    预测光流场，然后根据光流场对源图像进行变形。
    
    参数:
        image_nc: 输入图像通道数
        descriptor_nc: 姿态描述符通道数
        base_nc: 基础特征图数量
        max_nc: 最大特征图数量
        encoder_layer: 编码器层数
        decoder_layer: 解码器层数
        use_spect: 是否使用谱归一化
    """
    def __init__(
        self, 
        image_nc, 
        descriptor_nc, 
        base_nc, 
        max_nc, 
        encoder_layer, 
        decoder_layer, 
        use_spect
        ):
        super( WarpingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        # 使用2D层归一化
        norm_layer = functools.partial(LayerNorm2d, affine=True) 
        kwargs = {'nonlinearity':nonlinearity, 'use_spect':use_spect}

        self.descriptor_nc = descriptor_nc 
        # ADAIN沙漏网络：编码-解码结构，注入姿态条件
        self.hourglass = ADAINHourglass(image_nc, self.descriptor_nc, base_nc,
                                       max_nc, encoder_layer, decoder_layer, **kwargs)

        # 光流预测头：归一化 -> 激活 -> 卷积输出2通道光流
        self.flow_out = nn.Sequential(norm_layer(self.hourglass.output_nc), 
                                      nonlinearity,
                                      nn.Conv2d(self.hourglass.output_nc, 2, kernel_size=7, stride=1, padding=3))

        # 全局平均池化（备用）
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image, descriptor):
        """
        参数:
            input_image: 源人脸图像，形状为 (B, C, H, W)
            descriptor: 姿态描述符，形状为 (B, descriptor_nc, 1)
        返回:
            final_output: 包含光流场和变形图像的字典
        """
        final_output={}
        # 通过沙漏网络提取特征
        output = self.hourglass(input_image, descriptor)
        # 预测光流场
        final_output['flow_field'] = self.flow_out(output)

        # 将光流场转换为变形场
        deformation = convert_flow_to_deformation(final_output['flow_field'])
        # 根据变形场对源图像进行变形
        final_output['warp_image'] = warp_image(input_image, deformation)
        return final_output


class EditingNet(nn.Module):
    """编辑网络
    对变形后的图像进行精细编辑，融合源图像和变形图像的信息，
    生成高质量的人脸图像。使用编码器-解码器结构，
    通过ADAIN残差块实现条件特征精炼。
    
    参数:
        image_nc: 输入图像通道数
        descriptor_nc: 姿态描述符通道数
        layer: 编码器/解码器层数
        base_nc: 基础特征图数量
        max_nc: 最大特征图数量
        num_res_blocks: 每层的残差块数量
        use_spect: 是否使用谱归一化
    """  
    def __init__(
        self, 
        image_nc, 
        descriptor_nc, 
        layer, 
        base_nc, 
        max_nc, 
        num_res_blocks, 
        use_spect):  
        super(EditingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True) 
        kwargs = {'norm_layer':norm_layer, 'nonlinearity':nonlinearity, 'use_spect':use_spect}
        self.descriptor_nc = descriptor_nc

        # 编码器：输入为源图像和变形图像的拼接（通道数翻倍）
        self.encoder = FineEncoder(image_nc*2, base_nc, max_nc, layer, **kwargs)
        # 解码器：使用ADAIN残差块和跳跃连接生成输出图像
        self.decoder = FineDecoder(image_nc, self.descriptor_nc, base_nc, max_nc, layer, num_res_blocks, **kwargs)

    def forward(self, input_image, warp_image, descriptor):
        """
        参数:
            input_image: 源图像，形状为 (B, C, H, W)
            warp_image: 变形后的图像，形状为 (B, C, H, W)
            descriptor: 姿态描述符，形状为 (B, descriptor_nc, 1)
        返回:
            gen_image: 生成的编辑后图像
        """
        # 将源图像和变形图像在通道维度拼接
        x = torch.cat([input_image, warp_image], 1)
        # 编码
        x = self.encoder(x)
        # 解码生成最终图像
        gen_image = self.decoder(x, descriptor)
        return gen_image
