# -*- coding: utf-8 -*-
# base_function.py - 基础网络构建函数模块
# 本文件定义了用于PIRender人脸渲染的神经网络基础组件，
# 包括归一化层、编码器/解码器、残差块等模块。

import sys
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm


class LayerNorm2d(nn.Module):
    """2D层归一化模块
    对2D特征图在通道维度上进行层归一化（Layer Normalization），
    可选是否使用可学习的缩放（weight）和偏移（bias）参数。
    输入形状为 (N, C, H, W)，沿 (C, H, W) 维度计算归一化。
    """
    def __init__(self, n_out, affine=True):
        super(LayerNorm2d, self).__init__()
        self.n_out = n_out
        self.affine = affine

        # 如果使用仿射变换，创建可学习的缩放参数weight和偏移参数bias
        if self.affine:
          self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
          self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        # 获取除batch维度外的所有维度作为归一化维度
        normalized_shape = x.size()[1:]
        if self.affine:
          # 使用可学习参数进行仿射变换归一化
          return F.layer_norm(x, normalized_shape, \
              self.weight.expand(normalized_shape), 
              self.bias.expand(normalized_shape))
              
        else:
          # 不使用可学习参数，仅做归一化
          return F.layer_norm(x, normalized_shape)  

class ADAINHourglass(nn.Module):
    """自适应实例归一化（ADAIN）沙漏网络
    沙漏（Hourglass）结构是一种编码器-解码器架构，
    通过ADAIN机制将姿态条件信息注入到图像生成过程中，
    实现基于姿态的图像变形。
    
    参数:
        image_nc: 输入图像的通道数
        pose_nc: 姿态描述符的通道数
        ngf: 生成器基础特征图数量
        img_f: 特征图数量的上限
        encoder_layers: 编码器层数
        decoder_layers: 解码器层数
        nonlinearity: 激活函数
        use_spect: 是否使用谱归一化
    """
    def __init__(self, image_nc, pose_nc, ngf, img_f, encoder_layers, decoder_layers, nonlinearity, use_spect):
        super(ADAINHourglass, self).__init__()
        # 创建ADAIN编码器，将图像和姿态编码为多尺度特征
        self.encoder = ADAINEncoder(image_nc, pose_nc, ngf, img_f, encoder_layers, nonlinearity, use_spect)
        # 创建ADAIN解码器，将编码特征解码回图像空间
        self.decoder = ADAINDecoder(pose_nc, ngf, img_f, encoder_layers, decoder_layers, True, nonlinearity, use_spect)
        self.output_nc = self.decoder.output_nc

    def forward(self, x, z):
        # 编码输入图像和姿态，然后解码得到输出
        return self.decoder(self.encoder(x, z), z)                



class ADAINEncoder(nn.Module):
    """ADAIN编码器
    将输入图像通过一系列卷积块进行下采样编码，
    每个编码块都使用ADAIN将姿态条件信息注入。
    逐层提取图像的多尺度特征。
    
    参数:
        image_nc: 输入图像通道数
        pose_nc: 姿态描述符通道数
        ngf: 基础特征图数量
        img_f: 特征图数量上限
        layers: 编码器层数
        nonlinearity: 激活函数
        use_spect: 是否使用谱归一化
    """
    def __init__(self, image_nc, pose_nc, ngf, img_f, layers, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINEncoder, self).__init__()
        self.layers = layers
        # 第一层使用7x7卷积进行初始特征提取
        self.input_layer = nn.Conv2d(image_nc, ngf, kernel_size=7, stride=1, padding=3)
        # 逐层构建编码块，通道数逐层加倍（上限为img_f）
        for i in range(layers):
            in_channels = min(ngf * (2**i), img_f)
            out_channels = min(ngf *(2**(i+1)), img_f)
            model = ADAINEncoderBlock(in_channels, out_channels, pose_nc, nonlinearity, use_spect)
            setattr(self, 'encoder' + str(i), model)
        self.output_nc = out_channels
        
    def forward(self, x, z):
        # x: 输入图像特征, z: 姿态描述符
        out = self.input_layer(x)
        # 保存每层的输出，用于解码器的跳跃连接
        out_list = [out]
        for i in range(self.layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out, z)
            out_list.append(out)
        return out_list
        
class ADAINDecoder(nn.Module):
    """ADAIN解码器
    通过转置卷积逐步上采样，将编码器的多尺度特征解码回图像空间。
    支持跳跃连接（skip connection），将编码器对应层的特征与解码器特征拼接。
    
    参数:
        pose_nc: 姿态描述符通道数
        ngf: 基础特征图数量
        img_f: 特征图数量上限
        encoder_layers: 编码器层数（用于确定解码器的起始层）
        decoder_layers: 解码器层数
        skip_connect: 是否使用跳跃连接
        nonlinearity: 激活函数
        use_spect: 是否使用谱归一化
    """
    def __init__(self, pose_nc, ngf, img_f, encoder_layers, decoder_layers, skip_connect=True, 
                 nonlinearity=nn.LeakyReLU(), use_spect=False):

        super(ADAINDecoder, self).__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.skip_connect = skip_connect
        use_transpose = True

        # 从最深层开始逐层构建解码块
        for i in range(encoder_layers-decoder_layers, encoder_layers)[::-1]:
            in_channels = min(ngf * (2**(i+1)), img_f)
            # 如果使用跳跃连接且不是最深层，输入通道数翻倍（与编码器特征拼接）
            in_channels = in_channels*2 if i != (encoder_layers-1) and self.skip_connect else in_channels
            out_channels = min(ngf * (2**i), img_f)
            model = ADAINDecoderBlock(in_channels, out_channels, out_channels, pose_nc, use_transpose, nonlinearity, use_spect)
            setattr(self, 'decoder' + str(i), model)

        self.output_nc = out_channels*2 if self.skip_connect else out_channels

    def forward(self, x, z):
        # x: 编码器输出的多尺度特征列表, z: 姿态描述符
        # 取出最深层的特征作为解码器的输入
        out = x.pop() if self.skip_connect else x
        for i in range(self.encoder_layers-self.decoder_layers, self.encoder_layers)[::-1]:
            model = getattr(self, 'decoder' + str(i))
            out = model(out, z)
            # 如果使用跳跃连接，将当前层特征与编码器对应层特征拼接
            out = torch.cat([out, x.pop()], 1) if self.skip_connect else out
        return out

class ADAINEncoderBlock(nn.Module):
    """ADAIN编码块
    由两个卷积层和对应的ADAIN归一化层组成。
    第一个卷积层进行2倍下采样，第二个保持空间尺寸。
    每个卷积前使用ADAIN根据姿态条件调整特征统计量。
    """       
    def __init__(self, input_nc, output_nc, feature_nc, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINEncoderBlock, self).__init__()
        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        # 下采样卷积（4x4，步长2）和精细卷积（3x3，步长1）
        self.conv_0 = spectral_norm(nn.Conv2d(input_nc,  output_nc, **kwargs_down), use_spect)
        self.conv_1 = spectral_norm(nn.Conv2d(output_nc, output_nc, **kwargs_fine), use_spect)

        # ADAIN归一化层，根据姿态特征调整输入特征
        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(output_nc, feature_nc)
        self.actvn = nonlinearity

    def forward(self, x, z):
        # x: 输入特征图, z: 姿态描述符
        # ADAIN归一化 -> 激活 -> 卷积
        x = self.conv_0(self.actvn(self.norm_0(x, z)))
        x = self.conv_1(self.actvn(self.norm_1(x, z)))
        return x

class ADAINDecoderBlock(nn.Module):
    """ADAIN解码块
    由卷积层、上采样层和跳跃连接组成。
    使用ADAIN归一化层将姿态条件信息注入到特征中。
    支持转置卷积或上采样+卷积两种上采样方式。
    
    参数:
        input_nc: 输入通道数
        output_nc: 输出通道数
        hidden_nc: 隐藏层通道数
        feature_nc: 姿态特征通道数
        use_transpose: 是否使用转置卷积上采样
        nonlinearity: 激活函数
        use_spect: 是否使用谱归一化
    """
    def __init__(self, input_nc, output_nc, hidden_nc, feature_nc, use_transpose=True, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINDecoderBlock, self).__init__()        
        # 设置激活函数
        self.actvn = nonlinearity
        hidden_nc = min(input_nc, output_nc) if hidden_nc is None else hidden_nc

        kwargs_fine = {'kernel_size':3, 'stride':1, 'padding':1}
        if use_transpose:
            # 使用转置卷积进行上采样
            kwargs_up = {'kernel_size':3, 'stride':2, 'padding':1, 'output_padding':1}
        else:
            # 使用普通卷积
            kwargs_up = {'kernel_size':3, 'stride':1, 'padding':1}

        # 创建卷积层
        self.conv_0 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, **kwargs_fine), use_spect)
        if use_transpose:
            # 主路径的转置卷积上采样
            self.conv_1 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, **kwargs_up), use_spect)
            # 跳跃连接路径的转置卷积上采样
            self.conv_s = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, **kwargs_up), use_spect)
        else:
            # 使用普通卷积 + 最近邻上采样
            self.conv_1 = nn.Sequential(spectral_norm(nn.Conv2d(hidden_nc, output_nc, **kwargs_up), use_spect),
                                        nn.Upsample(scale_factor=2))
            self.conv_s = nn.Sequential(spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs_up), use_spect),
                                        nn.Upsample(scale_factor=2))
        # 定义ADAIN归一化层
        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(hidden_nc, feature_nc)
        self.norm_s = ADAIN(input_nc, feature_nc)
        
    def forward(self, x, z):
        # 计算跳跃连接路径
        x_s = self.shortcut(x, z)
        # 主路径: ADAIN归一化 -> 激活 -> 卷积 -> ADAIN归一化 -> 激活 -> 上采样卷积
        dx = self.conv_0(self.actvn(self.norm_0(x, z)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, z)))
        # 残差连接：跳跃路径 + 主路径
        out = x_s + dx
        return out

    def shortcut(self, x, z):
        # 跳跃连接分支：ADAIN归一化 -> 激活 -> 上采样卷积
        x_s = self.conv_s(self.actvn(self.norm_s(x, z)))
        return x_s              


def spectral_norm(module, use_spect=True):
    """谱归一化包装函数
    使用谱归一化来稳定训练过程，防止判别器/生成器的梯度爆炸。
    谱归一化通过限制权重矩阵的谱范数来约束网络的Lipschitz常数。
    
    参数:
        module: 需要应用谱归一化的模块
        use_spect: 是否启用谱归一化
    返回:
        应用了谱归一化的模块，或原始模块（如果未启用）
    """
    if use_spect:
        return SpectralNorm(module)
    else:
        return module


class ADAIN(nn.Module):
    """自适应实例归一化（Adaptive Instance Normalization）层
    ADAIN的核心思想是：先对输入特征做实例归一化（去除风格信息），
    然后根据条件特征（如姿态描述符）重新生成缩放（gamma）和偏移（beta）参数，
    实现风格/姿态的自适应转换。
    
    数学公式: out = (x - mean(x)) / std(x) * (1 + gamma) + beta
    其中 gamma 和 beta 由条件特征通过MLP生成。
    
    参数:
        norm_nc: 需要归一化的特征通道数
        feature_nc: 条件特征（姿态描述符）的通道数
    """
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        # 无参数的实例归一化层，仅做归一化不学习参数
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        use_bias=True

        # MLP网络：将条件特征映射为gamma和beta参数
        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        # 输出gamma（缩放参数）和beta（偏移参数）
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):

        # 第一部分：生成无参数的归一化激活值
        normalized = self.param_free_norm(x)

        # 第二部分：根据条件特征生成缩放和偏移参数
        # 将特征从(B, C, H, W)展平为(B, C*H*W)
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # 应用缩放和偏移
        # 将gamma和beta从(B, C) reshape为(B, C, 1, 1)以便广播
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta
        return out


class FineEncoder(nn.Module):
    """精细编码器
    用于编辑网络（EditingNet）中的图像编码。
    通过逐层下采样提取图像的多尺度特征，
    每层使用BatchNorm进行归一化。
    
    参数:
        image_nc: 输入图像通道数
        ngf: 基础特征图数量
        img_f: 特征图数量上限
        layers: 编码器层数
        norm_layer: 归一化层类型
        nonlinearity: 激活函数
        use_spect: 是否使用谱归一化
    """
    def __init__(self, image_nc, ngf, img_f, layers, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineEncoder, self).__init__()
        self.layers = layers
        # 第一层使用7x7大卷积核进行初始特征提取
        self.first = FirstBlock2d(image_nc, ngf, norm_layer, nonlinearity, use_spect)
        # 逐层构建下采样块，通道数逐层加倍
        for i in range(layers):
            in_channels = min(ngf*(2**i), img_f)
            out_channels = min(ngf*(2**(i+1)), img_f)
            model = DownBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            setattr(self, 'down' + str(i), model)
        self.output_nc = out_channels

    def forward(self, x):
        # 逐层编码，保存每层输出用于跳跃连接
        x = self.first(x)
        out=[x]
        for i in range(self.layers):
            model = getattr(self, 'down'+str(i))
            x = model(x)
            out.append(x)
        return out

class FineDecoder(nn.Module):
    """精细解码器
    用于编辑网络（EditingNet）中的图像解码。
    通过逐层上采样恢复图像空间尺寸，
    每层使用ADAIN残差块进行特征精炼，
    并通过跳跃连接融合编码器的多尺度特征。
    
    参数:
        image_nc: 输出图像通道数
        feature_nc: 条件特征通道数
        ngf: 基础特征图数量
        img_f: 特征图数量上限
        layers: 解码器层数
        num_block: 每层的ADAIN残差块数量
        norm_layer: 归一化层类型
        nonlinearity: 激活函数
        use_spect: 是否使用谱归一化
    """
    def __init__(self, image_nc, feature_nc, ngf, img_f, layers, num_block, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineDecoder, self).__init__()
        self.layers = layers
        # 从最深层开始逐层构建解码块
        for i in range(layers)[::-1]:
            in_channels = min(ngf*(2**(i+1)), img_f)
            out_channels = min(ngf*(2**i), img_f)
            up = UpBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            res = FineADAINResBlocks(num_block, in_channels, feature_nc, norm_layer, nonlinearity, use_spect)
            jump = Jump(out_channels, norm_layer, nonlinearity, use_spect)

            setattr(self, 'up' + str(i), up)
            setattr(self, 'res' + str(i), res)            
            setattr(self, 'jump' + str(i), jump)

        # 最终输出层，使用tanh激活将输出限制在[-1, 1]范围
        self.final = FinalBlock2d(out_channels, image_nc, use_spect, 'tanh')

        self.output_nc = out_channels

    def forward(self, x, z):
        # x: 编码器输出的多尺度特征列表, z: 姿态描述符
        # 取出最深层特征作为起始
        out = x.pop()
        for i in range(self.layers)[::-1]:
            res_model = getattr(self, 'res' + str(i))
            up_model = getattr(self, 'up' + str(i))
            jump_model = getattr(self, 'jump' + str(i))
            # ADAIN残差块精炼特征
            out = res_model(out, z)
            # 上采样
            out = up_model(out)
            # 跳跃连接：融合编码器对应层的特征
            out = jump_model(x.pop()) + out
        # 通过最终输出层得到生成图像
        out_image = self.final(out)
        return out_image

class FirstBlock2d(nn.Module):
    """初始卷积块
    使用7x7大卷积核对输入图像进行初始特征提取，
    不进行空间尺寸的改变，主要用于扩展通道数。
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FirstBlock2d, self).__init__()
        # 7x7卷积，padding=3保持空间尺寸不变
        kwargs = {'kernel_size': 7, 'stride': 1, 'padding': 3}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity)


    def forward(self, x):
        out = self.model(x)
        return out  

class DownBlock2d(nn.Module):
    """下采样块
    由3x3卷积 + 归一化 + 激活 + 平均池化组成，
    将特征图的空间尺寸缩小一半（2倍下采样）。
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(DownBlock2d, self).__init__()


        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)
        # 2x2平均池化进行下采样
        pool = nn.AvgPool2d(kernel_size=(2, 2))

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity, pool)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity, pool)

    def forward(self, x):
        out = self.model(x)
        return out 

class UpBlock2d(nn.Module):
    """上采样块
    先通过双线性插值将特征图放大2倍，
    再通过3x3卷积 + 归一化 + 激活进行特征提取。
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(UpBlock2d, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)
        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity)

    def forward(self, x):
        # 先通过双线性插值上采样2倍，再通过卷积层
        out = self.model(F.interpolate(x, scale_factor=2))
        return out

class FineADAINResBlocks(nn.Module):
    """ADAIN残差块序列
    包含多个连续的FineADAINResBlock2d，用于特征精炼。
    通过残差连接和ADAIN条件归一化增强特征表达能力。
    
    参数:
        num_block: 残差块的数量
        input_nc: 输入通道数
        feature_nc: 条件特征通道数
    """
    def __init__(self, num_block, input_nc, feature_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineADAINResBlocks, self).__init__()                                
        self.num_block = num_block
        # 创建多个ADAIN残差块
        for i in range(num_block):
            model = FineADAINResBlock2d(input_nc, feature_nc, norm_layer, nonlinearity, use_spect)
            setattr(self, 'res'+str(i), model)

    def forward(self, x, z):
        # 逐个通过残差块进行特征精炼
        for i in range(self.num_block):
            model = getattr(self, 'res'+str(i))
            x = model(x, z)
        return x     

class Jump(nn.Module):
    """跳跃连接块
    对编码器传来的特征进行3x3卷积处理，
    用于与解码器特征进行跳跃连接融合。
    """
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(Jump, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(input_nc), nonlinearity)

    def forward(self, x):
        out = self.model(x)
        return out          

class FineADAINResBlock2d(nn.Module):
    """ADAIN残差块
    由两个卷积层和对应的ADAIN归一化层组成，
    配合残差连接实现特征的条件精炼。
    ADAIN层根据姿态条件特征调整特征的统计量。
    """
    def __init__(self, input_nc, feature_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineADAINResBlock2d, self).__init__()

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        # 两个3x3卷积层
        self.conv1 = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)
        self.conv2 = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)
        # 两个ADAIN归一化层
        self.norm1 = ADAIN(input_nc, feature_nc)
        self.norm2 = ADAIN(input_nc, feature_nc)

        self.actvn = nonlinearity


    def forward(self, x, z):
        # 第一个卷积 + ADAIN + 激活
        dx = self.actvn(self.norm1(self.conv1(x), z))
        # 第二个卷积 + ADAIN（无激活）
        dx = self.norm2(self.conv2(x), z)
        # 残差连接
        out = dx + x
        return out        

class FinalBlock2d(nn.Module):
    """最终输出块
    由7x7卷积和最终激活函数组成，
    将特征图转换为输出图像。
    支持tanh或sigmoid激活函数。
    """
    def __init__(self, input_nc, output_nc, use_spect=False, tanh_or_sigmoid='tanh'):
        super(FinalBlock2d, self).__init__()

        kwargs = {'kernel_size': 7, 'stride': 1, 'padding':3}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

        # 根据选择使用不同的最终激活函数
        if tanh_or_sigmoid == 'sigmoid':
            out_nonlinearity = nn.Sigmoid()
        else:
            out_nonlinearity = nn.Tanh()            

        self.model = nn.Sequential(conv, out_nonlinearity)
    def forward(self, x):
        out = self.model(x)
        return out          
