"""Deep3DFaceRecon_pytorch的深度神经网络定义脚本

该脚本定义了用于面部重建的深度神经网络，包括ResNet骨干网络、
面部重建网络包装器和人脸识别网络包装器。
"""

import os
import numpy as np
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from .arcface_torch.backbones import get_model
from kornia.geometry import warp_affine

def resize_n_crop(image, M, dsize=112):
    """调整图像大小并裁剪
    
    Args:
        image: 输入图像，形状 (b, c, h, w)
        M: 仿射变换矩阵，形状 (b, 2, 3)
        dsize: 目标尺寸，默认112
    
    Returns:
        裁剪后的图像
    """
    return warp_affine(image, M, dsize=(dsize, dsize), align_corners=True)

def filter_state_dict(state_dict, remove_name='fc'):
    """过滤状态字典，移除指定名称的层
    
    Args:
        state_dict: 模型状态字典
        remove_name: 要移除的层名称，默认为'fc'
    
    Returns:
        过滤后的状态字典
    """
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict

def get_scheduler(optimizer, opt):
    """返回学习率调度器
    
    根据配置的学习率策略返回相应的调度器。
    
    Args:
        optimizer: 优化器
        opt: 配置选项类
    
    Returns:
        scheduler: 学习率调度器
    
    支持的策略：
        - linear: 线性衰减
        - step: 步进衰减
        - plateau: 自适应衰减
        - cosine: 余弦退火
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        # 步进衰减：每隔一定epoch衰减
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epochs, gamma=0.2)
    elif opt.lr_policy == 'plateau':
        # 自适应衰减：当指标停止改善时衰减
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        # 余弦退火：使用余弦函数衰减学习率
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_net_recon(net_recon, use_last_fc=False, init_path=None):
    """定义面部重建网络
    
    Args:
        net_recon: 网络结构名称
        use_last_fc: 是否使用最后的全连接层
        init_path: 预训练权重路径
    
    Returns:
        面部重建网络包装器
    """
    return ReconNetWrapper(net_recon, use_last_fc=use_last_fc, init_path=init_path)

def define_net_recog(net_recog, pretrained_path=None):
    """定义人脸识别网络
    
    Args:
        net_recog: 人脸识别网络结构名称
        pretrained_path: 预训练权重路径
    
    Returns:
        人脸识别网络包装器（评估模式）
    """
    net = RecogNetWrapper(net_recog=net_recog, pretrained_path=pretrained_path)
    net.eval()
    return net

class ReconNetWrapper(nn.Module):
    """面部重建网络包装器
    
    包装ResNet骨干网络，添加多个1x1卷积层用于预测面部系数。
    """
    fc_dim=257  # 最终输出维度：身份80 + 表情64 + 纹理80 + 旋转3 + 光照27 + 平移3
    
    def __init__(self, net_recon, use_last_fc=False, init_path=None):
        """初始化重建网络包装器
        
        Args:
            net_recon: 骨干网络结构名称
            use_last_fc: 是否使用最后的全连接层
            init_path: 预训练权重路径
        """
        super(ReconNetWrapper, self).__init__()
        self.use_last_fc = use_last_fc
        # 根据网络名称获取对应的构建函数和输出维度
        if net_recon not in func_dict:
            return  NotImplementedError('network [%s] is not implemented', net_recon)
        func, last_dim = func_dict[net_recon]
        # 创建骨干网络
        backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)
        # 如果指定了预训练权重路径，加载权重
        if init_path and os.path.isfile(init_path):
            state_dict = filter_state_dict(torch.load(init_path, map_location='cpu'))
            backbone.load_state_dict(state_dict)
            print("loading init net_recon %s from %s" %(net_recon, init_path))
        self.backbone = backbone
        if not use_last_fc:
            # 如果不使用最后的全连接层，创建多个1x1卷积层
            # 每个卷积层预测一个系数组件
            self.final_layers = nn.ModuleList([
                conv1x1(last_dim, 80, bias=True), # 身份系数层
                conv1x1(last_dim, 64, bias=True), # 表情系数层
                conv1x1(last_dim, 80, bias=True), # 纹理系数层
                conv1x1(last_dim, 3, bias=True),  # 旋转角度层
                conv1x1(last_dim, 27, bias=True), # 光照参数层
                conv1x1(last_dim, 2, bias=True),  # 平移xy层
                conv1x1(last_dim, 1, bias=True)   # 平移z层
            ])
            # 初始化所有卷积层权重和偏置为0
            for m in self.final_layers:
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像，形状 (B, 3, H, W)
        
        Returns:
            输出系数向量，形状 (B, 257)
        """
        # 通过骨干网络提取特征
        x = self.backbone(x)
        if not self.use_last_fc:
            # 通过多个1x1卷积层预测各个系数组件
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            # 拼接所有输出并展平
            x = torch.flatten(torch.cat(output, dim=1), 1)
        return x


class RecogNetWrapper(nn.Module):
    """人脸识别网络包装器
    
    包装ArcFace人脸识别网络，用于计算感知损失。
    """
    
    def __init__(self, net_recog, pretrained_path=None, input_size=112):
        """初始化人脸识别网络包装器
        
        Args:
            net_recog: 人脸识别网络结构名称（如'r50'）
            pretrained_path: 预训练权重路径
            input_size: 输入图像尺寸，默认112
        """
        super(RecogNetWrapper, self).__init__()
        # 加载人脸识别模型
        net = get_model(name=net_recog, fp16=False)
        if pretrained_path:
            # 加载预训练权重
            state_dict = torch.load(pretrained_path, map_location='cpu')
            net.load_state_dict(state_dict)
            print("loading pretrained net_recog %s from %s" %(net_recog, pretrained_path))
        # 冻结所有参数，不进行梯度更新
        for param in net.parameters():
            param.requires_grad = False
        self.net = net
        # 预处理函数：将[0,1]范围转换为[-1,1]范围
        self.preprocess = lambda x: 2 * x - 1
        self.input_size=input_size
        
    def forward(self, image, M):
        """前向传播
        
        Args:
            image: 输入图像，形状 (B, 3, H, W)
            M: 仿射变换矩阵，形状 (B, 2, 3)
        
        Returns:
            id_feature: 人脸识别特征，L2归一化后
        """
        # 裁剪并对齐面部区域
        image = self.preprocess(resize_n_crop(image, M, self.input_size))
        # 提取并归一化特征
        id_feature = F.normalize(self.net(image), dim=-1, p=2)
        return id_feature


# 以下为ResNet实现（改编自PyTorch官方实现）
# 参考：https://github.com/pytorch/vision/edit/master/torchvision/models/resnet.py
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


# 各种ResNet模型的预训练权重下载链接
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """创建3x3卷积层
    
    Args:
        in_planes: 输入通道数
        out_planes: 输出通道数
        stride: 步长
        groups: 分组卷积数
        dilation: 膨胀卷积率
    
    Returns:
        3x3卷积层
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """创建1x1卷积层
    
    Args:
        in_planes: 输入通道数
        out_planes: 输出通道数
        stride: 步长
        bias: 是否使用偏置
    
    Returns:
        1x1卷积层
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    """ResNet基础残差块
    
    用于ResNet-18和ResNet-34网络。
    """
    expansion: int = 1  # 输出通道数扩展因子

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """初始化基础残差块
        
        Args:
            inplanes: 输入通道数
            planes: 输出通道数
            stride: 步长
            downsample: 下采样层
            groups: 分组卷积数
            base_width: 基础宽度
            dilation: 膨胀卷积率
            norm_layer: 归一化层
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # 两个3x3卷积层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """前向传播
        
        Args:
            x: 输入特征图
        
        Returns:
            输出特征图
        """
        # 保存输入用于残差连接
        identity = x

        # 第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，对输入进行处理
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet瓶颈残差块
    
    用于ResNet-50、ResNet-101和ResNet-152网络。
    包含三个卷积层：1x1 -> 3x3 -> 1x1
    """
    # Bottleneck在torchvision中将步长放在3x3卷积(self.conv2)
    # 而原始实现将步长放在第一个1x1卷积(self.conv1)
    # 参考："Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385
    # 此变体也称为ResNet V1.5，根据NVIDIA的实现提高了准确率

    expansion: int = 4  # 输出通道数扩展因子

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """初始化瓶颈残差块
        
        Args:
            inplanes: 输入通道数
            planes: 输出通道数
            stride: 步长
            downsample: 下采样层
            groups: 分组卷积数
            base_width: 基础宽度
            dilation: 膨胀卷积率
            norm_layer: 归一化层
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 计算实际宽度
        width = int(planes * (base_width / 64.)) * groups
        # 三个卷积层：1x1降维 -> 3x3特征提取 -> 1x1升维
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """前向传播
        
        Args:
            x: 输入特征图
        
        Returns:
            输出特征图
        """
        identity = x

        # 第一个1x1卷积：降维
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个3x3卷积：特征提取
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三个1x1卷积：升维
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果需要下采样，对输入进行处理
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet网络实现
    
    支持ResNet-18、ResNet-34、ResNet-50、ResNet-101、ResNet-152
    以及ResNeXt和Wide ResNet变体。
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        use_last_fc: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """初始化ResNet网络
        
        Args:
            block: 残差块类型（BasicBlock或Bottleneck）
            layers: 每个阶段的残差块数量
            num_classes: 分类类别数
            zero_init_residual: 是否将残差分支的最后一个BN层初始化为0
            use_last_fc: 是否使用最后的全连接层
            groups: 分组卷积数
            width_per_group: 每组的宽度
            replace_stride_with_dilation: 是否用膨胀卷积替代步长
            norm_layer: 归一化层类型
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # 初始输入通道数
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # 每个元素表示是否用膨胀卷积替代2x2步长
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.use_last_fc = use_last_fc
        self.groups = groups
        self.base_width = width_per_group
        # 第一个卷积层：7x7，步长2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化层：3x3，步长2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 四个残差阶段
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if self.use_last_fc:
            # 可选的全连接层用于分类
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 使用Kaiming初始化卷积层权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



        # 将每个残差分支的最后一个BN层初始化为0
        # 这样残差分支从零开始，每个残差块初始时相当于恒等映射
        # 根据论文，这可以将模型准确率提高0.2~0.3%
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """构建残差层
        
        Args:
            block: 残差块类型
            planes: 输出通道数
            blocks: 残差块数量
            stride: 步长
            dilate: 是否使用膨胀卷积
        
        Returns:
            残差层
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 如果需要改变通道数或空间尺寸，创建下采样层
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 添加第一个残差块（可能包含下采样）
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        # 添加剩余的残差块
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """前向传播实现
        
        Args:
            x: 输入图像
        
        Returns:
            特征图
        """
        # See note [TorchScript super()]
        # 第一个卷积阶段
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 四个残差阶段
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = self.avgpool(x)
        if self.use_last_fc:
            # 展平并通过全连接层
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """前向传播
        
        Args:
            x: 输入图像
        
        Returns:
            特征图
        """
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    """构建ResNet模型
    
    Args:
        arch: 网络架构名称
        block: 残差块类型
        layers: 每个阶段的残差块数量
        pretrained: 是否加载预训练权重
        progress: 是否显示下载进度
    
    Returns:
        ResNet模型
    """
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        # 加载预训练权重
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18模型
    
    参考论文："Deep Residual Learning for Image Recognition"
    
    Args:
        pretrained: 是否加载预训练权重
        progress: 是否显示下载进度
    
    Returns:
        ResNet-18模型
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34模型
    
    参考论文："Deep Residual Learning for Image Recognition"
    
    Args:
        pretrained: 是否加载预训练权重
        progress: 是否显示下载进度
    
    Returns:
        ResNet-34模型
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50模型
    
    参考论文："Deep Residual Learning for Image Recognition"
    
    Args:
        pretrained: 是否加载预训练权重
        progress: 是否显示下载进度
    
    Returns:
        ResNet-50模型
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101模型
    
    参考论文："Deep Residual Learning for Image Recognition"
    
    Args:
        pretrained: 是否加载预训练权重
        progress: 是否显示下载进度
    
    Returns:
        ResNet-101模型
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152模型
    
    参考论文："Deep Residual Learning for Image Recognition"
    
    Args:
        pretrained: 是否加载预训练权重
        progress: 是否显示下载进度
    
    Returns:
        ResNet-152模型
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNeXt-50 32x4d模型
    
    参考论文："Aggregated Residual Transformation for Deep Neural Networks"
    
    Args:
        pretrained: 是否加载预训练权重
        progress: 是否显示下载进度
    
    Returns:
        ResNeXt-50模型
    """
    kwargs['groups'] = 32  # 32个分组
    kwargs['width_per_group'] = 4  # 每组4个通道
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNeXt-101 32x8d模型
    
    参考论文："Aggregated Residual Transformation for Deep Neural Networks"
    
    Args:
        pretrained: 是否加载预训练权重
        progress: 是否显示下载进度
    
    Returns:
        ResNeXt-101模型
    """
    kwargs['groups'] = 32  # 32个分组
    kwargs['width_per_group'] = 8  # 每组8个通道
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """Wide ResNet-50-2模型
    
    参考论文："Wide Residual Networks"
    
    模型结构与ResNet相同，但瓶颈层的通道数是ResNet的2倍。
    例如：ResNet-50最后一个块的通道数为2048-512-2048，
    Wide ResNet-50-2为2048-1024-2048。
    
    Args:
        pretrained: 是否加载预训练权重
        progress: 是否显示下载进度
    
    Returns:
        Wide ResNet-50-2模型
    """
    kwargs['width_per_group'] = 64 * 2  # 宽度加倍
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """Wide ResNet-101-2模型
    
    参考论文："Wide Residual Networks"
    
    模型结构与ResNet相同，但瓶颈层的通道数是ResNet的2倍。
    
    Args:
        pretrained: 是否加载预训练权重
        progress: 是否显示下载进度
    
    Returns:
        Wide ResNet-101-2模型
    """
    kwargs['width_per_group'] = 64 * 2  # 宽度加倍
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


# 网络名称到构建函数的映射字典
func_dict = {
    'resnet18': (resnet18, 512),   # ResNet-18，输出512维特征
    'resnet50': (resnet50, 2048)   # ResNet-50，输出2048维特征
}
