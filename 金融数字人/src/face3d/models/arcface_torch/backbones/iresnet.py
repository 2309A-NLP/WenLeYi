"""
IResNet（改进ResNet）骨干网络模块
本模块实现了改进版的ResNet网络（IResNet），用于人脸特征提取（ArcFace）。

与标准ResNet的主要区别：
1. 在BN后使用PReLU激活函数代替ReLU
2. 使用Pre-activation结构（BN-Conv-BN-Act-Conv-BN）
3. 支持半精度训练（FP16）
4. 输出固定维度的特征向量

支持的网络深度：18, 34, 50, 100, 200层
"""
import torch
from torch import nn

# 模块导出列表
__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3卷积层（带填充），用于特征提取"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1卷积层，用于通道数调整和降采样"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    """
    改进的残差基本模块
    
    结构: BN -> Conv3x3 -> BN -> PReLU -> Conv3x3 -> BN + shortcut
    """
    expansion = 1  # 输出通道扩展倍数
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        """初始化残差模块"""
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # 定义各层
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)  # 第一个批量归一化层
        self.conv1 = conv3x3(inplanes, planes)  # 第一个3x3卷积
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)  # 第二个批量归一化层
        self.prelu = nn.PReLU(planes)  # PReLU激活函数
        self.conv2 = conv3x3(planes, planes, stride)  # 第二个3x3卷积（可带步长降采样）
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)  # 第三个批量归一化层
        self.downsample = downsample  # 下采样层（当维度不匹配时使用）
        self.stride = stride

    def forward(self, x):
        """前向传播"""
        identity = x  # 保存输入用于残差连接
        out = self.bn1(x)   # BN -> Conv -> BN -> PReLU -> Conv -> BN
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)  # 必要时对shortcut进行下采样
        out += identity  # 残差连接
        return out


class IResNet(nn.Module):
    """
    改进的ResNet网络主体
    
    由多个IBasicBlock堆叠组成，输出固定维度的人脸特征向量。
    支持半精度训练（FP16）以减少显存占用和加速训练。
    """
    fc_scale = 7 * 7  # 全连接层的缩放因子（用于展平后的特征维度计算）
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        """初始化IResNet"""
        super(IResNet, self).__init__()
        self.fp16 = fp16  # 是否使用半精度浮点数
        self.inplanes = 64  # 当前输入通道数
        self.dilation = 1   # 膨胀率

        # 检查replace_stride_with_dilation参数
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # 初始卷积层：3通道输入 -> 64通道
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        # 四个残差层（通道数分别为64, 128, 256, 512）
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 最后的批量归一化和分类层
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)  # 全连接层
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)  # 输出特征归一化
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False  # 输出归一化层不参与训练

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)  # 卷积层使用正态分布初始化
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)  # BN层权重初始化为1
                nn.init.constant_(m.bias, 0)    # BN层偏置初始化为0

        # 零初始化残差分支（可选，有助于训练稳定性）
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        构建一个残差层，由多个block堆叠组成
        
        参数:
            block: 残差模块类型
            planes: 输出通道数
            blocks: 模块数量
            stride: 步长（用于降采样）
            dilate: 是否使用膨胀卷积
        返回:
            nn.Sequential: 由多个残差模块组成的序列
        """
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 当步长不为1或通道数不匹配时，需要下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像张量，形状为 (N, 3, H, W)
        返回:
            人脸特征向量，形状为 (N, num_features)
        """
        with torch.cuda.amp.autocast(self.fp16):  # 半精度自动混合精度
            x = self.conv1(x)   # 初始卷积
            x = self.bn1(x)     # 批量归一化
            x = self.prelu(x)   # 激活函数
            x = self.layer1(x)  # 残差层1
            x = self.layer2(x)  # 残差层2
            x = self.layer3(x)  # 残差层3
            x = self.layer4(x)  # 残差层4
            x = self.bn2(x)     # 最终批量归一化
            x = torch.flatten(x, 1)  # 展平特征图
            x = self.dropout(x)      # Dropout正则化
        x = self.fc(x.float() if self.fp16 else x)  # 全连接层（转为float32）
        x = self.features(x)  # 特征归一化
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    """
    IResNet工厂函数
    
    参数:
        arch: 架构名称（用于错误提示）
        block: 残差模块类型
        layers: 各层的模块数量
        pretrained: 是否加载预训练权重（暂不支持）
        progress: 是否显示下载进度
    返回:
        IResNet模型实例
    """
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    """创建IResNet-18模型（各层模块数: [2, 2, 2, 2]）"""
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    """创建IResNet-34模型（各层模块数: [3, 4, 6, 3]）"""
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    """创建IResNet-50模型（各层模块数: [3, 4, 14, 3]）"""
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    """创建IResNet-100模型（各层模块数: [3, 13, 30, 3]）"""
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    """创建IResNet-200模型（各层模块数: [6, 26, 60, 6]）"""
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)
