"""
IResNet-2060超深骨干网络模块
本模块实现了超深版的IResNet网络（2060层），用于大规模人脸特征提取。

与普通IResNet的主要区别：
1. 网络深度极大增加（约2060层），第三阶段有1024层
2. 使用梯度检查点（gradient checkpointing）技术减少显存占用
3. 在训练时对中间层使用checkpoint_sequential，推理时不使用

适用场景：
- 需要极高模型容量的大规模人脸识别任务
- GPU显存有限但需要深层网络的场景
"""
import torch
from torch import nn

# 确保PyTorch版本支持梯度检查点
assert torch.__version__ >= "1.8.1"
from torch.utils.checkpoint import checkpoint_sequential  # 梯度检查点，用计算换显存

__all__ = ['iresnet2060']


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
    改进的残差基本模块（与iresnet.py相同）
    
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
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05, )  # 第一个批量归一化层
        self.conv1 = conv3x3(inplanes, planes)  # 第一个3x3卷积
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, )  # 第二个批量归一化层
        self.prelu = nn.PReLU(planes)  # PReLU激活函数
        self.conv2 = conv3x3(planes, planes, stride)  # 第二个3x3卷积
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05, )  # 第三个批量归一化层
        self.downsample = downsample  # 下采样层
        self.stride = stride

    def forward(self, x):
        """前向传播"""
        identity = x  # 残差连接的快捷路径
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # 残差连接
        return out


class IResNet(nn.Module):
    """
    超深改进ResNet网络主体（约2060层）
    
    通过梯度检查点技术在训练时节省显存，使得超深网络可以在有限显存上训练。
    """
    fc_scale = 7 * 7  # 全连接层缩放因子

    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        """初始化超深IResNet"""
        super(IResNet, self).__init__()
        self.fp16 = fp16  # 是否使用半精度训练
        self.inplanes = 64  # 初始输入通道数
        self.dilation = 1   # 膨胀率
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        # 四个残差层
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

        # 最后的BN层、Dropout和全连接层
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05, )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        构建一个残差层，由多个block堆叠组成
        """
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
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

    def checkpoint(self, func, num_seg, x):
        """
        梯度检查点包装器
        
        训练时使用checkpoint_sequential减少显存占用（用计算时间换取显存空间）
        推理时直接执行前向传播（不需要计算梯度）
        
        参数:
            func: 要执行的网络层（nn.Sequential）
            num_seg: 梯度检查点的分段数
            x: 输入张量
        返回:
            输出张量
        """
        if self.training:
            return checkpoint_sequential(func, num_seg, x)  # 训练时使用梯度检查点
        else:
            return func(x)  # 推理时直接执行

    def forward(self, x):
        """
        前向传播
        
        注意：layer2和layer3使用梯度检查点，因为它们包含大量残差块（128层和896层）
        """
        with torch.cuda.amp.autocast(self.fp16):  # 半精度自动混合精度
            x = self.conv1(x)   # 初始卷积
            x = self.bn1(x)     # 批量归一化
            x = self.prelu(x)   # 激活函数
            x = self.layer1(x)  # 残差层1
            x = self.checkpoint(self.layer2, 20, x)   # 残差层2（使用梯度检查点，分20段）
            x = self.checkpoint(self.layer3, 100, x)  # 残差层3（使用梯度检查点，分100段）
            x = self.layer4(x)  # 残差层4
            x = self.bn2(x)     # 最终BN
            x = torch.flatten(x, 1)  # 展平
            x = self.dropout(x)      # Dropout
        x = self.fc(x.float() if self.fp16 else x)  # 全连接层
        x = self.features(x)  # 特征归一化
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    """IResNet工厂函数"""
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet2060(pretrained=False, progress=True, **kwargs):
    """
    创建IResNet-2060模型
    
    网络结构分配（总深度约2060层）：
    - layer1: 3个残差块（通道64）
    - layer2: 128个残差块（通道128）- 使用梯度检查点
    - layer3: 896个残差块（通道256）- 使用梯度检查点
    - layer4: 3个残差块（通道512）
    """
    return _iresnet('iresnet2060', IBasicBlock, [3, 128, 1024 - 128, 3], pretrained, progress, **kwargs)
