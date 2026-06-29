import torch
from torch import nn
from torch.nn import functional as F


class Conv2d(nn.Module):
    """带BatchNorm和ReLU激活的二维卷积模块（V2版本）
    
    该模块封装了Conv2d + BatchNorm2d + ReLU的标准卷积块，
    并支持残差连接(residual connection)。当residual=True时，
    输入会与卷积输出相加，实现跳跃连接，有助于梯度传播和网络训练。
    
    V2版本与V1版本结构相同，是网络的基本构建单元。
    
    参数:
        cin: 输入通道数
        cout: 输出通道数
        kernel_size: 卷积核大小
        stride: 卷积步长
        padding: 填充大小
        residual: 是否使用残差连接，默认False
    """
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 构建卷积块：Conv2d + BatchNorm2d
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        # ReLU激活函数
        self.act = nn.ReLU()
        # 残差连接标志
        self.residual = residual

    def forward(self, x):
        # 通过卷积块进行特征提取
        out = self.conv_block(x)
        # 如果启用了残差连接，将输入与输出相加
        if self.residual:
            out += x
        # 经过ReLU激活函数输出
        return self.act(out)


class nonorm_Conv2d(nn.Module):
    """不带BatchNorm的二维卷积模块（V2版本）
    
    与Conv2d不同，该模块不使用BatchNorm，而是使用LeakyReLU激活。
    主要用于判别器(discriminator)网络中，避免BatchNorm对判别器训练的不稳定影响。
    LeakyReLU的负斜率为0.01，允许负值以较小的比例通过。
    
    参数:
        cin: 输入通道数
        cout: 输出通道数
        kernel_size: 卷积核大小
        stride: 卷积步长
        padding: 填充大小
        residual: 残差连接标志（本类未使用但保留参数接口）
    """
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 构建卷积块：仅包含Conv2d，不含BatchNorm
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        # LeakyReLU激活函数，负斜率为0.01，原地操作节省内存
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        # 通过卷积块进行特征提取
        out = self.conv_block(x)
        # 经过LeakyReLU激活函数输出
        return self.act(out)


class Conv2dTranspose(nn.Module):
    """带BatchNorm和ReLU激活的二维转置卷积模块（V2版本）
    
    转置卷积(反卷积)用于上采样操作，在解码器中将低分辨率特征图
    恢复到高分辨率。该模块封装了ConvTranspose2d + BatchNorm2d + ReLU。
    
    参数:
        cin: 输入通道数
        cout: 输出通道数
        kernel_size: 卷积核大小
        stride: 卷积步长
        padding: 填充大小
        output_padding: 输出填充大小，用于控制输出尺寸
    """
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 构建转置卷积块：ConvTranspose2d + BatchNorm2d
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        # ReLU激活函数
        self.act = nn.ReLU()

    def forward(self, x):
        # 通过转置卷积块进行上采样
        out = self.conv_block(x)
        # 经过ReLU激活函数输出
        return self.act(out)
