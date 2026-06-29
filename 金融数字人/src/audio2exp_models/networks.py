# networks.py - audio2exp模型的网络结构定义
# 定义了音频编码器和表情生成器的网络架构
# 主要包含：基础卷积模块 Conv2d 和 主生成器 SimpleWrapperV2

import torch
import torch.nn.functional as F
from torch import nn

class Conv2d(nn.Module):
    """
    基础2D卷积模块，封装了卷积层 + 批归一化 + 激活函数。
    
    该模块是整个网络中最基本的构建单元，支持残差连接（residual connection）。
    
    参数说明：
        cin: 输入通道数
        cout: 输出通道数
        kernel_size: 卷积核大小
        stride: 卷积步长
        padding: 填充大小
        residual: 是否使用残差连接（将输入直接加到输出上）
        use_act: 是否使用ReLU激活函数（False时仅返回卷积+BN的结果）
    """
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, use_act = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 卷积块：2D卷积 + 批归一化（BatchNorm2d）
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()       # ReLU激活函数
        self.residual = residual   # 是否启用残差连接标志
        self.use_act = use_act     # 是否使用激活函数标志

    def forward(self, x):
        # 执行卷积+BN操作
        out = self.conv_block(x)
        # 如果启用残差连接，将原始输入与卷积输出相加
        if self.residual:
            out += x
        
        # 根据标志决定是否应用激活函数
        if self.use_act:
            return self.act(out)
        else:
            return out

class SimpleWrapperV2(nn.Module):
    """
    SimpleWrapperV2 - audio2exp的核心生成器网络。
    
    该网络包含两个部分：
    1. 音频编码器（audio_encoder）：一个深度卷积神经网络，将梅尔频谱图
       编码为512维的音频嵌入向量。该编码器参考了Wav2Lip的架构设计。
    2. 映射层（mapping1）：将音频嵌入、参考表情系数和比例因子拼接后，
       映射为64维的表情系数输出。
    
    输入：
        x: 音频梅尔频谱图，形状 [bs, 1, 80, 16]
        ref: 参考帧的表情系数，形状 [bs, T, 64]
        ratio: 比例因子，形状 [bs, T, 1]
    
    输出：
        out: 预测的表情系数，形状 [bs, T, 64]
    """
    def __init__(self) -> None:
        super().__init__()
        
        # ===== 音频编码器网络 =====
        # 采用逐层增加通道数、减小空间尺寸的设计策略
        # 从1通道的梅尔频谱图逐步编码为512维的特征向量
        self.audio_encoder = nn.Sequential(
            # 第一层：输入1通道 -> 32通道，保持空间尺寸
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # 残差卷积块，保持32通道，增强特征提取
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            # 第二层：32通道 -> 64通道，时间维度下采样3倍
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            # 第三层：64通道 -> 128通道，空间维度下采样3倍
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            # 第四层：128通道 -> 256通道，非对称下采样（时间3倍，频率2倍）
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            # 第五层：256通道 -> 512通道，输出1x1的空间尺寸
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            # 1x1卷积进一步整合512通道特征
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )

        #### 加载预训练的音频编码器权重（来自Wav2Lip模型）
        # 注意：当前代码中此部分已注释掉，使用随机初始化的权重
        #self.audio_encoder = self.audio_encoder.to(device)  
        '''
        wav2lip_state_dict = torch.load('/apdcephfs_cq2/share_1290939/wenxuazhang/checkpoints/wav2lip.pth')['state_dict']
        state_dict = self.audio_encoder.state_dict()

        for k,v in wav2lip_state_dict.items():
            if 'audio_encoder' in k:
                print('init:', k)
                state_dict[k.replace('module.audio_encoder.', '')] = v
        self.audio_encoder.load_state_dict(state_dict)
        '''

        # ===== 映射层 =====
        # 输入维度 = 512（音频嵌入）+ 64（参考表情系数）+ 1（比例因子）= 577
        # 输出维度 = 64（表情系数）
        self.mapping1 = nn.Linear(512+64+1, 64)
        #self.mapping2 = nn.Linear(30, 64)
        #nn.init.constant_(self.mapping1.weight, 0.)
        # 将偏置初始化为零，确保初始输出接近零，有利于训练稳定性
        nn.init.constant_(self.mapping1.bias, 0.)

    def forward(self, x, ref, ratio):
        """
        前向传播函数。
        
        处理流程：
        1. 通过音频编码器将梅尔频谱图编码为512维向量
        2. 将编码结果、参考表情系数和比例因子拼接
        3. 通过映射层得到64维表情系数
        """
        # 音频编码：输入 [bs*10, 1, 80, 16] -> 输出 [bs*10, 512, 1, 1] -> [bs*10, 512]
        x = self.audio_encoder(x).view(x.size(0), -1)
        # 将参考表情系数reshape为 [bs*10, 64]
        ref_reshape = ref.reshape(x.size(0), -1)
        # 将比例因子reshape为 [bs*10, 1]
        ratio = ratio.reshape(x.size(0), -1)
        
        # 拼接所有特征并映射为表情系数：[bs*10, 577] -> [bs*10, 64]
        y = self.mapping1(torch.cat([x, ref_reshape, ratio], dim=1)) 
        # 恢复为原始的batch和时间维度：[bs, 10, 64]
        out = y.reshape(ref.shape[0], ref.shape[1], -1) #+ ref # resudial
        return out
