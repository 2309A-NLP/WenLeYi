# audio_encoder.py - 音频编码器网络定义
# 该模块定义了用于将梅尔频谱图编码为音频嵌入向量的卷积神经网络
# 网络架构参考了Wav2Lip模型的音频编码器设计
# 输入：逐帧的梅尔频谱图 [B, T, 1, 80, 16]
# 输出：音频嵌入向量 [B, T, 512]

import torch
from torch import nn
from torch.nn import functional as F

class Conv2d(nn.Module):
    """
    基础2D卷积模块，包含卷积层、批归一化和ReLU激活。
    支持可选的残差连接。
    
    参数说明：
        cin: 输入通道数
        cout: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        residual: 是否使用残差连接
    """
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 卷积 + 批归一化
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()       # ReLU激活函数
        self.residual = residual   # 残差连接标志

    def forward(self, x):
        out = self.conv_block(x)
        # 残差连接：将输入直接加到输出上，有助于梯度流动和特征复用
        if self.residual:
            out += x
        return self.act(out)

class AudioEncoder(nn.Module):
    """
    音频编码器：将梅尔频谱图序列编码为高维音频嵌入向量。
    
    网络结构：
    - 采用逐层递增通道数（1->32->64->128->256->512）的CNN架构
    - 使用残差卷积块增强特征提取能力
    - 最终将80x16的梅尔频谱图压缩为1x1的512维向量
    
    参数说明：
        wav2lip_checkpoint: Wav2Lip预训练模型的路径（当前代码中加载部分已注释）
        device: 计算设备
    """
    def __init__(self, wav2lip_checkpoint, device):
        super(AudioEncoder, self).__init__()

        # ===== 音频编码器网络结构 =====
        self.audio_encoder = nn.Sequential(
            # 第一组：1通道 -> 32通道，保持空间尺寸不变
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            # 第二组：32通道 -> 64通道，时间维度下采样3倍
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            # 第三组：64通道 -> 128通道，空间维度下采样3倍
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            # 第四组：128通道 -> 256通道，非对称下采样
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            # 第五组：256通道 -> 512通道，最终输出1x1空间尺寸
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        # 加载预训练的音频编码器权重（来自Wav2Lip模型）
        # 注意：当前代码中此部分已注释掉，不加载预训练权重
        # wav2lip_state_dict = torch.load(wav2lip_checkpoint, map_location=torch.device(device))['state_dict']
        # state_dict = self.audio_encoder.state_dict()

        # for k,v in wav2lip_state_dict.items():
        #     if 'audio_encoder' in k:
        #         state_dict[k.replace('module.audio_encoder.', '')] = v
        # self.audio_encoder.load_state_dict(state_dict)


    def forward(self, audio_sequences):
        """
        前向传播：将梅尔频谱图序列编码为音频嵌入向量。
        
        处理流程：
        1. 将 [B, T, 1, 80, 16] 的输入reshape为 [B*T, 1, 80, 16]
           这样可以批量处理所有时间步的梅尔频谱图
        2. 通过CNN编码器得到 [B*T, 512, 1, 1] 的特征
        3. reshape回 [B, T, 512] 的序列格式
        
        参数：
            audio_sequences: 梅尔频谱图，形状 [B, T, 1, 80, 16]
        
        返回：
            audio_embedding: 音频嵌入向量，形状 [B, T, 512]
        """
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)  # batch size

        # 将序列维度和batch维度合并：[B, T, 1, 80, 16] -> [B*T, 1, 80, 16]
        # 这样每个时间步的梅尔频谱图都作为一个独立样本进行编码
        audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)

        # 通过编码器：[B*T, 1, 80, 16] -> [B*T, 512, 1, 1]
        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1
        dim = audio_embedding.shape[1]  # 特征维度=512
        
        # 恢复batch和时间维度：[B*T, 512, 1, 1] -> [B, T, 512, 1, 1]
        audio_embedding = audio_embedding.reshape((B, -1, dim, 1, 1))

        # 去掉最后两个单维度：[B, T, 512, 1, 1] -> [B, T, 512]
        return audio_embedding.squeeze(-1).squeeze(-1) #B seq_len+1 512 
