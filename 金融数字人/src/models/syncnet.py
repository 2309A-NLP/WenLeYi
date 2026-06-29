import torch
from torch import nn
from torch.nn import functional as F

# 从同级conv模块导入带BatchNorm的Conv2d卷积块
from .conv import Conv2d


class SyncNet_color(nn.Module):
    """音视频同步性判断网络 (SyncNet)
    
    SyncNet用于判断音频特征和视频面部特征是否同步（即嘴型与声音是否匹配）。
    它分别编码音频和面部图像特征，然后通过余弦相似度（L2归一化后点积）
    来计算音视频之间的同步分数。
    
    网络结构包含两个分支：
    1. 面部编码器(face_encoder): 处理多帧面部图像（3帧RGB图像，共9通道，
       加上额外的6通道特征，共15通道输入），提取面部空间特征
    2. 音频编码器(audio_encoder): 处理梅尔频谱图音频特征，提取音频时频特征
    
    两个编码器的输出都被展平并L2归一化，用于后续的相似度计算。
    """
    def __init__(self):
        super(SyncNet_color, self).__init__()

        # ========== 面部编码器 ==========
        # 输入: 15通道的面部图像序列（3帧RGB图像拼接 + 6通道额外特征）
        # 通过多层卷积逐步降低空间分辨率，增加通道数，提取深层面部特征
        self.face_encoder = nn.Sequential(
            # 第1层: 输入15通道 -> 32通道, 7x7大卷积核捕获较大感受野
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            # 第2层: 32通道 -> 64通道, 水平方向步长为2进行下采样
            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            # 残差卷积块: 保持64通道不变，增强特征表达
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            # 第3层: 64通道 -> 128通道, 2x2下采样
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            # 多个残差卷积块进一步提取特征
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            # 第4层: 128通道 -> 256通道, 2x2下采样
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            # 第5层: 256通道 -> 512通道, 2x2下采样
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            # 第6层: 512通道 -> 512通道, 最终下采样到1x1空间维度
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            # 1x1卷积整合通道信息
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        # ========== 音频编码器 ==========
        # 输入: 1通道的梅尔频谱图（时间-频率表示）
        # 通过多层卷积提取音频的时频特征
        self.audio_encoder = nn.Sequential(
            # 第1层: 1通道 -> 32通道, 标准3x3卷积
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # 残差卷积块
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            # 第2层: 32通道 -> 64通道, 时间维度步长为3进行下采样
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            # 第3层: 64通道 -> 128通道, 时间和频率维度步长为3进行下采样
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            # 第4层: 128通道 -> 256通道, 混合步长下采样
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            # 第5层: 256通道 -> 512通道, 最终压缩到1x1
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            # 1x1卷积整合通道信息
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): 
        # audio_sequences: 音频梅尔频谱序列, 形状为 (B, dim, T)
        # face_sequences: 面部图像序列, 形状为 (B, C, H, W)
        
        # 分别通过面部编码器和音频编码器提取特征
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        # 将特征展平为一维向量，形状变为 (B, 512)
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        # L2归一化: 将特征向量归一化到单位球面上，便于计算余弦相似度
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        # 返回归一化后的音频和面部嵌入向量，后续可通过点积计算同步分数
        return audio_embedding, face_embedding
