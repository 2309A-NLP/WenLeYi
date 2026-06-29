import torch
from torch import nn
from torch.nn import functional as F
import math

# 从同级conv模块导入自定义的卷积块
# Conv2dTranspose: 带BatchNorm的转置卷积，用于解码器上采样
# Conv2d: 带BatchNorm的普通卷积，用于编码器下采样和特征提取
# nonorm_Conv2d: 不带BatchNorm的卷积，用于判别器网络
from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class Wav2Lip(nn.Module):
    """Wav2Lip 嘴型生成器网络
    
    Wav2Lip是一个基于音频驱动的嘴型同步生成网络。它接收音频梅尔频谱图和
    面部图像作为输入，生成与音频同步的嘴型视频帧。
    
    网络架构采用编码器-解码器结构，包含U-Net风格的跳跃连接(skip connection)：
    
    1. 面部编码器(face_encoder_blocks): 7个编码块，逐层降低空间分辨率
       (96x96 -> 48x48 -> 24x24 -> 12x12 -> 6x6 -> 3x3 -> 1x1)
       同时增加特征通道数(16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 512)
       
    2. 音频编码器(audio_encoder): 将梅尔频谱图编码为512维的音频嵌入向量
    
    3. 面部解码器(face_decoder_blocks): 7个解码块，逐层恢复空间分辨率
       通过跳跃连接融合编码器的多尺度特征
    
    4. 输出块(output_block): 将解码器输出映射为3通道RGB图像
    """
    def __init__(self):
        super(Wav2Lip, self).__init__()

        # ========== 面部编码器 ==========
        # 使用ModuleList存储多个编码块，输入为6通道（3帧RGB图像拼接）
        self.face_encoder_blocks = nn.ModuleList([
            # 第1块: 6通道 -> 16通道, 空间分辨率 96x96
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)), # 96,96

            # 第2块: 16通道 -> 32通道, 空间分辨率降至 48x48
            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 48,48
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            # 第3块: 32通道 -> 64通道, 空间分辨率降至 24x24
            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 24,24
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            # 第4块: 64通道 -> 128通道, 空间分辨率降至 12x12
            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 12,12
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            # 第5块: 128通道 -> 256通道, 空间分辨率降至 6x6
            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 6,6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            # 第6块: 256通道 -> 512通道, 空间分辨率降至 3x3
            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),
            
            # 第7块: 512通道 -> 512通道, 空间分辨率降至 1x1（瓶颈层）
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        # ========== 音频编码器 ==========
        # 将1通道的梅尔频谱图编码为512维的音频嵌入向量
        self.audio_encoder = nn.Sequential(
            # 第1层: 1通道 -> 32通道
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            # 第2层: 32通道 -> 64通道, 时间维度下采样
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            # 第3层: 64通道 -> 128通道
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            # 第4层: 128通道 -> 256通道
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            # 第5层: 256通道 -> 512通道, 最终输出形状为 (B, 512, 1, 1)
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        # ========== 面部解码器 ==========
        # 使用转置卷积逐步上采样，通过跳跃连接融合编码器的多尺度特征
        self.face_decoder_blocks = nn.ModuleList([
            # 第1块: 512通道 -> 512通道, 瓶颈层处理
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            # 第2块: 1024通道(512+512跳跃连接) -> 512通道, 空间分辨率 3x3
            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            # 第3块: 上采样到 6x6
            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            # 第4块: 上采样到 12x12, 通道数从768降为384
            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            # 第5块: 上采样到 24x24
            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            # 第6块: 上采样到 48x48
            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            # 第7块: 上采样到 96x96, 恢复原始空间分辨率
            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 96,96

        # ========== 输出块 ==========
        # 将解码器的80通道特征映射为3通道RGB图像
        # 使用Sigmoid激活将输出值限制在[0, 1]范围内（像素值归一化）
        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences: 音频梅尔频谱序列, 形状为 (B, T, 1, 80, 16)
        # face_sequences: 面部图像序列, 形状为 (B, C, T, H, W) 或 (B, C, H, W)
        B = audio_sequences.size(0)

        # 检测输入维度，支持多帧输入（5维）和单帧输入（4维）
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            # 多帧模式: 将时间维度展开到batch维度进行批量处理
            # audio: (B, T, 1, 80, 16) -> (B*T, 1, 80, 16)
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            # face: (B, C, T, H, W) -> (B*T, C, H, W)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        # 通过音频编码器得到音频嵌入向量, 形状为 (B*T, 512, 1, 1)
        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        # ========== 面部编码阶段 ==========
        # 逐层编码面部图像，保存每一层的特征用于跳跃连接
        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        # ========== 解码阶段（带跳跃连接的U-Net结构）==========
        # 以音频嵌入作为起始输入，逐步解码生成嘴型图像
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                # 跳跃连接: 将解码器输出与编码器对应层的特征在通道维度拼接
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                # 调试用: 打印张量形状以便排查维度不匹配问题
                print(x.size())
                print(feats[-1].size())
                raise e
            
            # 弹出已使用的编码器特征
            feats.pop()

        # 最终输出块: 生成3通道RGB图像
        x = self.output_block(x)

        # 如果输入是多帧模式，将batch维度恢复
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # 按batch大小拆分: [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # 在时间维度堆叠: (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs


class Wav2Lip_disc_qual(nn.Module):
    """Wav2Lip 质量判别器网络
    
    该判别器用于判断生成的面部图像是否真实（即质量是否好）。
    它只处理面部图像的下半部分（嘴部区域），因为Wav2Lip主要影响嘴部区域。
    
    判别器使用nonorm_Conv2d（不带BatchNorm的卷积）构建，
    输出一个0-1之间的概率值，表示输入图像是真实图像的概率。
    
    网络结构：
    - 8个编码块逐步降低空间分辨率，提取深层特征
    - 二值预测头(binary_pred): 输出真假判断概率
    """
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        # ========== 面部编码器（判别器专用，使用无BN的卷积）==========
        # 输入: 3通道RGB面部图像的下半部分
        self.face_encoder_blocks = nn.ModuleList([
            # 第1块: 3通道 -> 32通道, 空间分辨率 48x96
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)), # 48,96

            # 第2块: 32通道 -> 64通道, 水平下采样
            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2), # 48,48
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            # 第3块: 64通道 -> 128通道, 2x2下采样
            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 24,24
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            # 第4块: 128通道 -> 256通道
            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),   # 12,12
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            # 第5块: 256通道 -> 512通道
            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),       # 6,6
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            # 第6块: 512通道 -> 512通道
            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),
            
            # 第7块: 512通道 -> 512通道, 压缩到1x1空间维度
            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        # 二值预测头: 1x1卷积将512通道压缩为1通道，Sigmoid输出概率值
        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        # 标签噪声系数，用于标签平滑（当前未使用）
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        """获取面部图像的下半部分
        
        Wav2Lip主要影响嘴部区域，因此判别器只关注图像下半部分。
        输入: (B, C, H, W)
        输出: (B, C, H//2, W) - 仅保留下半部分
        """
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        """将多帧面部序列展平为2D张量
        
        将时间维度展开到batch维度，使得多帧图像可以批量处理。
        输入: (B, C, T, H, W)
        输出: (B*T, C, H, W)
        """
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        """感知前向传播 - 处理生成的（假的）面部图像
        
        用于计算生成图像的判别损失。将生成的面部图像通过编码器，
        然后希望判别器将其判为假（但这里使用ones标签计算损失，
        用于感知损失的训练）。
        
        输入: false_face_sequences - 生成器生成的面部图像序列
        返回: 判别损失值
        """
        # 将多帧序列展平为2D
        false_face_sequences = self.to_2d(false_face_sequences)
        # 提取下半部分（嘴部区域）
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        # 通过编码器提取特征
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        # 计算二值交叉熵损失（使用ones标签）
        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1), 
                                        torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        """前向传播 - 判断输入面部图像的真实度
        
        将真实面部图像通过编码器，输出判别概率。
        概率接近1表示判别器认为是真实图像，接近0表示虚假图像。
        
        输入: face_sequences - 面部图像序列
        返回: 判别概率值，形状为 (B, 1)
        """
        # 展平多帧序列
        face_sequences = self.to_2d(face_sequences)
        # 提取下半部分
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        # 通过编码器提取特征
        for f in self.face_encoder_blocks:
            x = f(x)

        # 输出判别概率
        return self.binary_pred(x).view(len(x), -1)
