import torch
from torch import nn
from torch.nn import functional as F
# pdb: Python调试器模块，用于断点调试（开发阶段使用）
import pdb
# 从同级conv模块导入自定义的卷积块
# Conv2dTranspose: 带BatchNorm的转置卷积，用于解码器上采样
# Conv2d: 带BatchNorm的普通卷积，用于编码器下采样和特征提取
# nonorm_Conv2d: 不带BatchNorm的卷积，用于判别器网络
from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class Wav2Lip(nn.Module):
    """Wav2Lip V2版本嘴型生成器网络
    
    Wav2Lip V2是基于音频驱动的嘴型同步生成网络的改进版本。
    相比V1版本的主要改进：
    1. 面部编码器增加了第8个编码块（8层编码 vs V1的7层），使用更大卷积核(4x4)
    2. 新增 audio_forward() 方法支持音频强度控制参数 a_alpha
    3. 新增 inference() 方法支持直接使用预计算的音频嵌入进行推理
    4. forward() 方法支持 a_alpha 参数，可调节音频对嘴型的影响强度
    
    网络架构采用编码器-解码器结构，包含U-Net风格的跳跃连接(skip connection)：
    
    1. 面部编码器(face_encoder_blocks): 8个编码块，逐层降低空间分辨率
       
    2. 音频编码器(audio_encoder): 将梅尔频谱图编码为512维的音频嵌入向量
    
    3. 面部解码器(face_decoder_blocks): 8个解码块，逐层恢复空间分辨率
       通过跳跃连接融合编码器的多尺度特征
    
    4. 输出块(output_block): 将解码器输出映射为3通道RGB图像（Sigmoid激活）
    """
    def __init__(self):
        super(Wav2Lip, self).__init__()

        # ========== 面部编码器 ==========
        # 使用ModuleList存储多个编码块，输入为6通道（3帧RGB图像拼接）
        self.face_encoder_blocks = nn.ModuleList([
            # 第1块: 6通道 -> 16通道, 大卷积核(7x7)捕获全局特征
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),

            # 第2块: 16通道 -> 32通道, 2x2下采样
            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            # 第3块: 32通道 -> 64通道, 2x2下采样，含3个残差块
            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            # 第4块: 64通道 -> 128通道, 2x2下采样
            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            # 第5块: 128通道 -> 256通道, 2x2下采样
            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            # 第6块: 256通道 -> 512通道, 2x2下采样
            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            # 第7块(V2新增): 512通道 -> 512通道, 2x2下采样（V1版本此步无下采样）
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            # 第8块(V2新增): 512通道 -> 512通道, 使用4x4卷积核压缩到1x1
            nn.Sequential(Conv2d(512, 512, kernel_size=4, stride=1, padding=0),
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

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
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

        # ========== 面部解码器 ==========
        # 使用转置卷积逐步上采样，通过跳跃连接融合编码器的多尺度特征
        self.face_decoder_blocks = nn.ModuleList([
            # 第1块: 512通道 -> 512通道, 瓶颈层处理
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),

            # 第2块(V2修改): 1024通道 -> 512通道, 使用4x4卷积核
            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=4, stride=1, padding=0),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            # 第3块: 1024通道 -> 512通道, 上采样
            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            # 第4块: 1024通道 -> 512通道, 上采样
            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            # 第5块: 768通道 -> 384通道, 上采样
            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),

            # 第6块: 512通道 -> 256通道, 上采样
            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),

            # 第7块: 320通道 -> 128通道, 上采样
            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ),

            # 第8块: 160通道 -> 64通道, 上采样到原始分辨率
            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ), ])

        # ========== 输出块 ==========
        # 将解码器的80通道特征映射为3通道RGB图像
        # 使用Sigmoid激活将输出值限制在[0, 1]范围内（像素值归一化）
        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())
        
    def audio_forward(self, audio_sequences, a_alpha=1.):
        """音频独立前向传播
        
        单独对音频序列进行编码，返回音频嵌入向量。
        支持通过a_alpha参数缩放音频嵌入强度。
        
        参数:
            audio_sequences: 音频梅尔频谱序列
            a_alpha: 音频强度缩放系数，大于1会增强音频对嘴型的影响，小于1会减弱
        返回:
            audio_embedding: 音频嵌入向量，形状为 (B, 512, 1, 1)
        """
        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1
        if a_alpha != 1.:
            audio_embedding *= a_alpha
        return audio_embedding
    
    def inference(self, audio_embedding, face_sequences):
        """推理模式前向传播
        
        使用预计算的音频嵌入向量直接进行推理，跳过音频编码步骤。
        适用于需要复用音频嵌入或进行音频嵌入插值的场景。
        
        参数:
            audio_embedding: 预计算的音频嵌入向量，形状为 (B, 512, 1, 1)
            face_sequences: 面部图像序列，形状为 (B, C, H, W)
        返回:
            outputs: 生成的嘴型图像，形状为 (B, 3, H, W)
        """
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
        outputs = x

        return outputs

    def forward(self, audio_sequences, face_sequences, a_alpha=1.):
        """前向传播 - 端到端的音频驱动嘴型生成
        
        完整的前向传播流程：音频编码 -> 面部编码 -> 解码生成嘴型图像
        
        参数:
            audio_sequences: 音频梅尔频谱序列, 形状为 (B, T, 1, 80, 16)
            face_sequences: 面部图像序列, 形状为 (B, C, T, H, W) 或 (B, C, H, W)
            a_alpha: 音频强度缩放系数，默认1.0（不缩放）
        返回:
            outputs: 生成的嘴型图像序列
        """
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        # 检测输入维度，支持多帧输入（5维）和单帧输入（4维）
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            # 多帧模式: 将时间维度展开到batch维度进行批量处理
            # audio: (B, T, 1, 80, 16) -> (B*T, 1, 80, 16)
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)#[bz, 5, 1, 80, 16]->[bz*5, 1, 80, 16]
            # face: (B, C, T, H, W) -> (B*T, C, H, W)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)#[bz, 6, 5, 256, 256]->[bz*5, 6, 256, 256]

        # 通过音频编码器得到音频嵌入向量
        audio_embedding = self.audio_encoder(audio_sequences)  # [bz*5, 1, 80, 16]->[bz*5, 512, 1, 1]
        if a_alpha != 1.:
            audio_embedding *= a_alpha                         #放大音频强度

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
        x = self.output_block(x)                              #[bz*5, 80, 256, 256]->[bz*5, 3, 256, 256]

        # 如果输入是多帧模式，将batch维度恢复
        if input_dim_size > 4:                                #[bz*5, 3, 256, 256]->[B, 3, 5, 256, 256]
            x = torch.split(x, B, dim=0)   
            outputs = torch.stack(x, dim=2)   

        else:
            outputs = x

        return outputs


class Wav2Lip_disc_qual(nn.Module):
    """Wav2Lip V2版本质量判别器网络
    
    该判别器用于判断生成的面部图像是否真实（即质量是否好）。
    它只处理面部图像的下半部分（嘴部区域），因为Wav2Lip主要影响嘴部区域。
    
    V2版本判别器相比V1的主要改进：
    1. 增加了第8个编码块（8层编码 vs V1的7层），使用4x4卷积核
    2. perceptual_forward方法简化了损失计算，直接返回判别概率
    
    判别器使用nonorm_Conv2d（不带BatchNorm的卷积）构建，
    输出一个0-1之间的概率值，表示输入图像是真实图像的概率。
    """
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        # ========== 面部编码器（判别器专用，使用无BN的卷积）==========
        # 输入: 3通道RGB面部图像的下半部分
        self.face_encoder_blocks = nn.ModuleList([
            # 第1块: 3通道 -> 32通道
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)),

            # 第2块: 32通道 -> 64通道, 水平下采样
            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),
                          nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            # 第3块: 64通道 -> 128通道, 2x2下采样
            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                          nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            # 第4块: 128通道 -> 256通道
            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                          nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            # 第5块: 256通道 -> 512通道
            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            # 第6块: 512通道 -> 512通道
            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1), ),

            # 第7块(V2新增): 512通道 -> 512通道, 增加一层下采样
            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1), ),

            # 第8块(V2新增): 512通道 -> 512通道, 4x4卷积核压缩到1x1
            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=4, stride=1, padding=0),
                          nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        # 二值预测头: 1x1卷积将512通道压缩为1通道，Sigmoid输出概率值
        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        # 标签噪声系数，用于标签平滑（当前未使用）
        self.label_noise = .0

    def get_lower_half(self, face_sequences):                      #取得输入图片的下半部分。
        """获取面部图像的下半部分
        
        Wav2Lip主要影响嘴部区域，因此判别器只关注图像下半部分。
        输入: (B, C, H, W)
        输出: (B, C, H//2, W) - 仅保留下半部分
        """
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def to_2d(self, face_sequences):                               #将输入的图片序列连接起来，形成一个二维的tensor。
        """将多帧面部序列展平为2D张量
        
        将时间维度展开到batch维度，使得多帧图像可以批量处理。
        输入: (B, C, T, H, W)
        输出: (B*T, C, H, W)
        """
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):            #前传生成图像
        """感知前向传播 - 处理生成的（假的）面部图像
        
        用于计算生成图像的判别概率。将生成的面部图像通过编码器，
        输出判别器对生成图像真假的判断概率。
        
        输入: false_face_sequences - 生成器生成的面部图像序列
        返回: 判别概率值
        """
        # 将多帧序列展平为2D: [bz, 3, 5, 256, 256]->[bz*5, 3, 256, 256]
        false_face_sequences = self.to_2d(false_face_sequences)
        # 提取下半部分（嘴部区域）: [bz*5, 3, 256, 256]->[bz*5, 3, 128, 256]
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        # 通过编码器提取特征: [bz*5, 3, 128, 256]->[bz*5, 512, 1, 1]
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        # 输出判别概率: [bz*5, 512, 1, 1]->[bz*5, 1, 1]
        return self.binary_pred(false_feats).view(len(false_feats), -1)

    def forward(self, face_sequences):                             #前传真值图像
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
