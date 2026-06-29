# audio2pose.py - 音频到头部姿态的映射模块
# 该模块负责将音频特征转换为头部姿态运动参数（6维的旋转向量和平移向量）
# 是SadTalker系统中"audio2pose"分支的核心模块
# 包含训练（forward）和测试（test）两种模式

import torch
from torch import nn
from src.audio2pose_models.cvae import CVAE
from src.audio2pose_models.discriminator import PoseSequenceDiscriminator
from src.audio2pose_models.audio_encoder import AudioEncoder

class Audio2Pose(nn.Module):
    """
    音频到姿态（Audio-to-Pose）的神经网络模块。
    
    该模块使用条件变分自编码器（CVAE）架构：
    - 音频编码器将梅尔频谱图编码为音频嵌入向量
    - CVAE的编码器在训练时将真实姿态运动编码为潜在变量
    - CVAE的解码器根据音频嵌入和潜在变量生成姿态运动预测
    
    姿态参数为6维：前3维为旋转向量（rotation），后3维为平移向量（translation）
    
    参数说明：
        cfg: 配置文件对象
        wav2lip_checkpoint: Wav2Lip预训练模型路径，用于初始化音频编码器
        device: 计算设备
    """
    def __init__(self, cfg, wav2lip_checkpoint, device='cuda'):
        super().__init__()
        self.cfg = cfg                      # 配置对象
        self.seq_len = cfg.MODEL.CVAE.SEQ_LEN       # CVAE处理的序列长度
        self.latent_dim = cfg.MODEL.CVAE.LATENT_SIZE # 潜在空间维度
        self.device = device                # 计算设备

        # 初始化音频编码器（基于Wav2Lip的预训练权重）
        self.audio_encoder = AudioEncoder(wav2lip_checkpoint, device)
        # 设置音频编码器为评估模式（不使用BatchNorm的训练统计量）
        self.audio_encoder.eval()
        # 冻结音频编码器的参数（不参与梯度更新）
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # 初始化CVAE生成器网络（包含编码器和解码器）
        self.netG = CVAE(cfg)
        # 初始化姿态序列判别器（用于对抗训练）
        self.netD_motion = PoseSequenceDiscriminator(cfg)
        
        
    def forward(self, x):
        """
        训练时的前向传播。
        
        输入x包含：
            - 'gt': 真实的3DMM系数，形状 [bs, frame_len+1, 73]
              其中前64维为表情系数，后6维（64:70）为姿态系数
            - 'class': 说话人/身份类别标签
            - 'indiv_mels': 逐帧梅尔频谱图
        
        处理流程：
        1. 从真实系数中提取姿态运动（相邻帧的姿态差值）
        2. 用音频编码器提取音频嵌入
        3. 通过CVAE编码-解码得到预测的姿态运动
        4. 将预测的运动累积到参考帧姿态上，得到绝对姿态
        """

        batch = {}
        # 获取真实的3DMM系数 [bs, frame_len+1, 73]
        coeff_gt = x['gt'].cuda().squeeze(0)           #bs frame_len+1 73
        
        # 计算姿态运动：当前帧姿态 - 第一帧（参考帧）姿态，得到相对运动
        # 取64:70维，即姿态参数（旋转+平移）
        batch['pose_motion_gt'] = coeff_gt[:, 1:, 64:70] - coeff_gt[:, :1, 64:70] #bs frame_len 6
        
        # 提取参考帧的姿态（第一帧的姿态，作为基准）
        batch['ref'] = coeff_gt[:, 0, 64:70]  #bs  6
        
        # 提取说话人/身份类别标签
        batch['class'] = x['class'].squeeze(0).cuda() # bs
        
        # 提取梅尔频谱图 [bs, seq_len+1, 80, 16]
        indiv_mels= x['indiv_mels'].cuda().squeeze(0) # bs seq_len+1 80 16

        # ===== 前向推理 =====
        audio_emb_list = []
        
        # 用音频编码器提取音频嵌入向量
        # 输入去掉第一帧（因为第一帧是参考帧），增加一个维度
        # 输出形状：[bs, seq_len, 512]
        audio_emb = self.audio_encoder(indiv_mels[:, 1:, :, :].unsqueeze(2)) #bs seq_len 512
        batch['audio_emb'] = audio_emb
        
        # 通过CVAE网络进行编码和解码，得到姿态运动预测
        batch = self.netG(batch)

        # 获取预测的姿态运动 [bs, frame_len, 6]
        pose_motion_pred = batch['pose_motion_pred']           # bs frame_len 6
        
        # 获取真实的绝对姿态 [bs, frame_len, 6]
        pose_gt = coeff_gt[:, 1:, 64:70].clone()               # bs frame_len 6
        
        # 将预测的相对运动累积到参考帧姿态上，得到预测的绝对姿态
        # 预测姿态 = 参考帧姿态 + 预测的相对运动
        pose_pred = coeff_gt[:, :1, 64:70] + pose_motion_pred  # bs frame_len 6

        # 将预测姿态和真实姿态保存到batch字典中
        batch['pose_pred'] = pose_pred
        batch['pose_gt'] = pose_gt

        return batch

    def test(self, x):
        """
        测试/推理时的前向传播。
        
        与训练时不同，测试时：
        - 不需要真实姿态的编码，直接采样随机潜在变量z
        - 按固定长度seq_len分段处理长序列
        - 对最后一段不足seq_len的部分进行填充处理
        
        输入x包含：
            - 'ref': 参考帧系数，形状 [bs, 1, 70]
            - 'class': 身份类别
            - 'indiv_mels': 逐帧梅尔频谱图
            - 'num_frames': 总帧数
        """

        batch = {}
        # 获取参考帧的完整系数 [bs, 1, 70]
        ref = x['ref']                            #bs 1 70
        
        # 提取参考帧的姿态部分（后6维）
        batch['ref'] = x['ref'][:,0,-6:]  
        
        # 提取身份类别标签
        batch['class'] = x['class'] 
        bs = ref.shape[0]    # batch size
        
        # 获取梅尔频谱图 [bs, T, 1, 80, 16]
        indiv_mels= x['indiv_mels']               # bs T 1 80 16
        
        # 去掉第一帧（参考帧），只保留需要预测的帧的音频
        indiv_mels_use = indiv_mels[:, 1:]        # we regard the ref as the first frame
        
        # 获取需要预测的帧数（总帧数减去参考帧）
        num_frames = x['num_frames']
        num_frames = int(num_frames) - 1

        # ===== 分段处理长序列 =====
        # 计算需要完整处理的段数和剩余帧数
        div = num_frames//self.seq_len    # 完整段数
        re = num_frames%self.seq_len      # 剩余帧数
        audio_emb_list = []
        
        # 初始化姿态运动预测列表，第一项为零（参考帧无运动）
        pose_motion_pred_list = [torch.zeros(batch['ref'].unsqueeze(1).shape, dtype=batch['ref'].dtype, 
                                                device=batch['ref'].device)]

        # 处理完整的seq_len段
        for i in range(div):
            # 从标准正态分布采样随机潜在变量z
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch['z'] = z
            
            # 提取当前段的音频嵌入
            audio_emb = self.audio_encoder(indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len,:,:,:]) #bs seq_len 512
            batch['audio_emb'] = audio_emb
            
            # 通过CVAE解码器生成姿态运动预测
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch['pose_motion_pred'])  #list of bs seq_len 6
        
        # 处理最后一段不足seq_len的帧
        if re != 0:
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch['z'] = z
            
            # 取最后seq_len帧的梅尔频谱图（不足则取最后可用的）
            audio_emb = self.audio_encoder(indiv_mels_use[:, -1*self.seq_len:,:,:,:]) #bs seq_len  512
            
            # 如果音频嵌入的时间长度不足seq_len，用第一帧的嵌入进行填充
            if audio_emb.shape[1] != self.seq_len:
                pad_dim = self.seq_len-audio_emb.shape[1]
                # 复制第一帧的嵌入作为填充
                pad_audio_emb = audio_emb[:, :1].repeat(1, pad_dim, 1) 
                audio_emb = torch.cat([pad_audio_emb, audio_emb], 1) 
            
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            # 只取最后re帧的有效预测结果
            pose_motion_pred_list.append(batch['pose_motion_pred'][:,-1*re:,:])   
        
        # 将所有段的预测结果沿时间维度拼接
        pose_motion_pred = torch.cat(pose_motion_pred_list, dim = 1)
        batch['pose_motion_pred'] = pose_motion_pred

        # 计算预测的绝对姿态：参考帧姿态 + 累积的相对运动
        pose_pred = ref[:, :1, -6:] + pose_motion_pred  # bs T 6

        batch['pose_pred'] = pose_pred
        return batch
