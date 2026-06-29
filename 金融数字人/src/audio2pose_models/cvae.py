# cvae.py - 条件变分自编码器（Conditional Variational AutoEncoder）模块
# 该模块是audio2pose系统的核心，使用CVAE架构学习音频到姿态运动的映射
# 编码器在训练时将真实姿态运动编码为潜在变量，解码器根据音频和潜在变量生成姿态运动
# 训练时通过重参数化技巧实现端到端训练，测试时从标准正态分布采样z

import torch
import torch.nn.functional as F
from torch import nn
from src.audio2pose_models.res_unet import ResUnet

def class2onehot(idx, class_num):
    """
    将类别索引转换为one-hot编码向量。
    
    参数：
        idx: 类别索引，形状 [N, 1]，每个元素为0到class_num-1之间的整数
        class_num: 总类别数
    
    返回：
        onehot: one-hot编码，形状 [N, class_num]
    
    示例：
        idx=[0, 2, 1], class_num=3
        -> [[1,0,0], [0,0,1], [0,1,0]]
    """
    # 断言检查：确保索引值不超过类别数
    assert torch.max(idx).item() < class_num
    # 创建全零矩阵
    onehot = torch.zeros(idx.size(0), class_num).to(idx.device)
    # 使用scatter_将对应位置设为1
    onehot.scatter_(1, idx, 1)
    return onehot

class CVAE(nn.Module):
    """
    条件变分自编码器（Conditional VAE）。
    
    CVAE的核心思想：
    - 编码器：将真实数据和条件信息编码为潜在空间的分布参数（均值mu和方差logvar）
    - 重参数化：从该分布中采样潜在变量z，使得梯度可以反向传播
    - 解码器：根据潜在变量z和条件信息（音频嵌入）重构/生成数据
    
    条件信息包括：
    - 音频嵌入向量：来自音频编码器的特征
    - 身份类别：说话人的身份标签
    - 参考姿态：作为运动的基准
    
    参数说明：
        cfg: 配置文件，包含编码器/解码器层大小、潜在空间维度等
    """
    def __init__(self, cfg):
        super().__init__()
        # 从配置文件中读取各项参数
        encoder_layer_sizes = cfg.MODEL.CVAE.ENCODER_LAYER_SIZES    # 编码器MLP层大小
        decoder_layer_sizes = cfg.MODEL.CVAE.DECODER_LAYER_SIZES    # 解码器MLP层大小
        latent_size = cfg.MODEL.CVAE.LATENT_SIZE                    # 潜在空间维度
        num_classes = cfg.DATASET.NUM_CLASSES                       # 身份类别数
        audio_emb_in_size = cfg.MODEL.CVAE.AUDIO_EMB_IN_SIZE       # 音频嵌入输入维度
        audio_emb_out_size = cfg.MODEL.CVAE.AUDIO_EMB_OUT_SIZE     # 音频嵌入输出维度
        seq_len = cfg.MODEL.CVAE.SEQ_LEN                           # 处理的序列长度

        self.latent_size = latent_size  # 保存潜在空间维度

        # 初始化编码器和解码器
        self.encoder = ENCODER(encoder_layer_sizes, latent_size, num_classes,
                                audio_emb_in_size, audio_emb_out_size, seq_len)
        self.decoder = DECODER(decoder_layer_sizes, latent_size, num_classes,
                                audio_emb_in_size, audio_emb_out_size, seq_len)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧（Reparameterization Trick）。
        
        在VAE中，为了能够从潜在分布中采样并计算梯度，
        我们不直接采样z ~ N(mu, sigma^2)，而是：
        1. 从标准正态分布采样eps ~ N(0, I)
        2. 计算 z = mu + eps * exp(0.5 * logvar)
        
        这样z的分布为N(mu, exp(logvar))，且梯度可以通过eps反向传播。
        
        参数：
            mu: 潜在分布的均值，形状 [bs, latent_size]
            logvar: 潜在分布的对数方差，形状 [bs, latent_size]
        
        返回：
            z: 采样的潜在变量，形状 [bs, latent_size]
        """
        # 计算标准差：std = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        # 从标准正态分布采样噪声
        eps = torch.randn_like(std)
        # 重参数化：z = mu + eps * std
        return mu + eps * std

    def forward(self, batch):
        """
        训练时的前向传播。
        
        流程：
        1. 编码器将真实姿态运动编码为mu和logvar
        2. 通过重参数化采样潜在变量z
        3. 解码器根据z和音频嵌入生成姿态运动预测
        """
        # 编码：获取潜在分布参数
        batch = self.encoder(batch)
        mu = batch['mu']           # 均值
        logvar = batch['logvar']   # 对数方差
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        batch['z'] = z
        # 解码：生成预测
        return self.decoder(batch)

    def test(self, batch):
        """
        测试时的前向传播。
        
        与训练时不同，测试时直接使用batch中已有的z（从标准正态分布采样），
        不需要编码器。
        """
        '''
        class_id = batch['class']
        z = torch.randn([class_id.size(0), self.latent_size]).to(class_id.device)
        batch['z'] = z
        '''
        # 直接用解码器生成预测（z已在外部采样并放入batch中）
        return self.decoder(batch)

class ENCODER(nn.Module):
    """
    CVAE的编码器：将真实数据编码为潜在空间的分布参数。
    
    编码器的输入包括：
    - 真实姿态运动（通过ResUnet编码）
    - 音频嵌入向量（通过线性层映射）
    - 参考姿态
    - 身份类别（通过可学习的类别偏置）
    
    这些特征拼接后通过MLP网络，最终输出mu和logvar。
    
    参数说明：
        layer_sizes: MLP各层的大小列表
        latent_size: 潜在空间维度
        num_classes: 身份类别数
        audio_emb_in_size: 音频嵌入输入维度
        audio_emb_out_size: 音频嵌入输出维度
        seq_len: 序列长度
    """
    def __init__(self, layer_sizes, latent_size, num_classes, 
                audio_emb_in_size, audio_emb_out_size, seq_len):
        super().__init__()

        # ResUnet用于编码姿态运动序列
        self.resunet = ResUnet()
        self.num_classes = num_classes
        self.seq_len = seq_len

        # 多层感知机（MLP）网络
        self.MLP = nn.Sequential()
        # 输入层大小 = 原始输入 + 潜在维度 + 音频特征 + 参考姿态
        layer_sizes[0] += latent_size + seq_len*audio_emb_out_size + 6
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        # 输出层：分别输出均值和对数方差
        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)     # mu层
        self.linear_logvar = nn.Linear(layer_sizes[-1], latent_size)    # logvar层
        # 音频嵌入映射层
        self.linear_audio = nn.Linear(audio_emb_in_size, audio_emb_out_size)

        # 可学习的类别偏置参数，每个身份类别对应一个latent_size维的偏置向量
        self.classbias = nn.Parameter(torch.randn(self.num_classes, latent_size))

    def forward(self, batch):
        """
        编码器的前向传播。
        
        处理流程：
        1. 用ResUnet编码真实姿态运动为特征向量
        2. 将音频嵌入映射到目标维度
        3. 获取身份类别的偏置
        4. 拼接所有特征并通过MLP
        5. 输出mu和logvar
        """
        class_id = batch['class']                      # 身份类别索引
        pose_motion_gt = batch['pose_motion_gt']       #bs seq_len 6  真实姿态运动
        ref = batch['ref']                             #bs 6  参考姿态
        bs = pose_motion_gt.shape[0]                   # batch size
        audio_in = batch['audio_emb']                  # bs seq_len audio_emb_in_size  音频嵌入

        # ===== 姿态编码 =====
        # 增加通道维度后通过ResUnet编码：[bs, 1, seq_len, 6]
        pose_emb = self.resunet(pose_motion_gt.unsqueeze(1))          #bs 1 seq_len 6 
        # 展平为一维向量：[bs, seq_len*6]
        pose_emb = pose_emb.reshape(bs, -1)                    #bs seq_len*6

        # ===== 音频映射 =====
        print(audio_in.shape)
        # 将音频嵌入映射到目标维度：[bs, seq_len, audio_emb_out_size]
        audio_out = self.linear_audio(audio_in)                # bs seq_len audio_emb_out_size
        # 展平：[bs, seq_len*audio_emb_out_size]
        audio_out = audio_out.reshape(bs, -1)

        # 获取当前身份类别的偏置向量
        class_bias = self.classbias[class_id]                  #bs latent_size
        
        # 拼接所有特征：参考姿态 + 姿态嵌入 + 音频嵌入 + 类别偏置
        x_in = torch.cat([ref, pose_emb, audio_out, class_bias], dim=-1) #bs seq_len*(audio_emb_out_size+6)+latent_size
        
        # 通过MLP网络
        x_out = self.MLP(x_in)

        # 输出潜在分布的参数
        mu = self.linear_means(x_out)         # 均值 [bs, latent_size]
        logvar = self.linear_means(x_out)      # 对数方差 [bs, latent_size]
        # 注意：这里logvar使用了linear_means而非linear_logvar，可能是代码bug或特殊设计

        # 更新batch字典，返回mu和logvar
        batch.update({'mu':mu, 'logvar':logvar})
        return batch

class DECODER(nn.Module):
    """
    CVAE的解码器：根据潜在变量z和条件信息生成姿态运动预测。
    
    解码器的输入包括：
    - 潜在变量z（从编码器采样或从标准正态分布采样）
    - 音频嵌入向量
    - 参考姿态
    - 身份类别偏置
    
    输出：预测的姿态运动序列 [bs, seq_len, 6]
    
    参数说明：
        layer_sizes: MLP各层的大小列表
        latent_size: 潜在空间维度
        num_classes: 身份类别数
        audio_emb_in_size: 音频嵌入输入维度
        audio_emb_out_size: 音频嵌入输出维度
        seq_len: 序列长度
    """
    def __init__(self, layer_sizes, latent_size, num_classes, 
                audio_emb_in_size, audio_emb_out_size, seq_len):
        super().__init__()

        # ResUnet用于解码姿态运动
        self.resunet = ResUnet()
        self.num_classes = num_classes
        self.seq_len = seq_len

        # MLP网络：将拼接的特征映射为姿态运动的初步预测
        self.MLP = nn.Sequential()
        input_size = latent_size + seq_len*audio_emb_out_size + 6
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                # 中间层使用ReLU激活
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                # 最后一层使用Sigmoid激活，将输出限制在[0,1]范围
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
        
        # 姿态线性层：进一步精炼预测的姿态参数
        self.pose_linear = nn.Linear(6, 6)
        # 音频嵌入映射层
        self.linear_audio = nn.Linear(audio_emb_in_size, audio_emb_out_size)

        # 可学习的类别偏置参数
        self.classbias = nn.Parameter(torch.randn(self.num_classes, latent_size))

    def forward(self, batch):
        """
        解码器的前向传播。
        
        处理流程：
        1. 将音频嵌入映射到目标维度并展平
        2. 获取身份类别的偏置加到z上
        3. 拼接z、音频特征和参考姿态
        4. 通过MLP得到初步的姿态预测
        5. 通过ResUnet进行精细化处理
        6. 通过线性层输出最终预测
        """
        z = batch['z']                                          #bs latent_size  潜在变量
        bs = z.shape[0]                                         # batch size
        class_id = batch['class']                               # 身份类别
        ref = batch['ref']                                      #bs 6  参考姿态
        audio_in = batch['audio_emb']                           # bs seq_len audio_emb_in_size  音频嵌入
        #print('audio_in: ', audio_in[:, :, :10])

        # 音频嵌入映射：[bs, seq_len, audio_emb_in_size] -> [bs, seq_len, audio_emb_out_size]
        audio_out = self.linear_audio(audio_in)                 # bs seq_len audio_emb_out_size
        #print('audio_out: ', audio_out[:, :, :10])
        # 展平：[bs, seq_len*audio_emb_out_size]
        audio_out = audio_out.reshape([bs, -1])                 # bs seq_len*audio_emb_out_size
        
        # 获取身份类别偏置并加到z上（条件生成的关键步骤）
        class_bias = self.classbias[class_id]                   #bs latent_size
        z = z + class_bias
        
        # 拼接所有特征：参考姿态(6) + z(latent_size) + 音频特征(seq_len*audio_emb_out_size)
        x_in = torch.cat([ref, z, audio_out], dim=-1)
        
        # 通过MLP得到初步的姿态预测
        x_out = self.MLP(x_in)                                  # bs layer_sizes[-1]
        # reshape为序列格式：[bs, seq_len, 6]
        x_out = x_out.reshape((bs, self.seq_len, -1))

        #print('x_out: ', x_out)

        # 通过ResUnet进行精细化处理：[bs, 1, seq_len, 6]
        pose_emb = self.resunet(x_out.unsqueeze(1))             #bs 1 seq_len 6

        # 通过线性层输出最终的姿态运动预测：[bs, seq_len, 6]
        pose_motion_pred = self.pose_linear(pose_emb.squeeze(1))       #bs seq_len 6

        # 保存预测结果到batch字典
        batch.update({'pose_motion_pred':pose_motion_pred})
        return batch
