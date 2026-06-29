"""
mapping.py - 语义特征映射网络模块
该模块实现了 MappingNet，负责将输入的语义特征（如 3DMM 系数）
映射为头部姿态参数（yaw, pitch, roll）、平移向量和表情参数。

在 SadTalker 系统中，驱动视频的音频/动作信号先被编码为语义特征，
然后通过 MappingNet 转换为面部动画所需的几何参数，
最终驱动源面部图像生成动画帧。
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingNet(nn.Module):
    """
    映射网络（Mapping Network）。
    
    将输入的语义特征向量（如 3DMM 系数）映射为：
    - 头部姿态参数：yaw（偏航角）、pitch（俯仰角）、roll（翻滚角）
    - 平移向量 t：头部在 3D 空间中的位置偏移
    - 表情参数 exp：面部表情的偏移量
    
    网络结构：
    1. 一维卷积层提取局部特征
    2. 多层空洞一维卷积（dilation=3）扩大感受野
    3. 残差连接保持梯度流通
    4. 自适应平均池化压缩序列维度
    5. 多个全连接头分别预测各参数
    """

    def __init__(self, coeff_nc, descriptor_nc, layer, num_kp, num_bins):
        """
        参数:
            coeff_nc (int): 输入系数通道数（如 3DMM 的 64 维系数）
            descriptor_nc (int): 特征描述符通道数（隐藏层维度）
            layer (int): 空洞卷积层的数量
            num_kp (int): 关键点数量（用于确定表情参数维度）
            num_bins (int): 角度分类的 bin 数量（如 66）
        """
        super( MappingNet, self).__init__()

        self.layer = layer
        # LeakyReLU 激活函数，斜率 0.1
        nonlinearity = nn.LeakyReLU(0.1)

        # 第一层：一维卷积，kernel_size=7 提取较大的局部模式
        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        # 多层空洞一维卷积：扩大感受野而不增加参数量
        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)   

        # 自适应平均池化：将任意长度的序列压缩为固定长度（1）
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

        # 输出头：分别预测各参数
        self.fc_roll = nn.Linear(descriptor_nc, num_bins)     # 偏航角分类
        self.fc_pitch = nn.Linear(descriptor_nc, num_bins)    # 俯仰角分类
        self.fc_yaw = nn.Linear(descriptor_nc, num_bins)      # 翻滚角分类
        self.fc_t = nn.Linear(descriptor_nc, 3)               # 平移向量（3D）
        self.fc_exp = nn.Linear(descriptor_nc, 3*num_kp)      # 表情参数

    def forward(self, input_3dmm):
        """
        前向传播：将 3DMM 系数映射为头部姿态和表情参数。
        
        参数:
            input_3dmm (Tensor): 输入的 3DMM 系数，形状 (bs, coeff_nc, seq_len)
        
        返回:
            dict: 包含以下键值对：
                - 'yaw': 偏航角预测 (bs, num_bins)
                - 'pitch': 俯仰角预测 (bs, num_bins)
                - 'roll': 翻滚角预测 (bs, num_bins)
                - 't': 平移向量 (bs, 3)
                - 'exp': 表情参数 (bs, 3*num_kp)
        """
        # 第一层卷积提取初始特征
        out = self.first(input_3dmm)
        # 通过多层空洞卷积，使用残差连接
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            # 残差连接：输出 = 空洞卷积(输入) + 输入的中间部分（裁剪以匹配尺寸）
            out = model(out) + out[:,:,3:-3]
        # 自适应平均池化压缩序列维度
        out = self.pooling(out)
        # 展平为一维向量
        out = out.view(out.shape[0], -1)
        #print('out:', out.shape)

        # 通过各全连接头分别预测参数
        yaw = self.fc_yaw(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_roll(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp} 