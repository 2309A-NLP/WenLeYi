"""
ArcFace/CosFace损失函数模块
本模块实现了人脸识别中常用的间隔损失函数，用于增加类间距离、减小类内距离。

支持的损失函数：
1. CosFace（余弦间隔损失）：在余弦相似度上添加固定间隔
2. ArcFace（角度间隔损失）：在角度空间上添加固定间隔，更符合角度间隔的几何意义

参考论文：
- CosFace: Large Margin Cosine Loss for Deep Face Recognition (CVPR 2018)
- ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR 2019)
"""
import torch
from torch import nn


def get_loss(name):
    """
    损失函数工厂函数，根据名称返回对应的损失函数实例
    
    参数:
        name (str): 损失函数名称
            - "cosface": CosFace余弦间隔损失
            - "arcface": ArcFace角度间隔损失
    返回:
        对应的损失函数模块
    """
    if name == "cosface":
        return CosFace()
    elif name == "arcface":
        return ArcFace()
    else:
        raise ValueError()


class CosFace(nn.Module):
    """
    CosFace损失函数（余弦间隔损失）
    
    在余弦相似度上为正样本添加一个固定的间隔m，
    使得同类样本的余弦相似度至少比最大异类余弦相似度大m。
    
    公式: L = s * (cos(theta_yi) - m)
    
    参数:
        s: 缩放因子（控制softmax的温度）
        m: 间隔大小
    """
    def __init__(self, s=64.0, m=0.40):
        """初始化CosFace，s=64.0为缩放因子，m=0.40为间隔大小"""
        super(CosFace, self).__init__()
        self.s = s  # 缩放因子
        m = m       # 间隔大小

    def forward(self, cosine, label):
        """
        前向传播：计算CosFace损失
        
        参数:
            cosine: 余弦相似度矩阵，形状为 (N, C)，N为batch大小，C为类别数
            label: 标签，形状为 (N,)，值为-1表示该样本不属于当前rank的类别
        返回:
            应用间隔后的相似度分数
        """
        index = torch.where(label != -1)[0]  # 找到正样本的索引（跳过负样本）
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)  # 创建间隔热力图
        m_hot.scatter_(1, label[index, None], self.m)  # 在正样本位置填入间隔值m
        cosine[index] -= m_hot  # 减去间隔
        ret = cosine * self.s  # 乘以缩放因子
        return ret


class ArcFace(nn.Module):
    """
    ArcFace损失函数（角度间隔损失）
    
    在角度空间（arccos）上为正样本添加一个固定的间隔m，
    比CosFace更具几何解释性，因为角度间隔在超球面上是均匀分布的。
    
    公式: L = s * cos(theta_yi + m)
    
    参数:
        s: 缩放因子（控制softmax的温度）
        m: 角度间隔（弧度制）
    """
    def __init__(self, s=64.0, m=0.5):
        """初始化ArcFace，s=64.0为缩放因子，m=0.5为角度间隔"""
        super(ArcFace, self).__init__()
        self.s = s  # 缩放因子
        self.m = m  # 角度间隔

    def forward(self, cosine: torch.Tensor, label):
        """
        前向传播：计算ArcFace损失
        
        参数:
            cosine: 余弦相似度矩阵，形状为 (N, C)
            label: 标签，形状为 (N,)
        返回:
            应用角度间隔后的相似度分数
        """
        index = torch.where(label != -1)[0]  # 找到正样本的索引
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)  # 创建间隔热力图
        cosine.acos_()  # 将余弦值转换为角度
        cosine[index] += m_hot  # 在角度空间添加间隔
        cosine.cos_().mul_(self.s)  # 转回余弦值并缩放
        return cosine
