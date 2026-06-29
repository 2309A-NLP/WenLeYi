"""面部重建损失函数定义脚本

该脚本定义了用于3D面部重建的各种损失函数，
包括感知损失、光照损失、关键点损失、正则化损失和反射率损失。
"""

import numpy as np
import torch
import torch.nn as nn
from kornia.geometry import warp_affine
import torch.nn.functional as F

def resize_n_crop(image, M, dsize=112):
    """调整图像大小并裁剪
    
    Args:
        image: 输入图像，形状 (b, c, h, w)
        M: 仿射变换矩阵，形状 (b, 2, 3)
        dsize: 目标尺寸，默认112
    
    Returns:
        裁剪后的图像
    """
    return warp_affine(image, M, dsize=(dsize, dsize), align_corners=True)

### 感知损失（Perceptual Loss）
class PerceptualLoss(nn.Module):
    """感知损失类
    
    使用人脸识别网络计算两张图像的感知相似度。
    通过计算特征向量的余弦距离来衡量图像的语义相似性。
    """
    
    def __init__(self, recog_net, input_size=112):
        """初始化感知损失
        
        Args:
            recog_net: 人脸识别网络
            input_size: 输入图像尺寸，默认112
        """
        super(PerceptualLoss, self).__init__()
        self.recog_net = recog_net
        # 预处理函数：将[0,1]范围转换为[-1,1]范围
        self.preprocess = lambda x: 2 * x - 1
        self.input_size=input_size
        
    def forward(imageA, imageB, M):
        """
        计算两张图像的感知损失（1 - 余弦距离）
        
        Args:
            imageA: 图像A，torch.tensor (B, 3, H, W)，范围(0, 1)，RGB顺序
            imageB: 图像B，与imageA相同格式
            M: 仿射变换矩阵
        
        Returns:
            感知损失值
        """
        # 裁剪并对齐面部区域
        imageA = self.preprocess(resize_n_crop(imageA, M, self.input_size))
        imageB = self.preprocess(resize_n_crop(imageB, M, self.input_size))

        # 冻结批归一化层
        self.recog_net.eval()
        
        # 提取并归一化特征向量
        id_featureA = F.normalize(self.recog_net(imageA), dim=-1, p=2)
        id_featureB = F.normalize(self.recog_net(imageB), dim=-1, p=2)  
        # 计算余弦相似度
        cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
        # 返回平均损失（1 - 余弦相似度）
        return torch.sum(1 - cosine_d) / cosine_d.shape[0]        

def perceptual_loss(id_featureA, id_featureB):
    """感知损失函数（简化版本）
    
    直接计算两个特征向量的余弦距离。
    
    Args:
        id_featureA: 特征向量A
        id_featureB: 特征向量B
    
    Returns:
        感知损失值
    """
    # 计算余弦相似度
    cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
    # 返回平均损失
    return torch.sum(1 - cosine_d) / cosine_d.shape[0]  

### 图像级损失（Image Level Loss）
def photo_loss(imageA, imageB, mask, eps=1e-6):
    """光照损失/图像重建损失
    
    使用L2范数计算两张图像之间的差异。
    
    Args:
        imageA: 图像A，torch.tensor (B, 3, H, W)，范围(0, 1)，RGB顺序
        imageB: 图像B，与imageA相同格式
        mask: 掩码，指示需要计算损失的区域
        eps: 防止数值不稳定的极小值
    
    Returns:
        光照损失值
    """
    # 计算L2范数（带sqrt以确保反向传播稳定性）
    loss = torch.sqrt(eps + torch.sum((imageA - imageB) ** 2, dim=1, keepdims=True)) * mask
    # 归一化损失
    loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    return loss

def landmark_loss(predict_lm, gt_lm, weight=None):
    """关键点损失
    
    使用加权MSE损失计算预测关键点与真实关键点之间的差异。
    
    Args:
        predict_lm: 预测的关键点，torch.tensor (B, 68, 2)
        gt_lm: 真实的关键点，torch.tensor (B, 68, 2)
        weight: 关键点权重，numpy.array (1, 68)
    
    Returns:
        关键点损失值
    """
    if not weight:
        # 创建默认权重，对某些关键点给予更高权重
        weight = np.ones([68])
        weight[28:31] = 20  # 鼻子关键点
        weight[-8:] = 20  # 嘴巴关键点
        weight = np.expand_dims(weight, 0)
        weight = torch.tensor(weight).to(predict_lm.device)
    # 计算加权MSE损失
    loss = torch.sum((predict_lm - gt_lm)**2, dim=-1) * weight
    # 归一化损失
    loss = torch.sum(loss) / (predict_lm.shape[0] * predict_lm.shape[1])
    return loss


### 正则化损失（Regularization Loss）
def reg_loss(coeffs_dict, opt=None):
    """正则化损失
    
    对面部系数进行正则化，确保生成的3D面部是合理的。
    
    Args:
        coeffs_dict: 面部系数字典，包含id、exp、tex、angle、gamma、trans
        opt: 配置选项，包含各系数的权重
    
    Returns:
        creg_loss: 系数正则化损失
        gamma_loss: 光照参数正则化损失
    """
    # 系数正则化，确保生成合理的3D面部
    if opt:
        w_id, w_exp, w_tex = opt.w_id, opt.w_exp, opt.w_tex
    else:
        w_id, w_exp, w_tex = 1, 1, 1, 1
    # 计算身份、表情和纹理系数的L2正则化损失
    creg_loss = w_id * torch.sum(coeffs_dict['id'] ** 2) +  \
           w_exp * torch.sum(coeffs_dict['exp'] ** 2) + \
           w_tex * torch.sum(coeffs_dict['tex'] ** 2)
    # 归一化损失
    creg_loss = creg_loss / coeffs_dict['id'].shape[0]

    # gamma正则化，确保近似单色光照
    gamma = coeffs_dict['gamma'].reshape([-1, 3, 9])
    # 计算三个通道的光照参数均值
    gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
    # 计算方差作为损失
    gamma_loss = torch.mean((gamma - gamma_mean) ** 2)

    return creg_loss, gamma_loss

def reflectance_loss(texture, mask):
    """反射率损失
    
    最小化纹理方差，确保皮肤反射率均匀。
    
    Args:
        texture: 面部纹理，torch.tensor (B, N, 3)
        mask: 皮肤区域掩码，torch.tensor (N)，值为0或1
    
    Returns:
        反射率损失值
    """
    # 重塑掩码形状
    mask = mask.reshape([1, mask.shape[0], 1])
    # 计算加权纹理均值
    texture_mean = torch.sum(mask * texture, dim=1, keepdims=True) / torch.sum(mask)
    # 计算方差损失
    loss = torch.sum(((texture - texture_mean) * mask)**2) / (texture.shape[0] * torch.sum(mask))
    return loss
