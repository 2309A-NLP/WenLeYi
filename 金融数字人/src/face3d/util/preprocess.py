"""Deep3DFaceRecon_pytorch的图像预处理脚本

该脚本包含用于3D面部重建的图像预处理代码，
包括面部对齐、图像裁剪、关键点提取等功能。
"""

import numpy as np
from scipy.io import loadmat
from PIL import Image
import cv2
import os
from skimage import transform as trans
import torch
import warnings
# 忽略numpy弃用警告
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
# 忽略未来警告
warnings.filterwarnings("ignore", category=FutureWarning) 


# 计算最小二乘问题用于图像对齐
def POS(xp, x):
    """透视正交投影（Perspective Orthographic Projection）
    
    通过最小二乘法计算2D-3D点对的变换参数（平移和缩放）。
    
    Args:
        xp: 2D点坐标，形状 (2, N)
        x: 3D点坐标，形状 (3, N)
    
    Returns:
        t: 平移向量，形状 (2,)
        s: 缩放因子
    """
    npts = xp.shape[1]

    # 构建最小二乘问题的系数矩阵A
    A = np.zeros([2*npts, 8])

    # 填充A矩阵的偶数行（x坐标）
    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    # 填充A矩阵的奇数行（y坐标）
    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    # 构建目标向量b
    b = np.reshape(xp.transpose(), [2*npts, 1])

    # 使用最小二乘法求解
    k, _, _, _ = np.linalg.lstsq(A, b)

    # 提取旋转分量和缩放/平移分量
    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]  # 缩放后的x平移
    sTy = k[7]  # 缩放后的y平移
    # 计算缩放因子（两个旋转向量的平均范数）
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    # 提取平移向量
    t = np.stack([sTx, sTy], axis=0)

    return t, s
    
# 调整图像大小并裁剪，用于面部重建
def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    """调整图像大小并裁剪到目标尺寸
    
    Args:
        img: 输入图像（PIL.Image）
        lm: 68个面部关键点坐标
        t: 平移向量
        s: 缩放因子
        target_size: 目标图像尺寸，默认224
        mask: 可选的掩码图像
    
    Returns:
        img: 裁剪后的图像
        lm: 调整后的关键点坐标
        mask: 裁剪后的掩码（如果有）
    """
    # 获取原始图像尺寸
    w0, h0 = img.size
    # 计算缩放后的尺寸
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    # 计算裁剪区域的边界
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    # 调整图像大小
    img = img.resize((w, h), resample=Image.BICUBIC)
    # 裁剪到目标尺寸
    img = img.crop((left, up, right, below))

    if mask is not None:
        # 同样处理掩码图像
        mask = mask.resize((w, h), resample=Image.BICUBIC)
        mask = mask.crop((left, up, right, below))

    # 调整关键点坐标到新的图像坐标系
    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img, lm, mask

# 面部重建的工具函数
def extract_5p(lm):
    """从68个关键点中提取5个关键点
    
    用于计算面部对齐变换。5个关键点包括：
    - 鼻尖
    - 左眼中心（左右眼各两个点的平均值）
    - 右眼中心（左右眼各两个点的平均值）
    - 左嘴角
    - 右嘴角
    
    Args:
        lm: 68个面部关键点坐标，形状 (68, 2)
    
    Returns:
        lm5p: 5个关键点坐标，形状 (5, 2)
    """
    # 关键点索引（从1开始，需要减1转换为0-based）
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    # 提取5个关键点
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    # 重新排序：鼻尖、左眼、右眼、左嘴角、右嘴角
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# 面部重建的工具函数
def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    """对齐图像
    
    使用5个关键点计算仿射变换，将图像对齐到标准位置。
    
    Args:
        img: 输入图像（PIL.Image），形状 (raw_H, raw_W, 3)
        lm: 68个面部关键点坐标，形状 (68, 2)
        lm3D: 标准3D面部关键点坐标，形状 (5, 3)
        mask: 可选的掩码图像（PIL.Image），形状 (raw_H, raw_W, 3)
        target_size: 目标图像尺寸，默认224
        rescale_factor: 重缩放因子，默认102
    
    Returns:
        transparams: 变换参数，numpy.array (raw_W, raw_H, scale, tx, ty)
        img_new: 对齐后的图像（PIL.Image），形状 (target_size, target_size, 3)
        lm_new: 调整后的关键点坐标，numpy.array (68, 2)
        mask_new: 对齐后的掩码（PIL.Image），形状 (target_size, target_size)
    """
    # 获取原始图像尺寸
    w0, h0 = img.size
    # 如果输入不是5个关键点，提取5个关键点
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # 使用5个关键点和3D标准关键点计算变换参数
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    # 调整缩放因子
    s = rescale_factor/s

    # 处理图像
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    # 保存变换参数
    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lm_new, mask_new
