# croper.py - 人脸裁剪预处理模块
# 本模块提供人脸检测、关键点提取和人脸对齐裁剪功能
# 用于将输入视频/图片中的人脸区域提取出来，作为SadTalker模型的输入

import os
import cv2
import time
import glob
import argparse
import scipy
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from itertools import cycle

from src.face3d.extract_kp_videos_safe import KeypointExtractor  # 关键点提取器（基于深度学习的人脸关键点检测）
from facexlib.alignment import landmark_98_to_68  # 将98点人脸关键点转换为68点格式

import numpy as np
from PIL import Image


class Preprocesser:
    """人脸预处理器类
    
    负责人脸检测、关键点提取和人脸对齐裁剪。
    使用深度学习模型进行人脸关键点检测，然后根据关键点进行人脸对齐。
    """

    def __init__(self, device='cuda'):
        """初始化预处理器
        
        Args:
            device: 计算设备，'cuda'或'cpu'
        """
        self.predictor = KeypointExtractor(device)  # 初始化关键点提取器

    def get_landmark(self, img_np):
        """使用深度学习模型检测人脸并提取68个关键点
        
        流程：人脸检测 -> 裁剪人脸区域 -> 提取关键点 -> 坐标映射回原图
        
        Args:
            img_np: 输入图片（numpy数组，RGB格式）
        
        Returns:
            68个人脸关键点坐标，shape=(68, 2)；如果未检测到人脸则返回None
        """
        with torch.no_grad():  # 禁用梯度计算以提高推理速度
            # 使用检测网络检测人脸，置信度阈值0.97
            dets = self.predictor.det_net.detect_faces(img_np, 0.97)

        if len(dets) == 0:
            return None  # 未检测到人脸
        det = dets[0]  # 取第一个检测到的人脸

        # 根据检测框裁剪人脸区域
        img = img_np[int(det[1]):int(det[3]), int(det[0]):int(det[2]), :]
        # 提取关键点并将98点格式转换为68点格式
        lm = landmark_98_to_68(self.predictor.detector.get_landmarks(img)) # [0]

        # 将关键点坐标从裁剪区域映射回原图坐标系
        lm[:,0] += int(det[0])  # x坐标偏移
        lm[:,1] += int(det[1])  # y坐标偏移

        return lm

    def align_face(self, img, lm, output_size=1024):
        """根据人脸关键点进行人脸对齐
        
        根据眼睛、嘴巴等关键点的位置，计算仿射变换矩阵，
        将人脸对齐到标准位置，便于后续模型处理。
        
        Args:
            img: PIL Image格式的输入图片
            lm: 68个人脸关键点坐标
            output_size: 输出图片尺寸
        
        Returns:
            (缩放后的尺寸, 裁剪区域, 对齐后的四边形坐标)
        """
        # 提取各个面部区域的关键点
        lm_chin = lm[0: 17]  # 下巴轮廓点（从左到右）
        lm_eyebrow_left = lm[17: 22]  # 左眉毛点（从左到右）
        lm_eyebrow_right = lm[22: 27]  # 右眉毛点（从左到右）
        lm_nose = lm[27: 31]  # 鼻梁点（从上到下）
        lm_nostrils = lm[31: 36]  # 鼻翼点（从上到下）
        lm_eye_left = lm[36: 42]  # 左眼轮廓点（逆时针）
        lm_eye_right = lm[42: 48]  # 右眼轮廓点（逆时针）
        lm_mouth_outer = lm[48: 60]  # 嘴巴外轮廓点（逆时针）
        lm_mouth_inner = lm[60: 68]  # 嘴巴内轮廓点（逆时针）

        # 计算辅助向量（用于确定人脸的方向和大小）
        eye_left = np.mean(lm_eye_left, axis=0)  # 左眼中心
        eye_right = np.mean(lm_eye_right, axis=0)  # 右眼中心
        eye_avg = (eye_left + eye_right) * 0.5  # 双眼中心
        eye_to_eye = eye_right - eye_left  # 双眼之间的向量
        mouth_left = lm_mouth_outer[0]  # 嘴巴左侧点
        mouth_right = lm_mouth_outer[6]  # 嘴巴右侧点
        mouth_avg = (mouth_left + mouth_right) * 0.5  # 嘴巴中心
        eye_to_mouth = mouth_avg - eye_avg  # 眼睛中心到嘴巴中心的向量

        # 选择旋转裁剪矩形（计算对齐后的四个顶点）
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]  # 双眼差异与嘴巴差异的组合向量
        x /= np.hypot(*x)   # 归一化：hypot函数计算直角三角形的斜边长，用斜边长对两条直边做归一化
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)    # 双眼差和眼嘴差，选较大的作为基准尺度
        y = np.flipud(x) * [-1, 1]  # 与x垂直的向量（旋转90度）
        c = eye_avg + eye_to_mouth * 0.1  # 裁剪中心点（眼睛中心向下偏移10%的眼嘴距离）
        # 定义四边形，以面部基准位置为中心上下左右平移得到四个顶点
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2    # 定义四边形的大小（边长），为基准尺度的2倍

        # 缩放：如果计算出的四边形太大了，就按比例缩小它
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)  # 等比缩放图片
            quad /= shrink  # 同步缩放四边形坐标
            qsize /= shrink
        else:
            rsize = (int(np.rint(float(img.size[0]))), int(np.rint(float(img.size[1]))))

        # 裁剪：计算裁剪区域（包含边框）
        border = max(int(np.rint(qsize * 0.1)), 3)  # 边框大小为四边形边长的10%，最小3像素
        # 计算裁剪框的四个边界
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        # 扩展裁剪框并限制在图片范围内
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            # img = img.crop(crop)
            quad -= crop[0:2]  # 调整四边形坐标相对于裁剪区域

        # 填充：计算需要填充的区域
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        # 以下为可选的填充和模糊处理代码（当前已注释）
        # if enable_padding and max(pad) > border - 4:
        #     pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        #     img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        #     h, w, _ = img.shape
        #     y, x, _ = np.ogrid[:h, :w, :1]
        #     mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
        #                       1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        #     blur = qsize * 0.02
        #     img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        #     img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        #     img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        #     quad += pad[:2]

        # 变换：将四边形坐标展平并计算最终裁剪范围
        quad = (quad + 0.5).flatten()
        lx = max(min(quad[0], quad[2]), 0)  # 左边界
        ly = max(min(quad[1], quad[7]), 0)  # 上边界
        rx = min(max(quad[4], quad[6]), img.size[0])  # 右边界
        ry = min(max(quad[3], quad[5]), img.size[0])  # 下边界

        # 保存对齐后的图片
        return rsize, crop, [lx, ly, rx, ry]
    
    def crop(self, img_np_list, still=False, xsize=512):
        """对视频帧列表进行人脸裁剪
        
        以第一帧为基准检测人脸，然后对所有帧应用相同的裁剪。
        
        Args:
            img_np_list: 视频帧列表（numpy数组，RGB格式）
            still: 是否为静态图片模式（True时不对每帧单独裁剪）
            xsize: 输出图片尺寸
        
        Returns:
            (裁剪后的帧列表, 裁剪信息, 对齐四边形坐标)
        """
        img_np = img_np_list[0]  # 取第一帧进行人脸检测
        lm = self.get_landmark(img_np)  # 提取关键点

        if lm is None:
            raise 'can not detect the landmark from source image'  # 未检测到人脸则报错
        # 对第一帧进行人脸对齐，计算裁剪参数
        rsize, crop, quad = self.align_face(img=Image.fromarray(img_np), lm=lm, output_size=xsize)
        clx, cly, crx, cry = crop  # 裁剪区域的四个边界
        lx, ly, rx, ry = quad  # 对齐四边形的四个边界
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        for _i in range(len(img_np_list)):
            _inp = img_np_list[_i]
            _inp = cv2.resize(_inp, (rsize[0], rsize[1]))  # 统一缩放到相同尺寸
            _inp = _inp[cly:cry, clx:crx]  # 应用裁剪
            if not still:
                _inp = _inp[ly:ry, lx:rx]  # 非静态模式下进一步裁剪对齐区域
            img_np_list[_i] = _inp  # 替换原始帧
        return img_np_list, crop, quad
