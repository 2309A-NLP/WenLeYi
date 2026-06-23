"""
工具函数模块
工单编号：人工智能NLP-Agent 数字人项目-文生图智能体任务
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path


def get_device():
    """获取计算设备"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_output_dir(output_dir):
    """创建输出目录"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")


def load_image(image_path):
    """加载图片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    return Image.open(image_path).convert("RGB")


def save_image(image, save_path, quality=95):
    """保存图片"""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    image.save(save_path, quality=quality)
    print(f"💾 已保存: {save_path}")


def resize_image(image, target_size=(512, 512)):
    """调整图片大小"""
    return image.resize(target_size, Image.Resampling.LANCZOS)


def create_circular_mask(image_size, feather=30):
    """创建圆形渐变遮罩（用于扩图边缘融合）"""
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 创建椭圆形遮罩
    draw.ellipse([feather, feather, width - feather, height - feather], fill=255)
    
    # 高斯模糊使边缘平滑
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    
    return mask


def blend_images(background, foreground, mask):
    """混合图片（使用遮罩）"""
    # 确保尺寸一致
    if background.size != foreground.size:
        foreground = foreground.resize(background.size, Image.Resampling.LANCZOS)
    if mask.size != background.size:
        mask = mask.resize(background.size, Image.Resampling.LANCZOS)
    
    # 混合
    result = Image.composite(foreground, background, mask)
    return result


def calculate_face_angle(face_landmarks):
    """
    计算面部角度
    使用眼睛和鼻子的位置计算旋转角度
    """
    # 获取关键点（假设使用DWPose格式）
    left_eye = face_landmarks.get("left_eye", [0, 0])
    right_eye = face_landmarks.get("right_eye", [0, 0])
    nose = face_landmarks.get("nose", [0, 0])
    
    # 计算眼睛中心
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    
    # 计算角度
    angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    angle_degrees = np.degrees(angle)
    
    return angle_degrees


def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    """调整亮度和对比度"""
    from PIL import ImageEnhance
    
    # 调整亮度
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # 调整对比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    return image


def reduce_noise(image):
    """降噪处理"""
    # 使用中值滤波降噪
    return image.filter(ImageFilter.MedianFilter(size=3))


def enhance_image(image):
    """图像增强：提升清晰度、色彩、对比度"""
    from PIL import ImageEnhance
    
    # 提升锐度
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    
    # 提升色彩饱和度
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.1)
    
    # 轻微提升对比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.05)
    
    return image


def format_angle(degrees):
    """格式化角度显示"""
    return f"{degrees:.1f}°"


def print_separator(title=""):
    """打印分隔线"""
    print("\n" + "=" * 50)
    if title:
        print(f" {title}")
        print("=" * 50)
