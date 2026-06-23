"""
姿态生成模块 - 生成旋转后的OpenPose控制图
工单编号：人工智能NLP-Agent 数字人项目-文生图智能体任务

根据面部关键点生成不同角度的OpenPose骨骼图
"""

import numpy as np
from PIL import Image, ImageDraw
import math


class PoseGenerator:
    """姿态生成器 - 生成OpenPose控制图"""
    
    def __init__(self, image_size=(512, 512)):
        """
        初始化
        
        Args:
            image_size: 输出图片尺寸
        """
        self.image_size = image_size
        print("✅ 姿态生成器初始化完成")
    
    def rotate_point(self, point, center, angle_degrees):
        """
        旋转单个点
        
        Args:
            point: (x, y) 要旋转的点
            center: (x, y) 旋转中心
            angle_degrees: 旋转角度（度）
            
        Returns:
            tuple: 旋转后的点
        """
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        x, y = point
        cx, cy = center
        
        # 旋转公式
        new_x = cos_a * (x - cx) - sin_a * (y - cy) + cx
        new_y = sin_a * (x - cx) + cos_a * (y - cy) + cy
        
        return (int(new_x), int(new_y))
    
    def rotate_landmarks(self, landmarks, angle_degrees):
        """
        旋转面部关键点
        
        Args:
            landmarks: 原始面部关键点字典
            angle_degrees: 旋转角度（度），正值左转，负值右转
            
        Returns:
            dict: 旋转后的关键点
        """
        center = landmarks["face_center"]
        rotated = {}
        
        for key, value in landmarks.items():
            if key == "face_boundary":
                # 旋转边界点列表
                rotated[key] = [self.rotate_point(p, center, angle_degrees) for p in value]
            elif isinstance(value, tuple) and len(value) == 2:
                rotated[key] = self.rotate_point(value, center, angle_degrees)
            else:
                rotated[key] = value
        
        return rotated
    
    def generate_openpose_image(self, landmarks, angle_degrees=0):
        """
        生成OpenPose格式的控制图
        
        Args:
            landmarks: 面部关键点字典
            angle_degrees: 旋转角度（度）
            
        Returns:
            PIL Image: OpenPose控制图
        """
        # 创建黑色背景
        pose_image = Image.new("RGB", self.image_size, (0, 0, 0))
        draw = ImageDraw.Draw(pose_image)
        
        # 如果需要旋转，先旋转关键点
        if angle_degrees != 0:
            landmarks = self.rotate_landmarks(landmarks, angle_degrees)
        
        # 定义骨骼连接（简化版面部骨骼）
        skeleton = [
            ("left_eye", "right_eye"),  # 眼睛连线
            ("left_eye", "nose"),        # 左眼到鼻子
            ("right_eye", "nose"),       # 右眼到鼻子
            ("nose", "mouth"),           # 鼻子到嘴巴
            ("left_eyebrow", "left_eye"),  # 左眉到左眼
            ("right_eyebrow", "right_eye"),  # 右眉到右眼
        ]
        
        # 绘制骨骼线（红色）
        for start_key, end_key in skeleton:
            if start_key in landmarks and end_key in landmarks:
                start = landmarks[start_key]
                end = landmarks[end_key]
                draw.line([start, end], fill=(255, 0, 0), width=3)
        
        # 绘制关键点（白色）
        for key, point in landmarks.items():
            if key not in ["face_boundary", "face_center"] and isinstance(point, tuple):
                draw.ellipse([point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4], 
                            fill=(255, 255, 255))
        
        # 绘制面部边界（绿色）
        if "face_boundary" in landmarks:
            boundary = landmarks["face_boundary"]
            for i in range(len(boundary) - 1):
                draw.line([boundary[i], boundary[i + 1]], fill=(0, 255, 0), width=2)
            # 闭合
            draw.line([boundary[-1], boundary[0]], fill=(0, 255, 0), width=2)
        
        return pose_image
    
    def generate_all_poses(self, landmarks, angles=None):
        """
        生成所有角度的控制图
        
        Args:
            landmarks: 面部关键点字典
            angles: 角度列表，默认 [-30, 0, 30]
            
        Returns:
            dict: {角度: OpenPose图片}
        """
        if angles is None:
            angles = [-30, 0, 30]  # 右转、端正、左转
        
        poses = {}
        for angle in angles:
            pose_img = self.generate_openpose_image(landmarks, angle)
            poses[angle] = pose_img
            print(f"  ✅ 生成角度 {angle}° 的控制图")
        
        return poses
    
    def create_depth_map(self, landmarks, angle_degrees=0):
        """
        生成深度图（辅助ControlNet）
        
        Args:
            landmarks: 面部关键点字典
            angle_degrees: 旋转角度
            
        Returns:
            PIL Image: 深度图
        """
        depth_image = Image.new("L", self.image_size, 0)
        draw = ImageDraw.Draw(depth_image)
        
        # 如果需要旋转
        if angle_degrees != 0:
            landmarks = self.rotate_landmarks(landmarks, angle_degrees)
        
        # 绘制面部区域的深度
        if "face_boundary" in landmarks:
            boundary = landmarks["face_boundary"]
            draw.polygon(boundary, fill=200)  # 面部区域较亮
            
            # 中心最亮
            center = landmarks["face_center"]
            draw.ellipse([center[0] - 30, center[1] - 30, center[0] + 30, center[1] + 30],
                        fill=255)
        
        return depth_image


def create_rotation_pose(landmarks, angle_degrees, image_size=(512, 512)):
    """
    便捷函数：生成指定角度的OpenPose控制图
    
    Args:
        landmarks: 面部关键点
        angle_degrees: 旋转角度
        image_size: 图片尺寸
        
    Returns:
        PIL Image: OpenPose控制图
    """
    generator = PoseGenerator(image_size)
    return generator.generate_openpose_image(landmarks, angle_degrees)
