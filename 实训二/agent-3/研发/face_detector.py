"""
面部检测和关键点提取模块
工单编号：人工智能NLP-Agent 数字人项目-文生图智能体任务

使用MediaPipe或DWPose进行面部检测和关键点提取
"""

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp


class FaceDetector:
    """面部检测器"""
    
    def __init__(self):
        """初始化面部检测器"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("✅ 面部检测器初始化完成")
    
    def detect_face(self, image):
        """
        检测面部并返回关键点
        
        Args:
            image: PIL Image对象
            
        Returns:
            dict: 包含面部关键点的字典
        """
        # 转换为OpenCV格式
        img_array = np.array(image)
        # PIL图片已经是RGB格式，MediaPipe期望RGB输入，无需转换
        img_rgb = img_array
        
        # 检测面部
        results = self.face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            raise ValueError("未检测到面部，请确保输入图片包含清晰的面部")
        
        # 获取面部关键点
        face_landmarks = results.multi_face_landmarks[0]
        h, w = img_array.shape[:2]
        
        # 提取关键点坐标
        landmarks = {}
        
        # 左眼关键点（索引33）
        landmarks["left_eye"] = (
            int(face_landmarks.landmark[33].x * w),
            int(face_landmarks.landmark[33].y * h)
        )
        
        # 右眼关键点（索引263）
        landmarks["right_eye"] = (
            int(face_landmarks.landmark[263].x * w),
            int(face_landmarks.landmark[263].y * h)
        )
        
        # 鼻子关键点（索引1）
        landmarks["nose"] = (
            int(face_landmarks.landmark[1].x * w),
            int(face_landmarks.landmark[1].y * h)
        )
        
        # 嘴巴关键点（索引13）
        landmarks["mouth"] = (
            int(face_landmarks.landmark[13].x * w),
            int(face_landmarks.landmark[13].y * h)
        )
        
        # 左眉（索引70）
        landmarks["left_eyebrow"] = (
            int(face_landmarks.landmark[70].x * w),
            int(face_landmarks.landmark[70].y * h)
        )
        
        # 右眉（索引300）
        landmarks["right_eyebrow"] = (
            int(face_landmarks.landmark[300].x * w),
            int(face_landmarks.landmark[300].y * h)
        )
        
        # 面部边界框
        face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
        
        face_points = []
        for idx in face_oval:
            face_points.append((
                int(face_landmarks.landmark[idx].x * w),
                int(face_landmarks.landmark[idx].y * h)
            ))
        
        landmarks["face_boundary"] = face_points
        
        # 计算面部中心
        landmarks["face_center"] = (
            int(np.mean([p[0] for p in face_points])),
            int(np.mean([p[1] for p in face_points]))
        )
        
        return landmarks
    
    def calculate_face_angle(self, landmarks):
        """
        计算面部旋转角度
        
        Args:
            landmarks: 面部关键点字典
            
        Returns:
            float: 面部角度（度）
        """
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]
        
        # 计算眼睛连线的角度
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    
    def draw_landmarks(self, image, landmarks):
        """
        在图片上绘制面部关键点（用于调试）
        
        Args:
            image: PIL Image对象
            landmarks: 面部关键点字典
            
        Returns:
            PIL Image: 绘制了关键点的图片
        """
        img_array = np.array(image.copy())
        
        # 绘制关键点
        for name, point in landmarks.items():
            if name == "face_boundary":
                # 绘制面部边界
                pts = np.array(landmarks["face_boundary"], np.int32)
                cv2.polylines(img_array, [pts], True, (0, 255, 0), 2)
            elif isinstance(point, tuple) and len(point) == 2:
                cv2.circle(img_array, point, 5, (0, 0, 255), -1)
                cv2.putText(img_array, name, (point[0] + 10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))


def get_face_landmarks(image):
    """
    便捷函数：获取面部关键点
    
    Args:
        image: PIL Image对象
        
    Returns:
        dict: 面部关键点
    """
    detector = FaceDetector()
    return detector.detect_face(image)
