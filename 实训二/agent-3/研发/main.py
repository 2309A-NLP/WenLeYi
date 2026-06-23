"""
文生图智能体 - 主程序
工单编号：人工智能NLP-Agent 数字人项目-文生图智能体任务

功能：
1. 输入一张面部图片
2. 生成三张图片：面部右转、左转、端正
3. 对生成的图片进行扩图处理

技术栈：
- IP-Adapter：保持面部特征一致性
- ControlNet + OpenPose：控制面部旋转角度
- Stable Diffusion：高质量图像生成
- Real-ESRGAN：超分辨率增强
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# 导入自定义模块
from face_detector import FaceDetector
from pose_generator import PoseGenerator
from image_generator import FaceImageGenerator
from outpainter import ImageOutpainter
from super_resolution import SuperResolution
from utils import (
    get_device, create_output_dir, load_image, save_image,
    print_separator, enhance_image
)


class Text2FacePipeline:
    """文生图完整流程"""
    
    def __init__(self, output_dir="output"):
        """
        初始化流水线
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.device = get_device()
        
        print_separator("文生图智能体 - 初始化")
        print(f"🔧 计算设备: {self.device}")
        print(f"📁 输出目录: {output_dir}")
        
        # 创建输出目录
        create_output_dir(output_dir)
        
        # 初始化各模块
        self.face_detector = FaceDetector()
        self.pose_generator = PoseGenerator()
        
        # 延迟加载重量级模型（节省内存）
        self.image_generator = None
        self.outpainter = None
        self.super_resolution = None
    
    def load_models(self, skip_sr=False):
        """
        加载所有模型
        
        Args:
            skip_sr: 是否跳过超分辨率模型（节省内存）
        """
        print_separator("加载模型")
        
        # 加载图像生成器
        self.image_generator = FaceImageGenerator(self.device)
        self.image_generator.load_models()
        
        # 加载扩图器
        self.outpainter = ImageOutpainter(self.device)
        self.outpainter.load_model()
        
        # 加载超分辨率（可选）
        if not skip_sr:
            try:
                self.super_resolution = SuperResolution(self.device)
                self.super_resolution.load_model()
            except Exception as e:
                print(f"⚠️ 超分辨率模型加载失败: {e}")
                print("  将跳过超分辨率处理")
                self.super_resolution = None
    
    def process_single_image(self, image_path, expand=True):
        """
        处理单张图片
        
        Args:
            image_path: 输入图片路径
            expand: 是否进行扩图
            
        Returns:
            dict: 处理结果
        """
        print_separator(f"处理图片: {image_path}")
        
        # 1. 加载图片
        print("\n📂 加载图片...")
        original_image = load_image(image_path)
        print(f"  图片尺寸: {original_image.size}")
        
        # 2. 检测面部
        print("\n🔍 检测面部...")
        landmarks = self.face_detector.detect_face(original_image)
        print(f"  ✅ 检测到面部")
        
        # 3. 生成三视图
        print("\n🎨 生成三视图...")
        three_views = self.image_generator.generate_three_views(
            original_image, landmarks, self.pose_generator
        )
        
        # 4. 保存原始三视图
        results = {"original_views": {}}
        angle_names = {-30: "right", 0: "front", 30: "left"}
        
        for angle, view_image in three_views.items():
            name = angle_names[angle]
            save_path = os.path.join(self.output_dir, f"view_{name}.jpg")
            view_image.save(save_path, quality=95)
            results["original_views"][name] = save_path
            print(f"  💾 保存: {save_path}")
        
        # 5. 扩图处理
        if expand:
            print("\n🖼️ 扩图处理...")
            results["expanded_views"] = {}
            
            for angle, view_image in three_views.items():
                name = angle_names[angle]
                print(f"\n  扩图: {name}...")
                
                expanded = self.outpainter.expand_with_context(view_image)
                save_path = os.path.join(self.output_dir, f"expanded_{name}.jpg")
                expanded.save(save_path, quality=95)
                results["expanded_views"][name] = save_path
                print(f"  💾 保存: {save_path}")
        
        # 6. 超分辨率增强
        if self.super_resolution is not None:
            print("\n✨ 超分辨率增强...")
            results["enhanced_views"] = {}
            
            source_views = results.get("expanded_views", results["original_views"])
            
            for angle_name, source_path in source_views.items():
                print(f"\n  增强: {angle_name}...")
                
                source_img = load_image(source_path)
                enhanced = self.super_resolution.enhance_image(source_img, outscale=2)
                save_path = os.path.join(self.output_dir, f"enhanced_{angle_name}.jpg")
                enhanced.save(save_path, quality=95)
                results["enhanced_views"][angle_name] = save_path
                print(f"  💾 保存: {save_path}")
        
        print_separator("处理完成")
        return results
    
    def process_batch(self, input_dir, expand=True):
        """
        批量处理目录中的所有图片
        
        Args:
            input_dir: 输入目录
            expand: 是否扩图
        """
        print_separator("批量处理")
        
        # 获取所有图片
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            f for f in Path(input_dir).iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"❌ 未找到图片文件: {input_dir}")
            return
        
        print(f"📂 找到 {len(image_files)} 张图片")
        
        # 处理每张图片
        for i, image_path in enumerate(image_files, 1):
            print(f"\n{'=' * 50}")
            print(f"处理第 {i}/{len(image_files)} 张: {image_path.name}")
            print(f"{'=' * 50}")
            
            try:
                self.process_single_image(str(image_path), expand)
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                continue
        
        print_separator("全部完成")


def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║              文生图智能体 - 面部旋转+扩图                    ║
║                                                            ║
║   工单编号: 人工智能NLP-Agent 数字人项目-文生图智能体任务     ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # 配置
    INPUT_DIR = "/root/autodl-tmp/input"  # AutoDL数据盘
    OUTPUT_DIR = "/root/autodl-tmp/output"
    
    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"📁 已创建输入目录: {INPUT_DIR}")
        print("   请将面部图片放入此目录，然后重新运行程序")
        return
    
    # 检查是否有图片
    image_files = [
        f for f in Path(INPUT_DIR).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    
    if not image_files:
        print(f"❌ 输入目录为空: {INPUT_DIR}")
        print("   请将面部图片放入此目录，然后重新运行程序")
        return
    
    # 创建流水线
    pipeline = Text2FacePipeline(output_dir=OUTPUT_DIR)
    
    # 加载模型（首次加载较慢）
    print("\n⏳ 首次运行需要下载模型，请耐心等待...")
    pipeline.load_models(skip_sr=False)
    
    # 处理图片
    pipeline.process_batch(INPUT_DIR, expand=True)
    
    print(f"\n🎉 全部完成！输出目录: {OUTPUT_DIR}")
    print("   包含：三视图 + 扩图 + 超分辨率增强")


if __name__ == "__main__":
    main()
