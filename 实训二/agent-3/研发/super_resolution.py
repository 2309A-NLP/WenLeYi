"""
超分辨率模块 - 图像增强
工单编号：人工智能NLP-Agent 数字人项目-文生图智能体任务

使用Real-ESRGAN提升图像清晰度
"""

import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class SuperResolution:
    """超分辨率处理器"""
    
    def __init__(self, device="cuda"):
        """
        初始化
        
        Args:
            device: 计算设备
        """
        self.device = device
        self.upsampler = None
        print(f"🔧 超分辨率使用设备: {device}")
    
    def load_model(self, model_path=None, scale=4):
        """
        加载Real-ESRGAN模型
        
        Args:
            model_path: 模型路径，None则使用默认
            scale: 放大倍数
        """
        print("📥 加载Real-ESRGAN模型...")
        
        # 检查模型路径是否有效
        if model_path is None:
            print("⚠️ 模型路径未指定，跳过模型加载")
            self.upsampler = None
            return
        
        # 创建模型架构
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale
        )
        
        # 创建upsampler
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True if self.device == "cuda" else False,
            gpu_id=0 if self.device == "cuda" else None
        )
        
        print("✅ Real-ESRGAN模型加载完成")
    
    def enhance_image(self, image, outscale=4):
        """
        增强图像清晰度
        
        Args:
            image: 输入图片 (PIL Image)
            outscale: 输出放大倍数
            
        Returns:
            PIL Image: 增强后的图片
        """
        if self.upsampler is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        print("🔄 提升图像清晰度...")
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 处理（Real-ESRGAN期望BGR格式）
        img_bgr = img_array[:, :, ::-1].copy()
        
        # 超分辨率处理
        output, _ = self.upsampler.enhance(img_bgr, outscale=outscale)
        
        # 转换回RGB并创建PIL Image
        output_rgb = output[:, :, ::-1]
        result = Image.fromarray(output_rgb)
        
        print(f"✅ 清晰度提升完成 (放大{outscale}倍)")
        return result
    
    def batch_enhance(self, images, outscale=4):
        """
        批量增强图片
        
        Args:
            images: 图片列表
            outscale: 放大倍数
            
        Returns:
            list: 增强后的图片列表
        """
        results = []
        for i, img in enumerate(images):
            print(f"\n  🔄 处理第 {i + 1}/{len(images)} 张图片...")
            enhanced = self.enhance_image(img, outscale)
            results.append(enhanced)
        return results


def create_super_resolution(device="cuda"):
    """
    便捷函数：创建超分辨率处理器
    
    Args:
        device: 计算设备
        
    Returns:
        SuperResolution: 处理器实例
    """
    sr = SuperResolution(device)
    sr.load_model()
    return sr
