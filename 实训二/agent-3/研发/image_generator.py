"""
图像生成模块 - IP-Adapter + ControlNet
工单编号：人工智能NLP-Agent 数字人项目-文生图智能体任务

使用IP-Adapter保持面部特征，ControlNet控制旋转角度
"""

import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image


class FaceImageGenerator:
    """面部图像生成器"""
    
    def __init__(self, device="cuda"):
        """
        初始化生成器
        
        Args:
            device: 计算设备
        """
        self.device = device
        self.pipe = None
        print(f"🔧 使用设备: {device}")
    
    def load_models(self):
        """加载模型"""
        print("📥 加载ControlNet模型...")
        
        # 加载ControlNet（OpenPose控制）
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16
        )
        
        print("📥 加载Stable Diffusion模型...")
        
        # 加载SD Img2Img管道
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # 使用优化的调度器
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # 启用内存优化
        self.pipe.enable_model_cpu_offload()
        
        print("✅ 模型加载完成")
    
    def generate_face_image(
        self,
        original_image,
        pose_image,
        prompt="face, portrait, high quality, detailed face, natural lighting",
        negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy, bad hands, missing fingers",
        strength=0.4,
        guidance_scale=7.5,
        num_inference_steps=50,
        seed=None
    ):
        """
        生成面部图像
        
        Args:
            original_image: 原始面部图片
            pose_image: OpenPose控制图
            prompt: 提示词
            negative_prompt: 负面提示词
            strength: 变化强度（0-1，越小越像原图）
            guidance_scale: 引导强度
            num_inference_steps: 推理步数
            seed: 随机种子
            
        Returns:
            PIL Image: 生成的图片
        """
        if self.pipe is None:
            raise RuntimeError("模型未加载，请先调用load_models()")
        
        # 设置随机种子
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # 调整图片尺寸为512x512
        original_image = original_image.resize((512, 512), Image.Resampling.LANCZOS)
        pose_image = pose_image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # 生成图片
        result = self.pipe(
            prompt=prompt,
            image=original_image,
            control_image=pose_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            generator=generator
        ).images[0]
        
        return result
    
    def generate_three_views(self, original_image, landmarks, pose_generator):
        """
        生成三个角度的面部图片
        
        Args:
            original_image: 原始面部图片
            landmarks: 面部关键点
            pose_generator: 姿态生成器
            
        Returns:
            dict: {角度: 生成的图片}
        """
        print("\n🎨 开始生成三视图...")
        
        # 生成三个角度的控制图
        poses = pose_generator.generate_all_poses(landmarks, angles=[-30, 0, 30])
        
        results = {}
        
        # 角度描述映射
        angle_names = {
            -30: "right",
            0: "front", 
            30: "left"
        }
        
        # 针对不同角度优化提示词
        prompts = {
            -30: "face turned to the right, portrait photo, high quality, detailed face, natural lighting, sharp focus",
            0: "face looking straight ahead, portrait photo, high quality, detailed face, natural lighting, sharp focus",
            30: "face turned to the left, portrait photo, high quality, detailed face, natural lighting, sharp focus"
        }
        
        for angle, pose_img in poses.items():
            print(f"\n  🔄 生成角度 {angle}° ({angle_names[angle]})...")
            
            result = self.generate_face_image(
                original_image=original_image,
                pose_image=pose_img,
                prompt=prompts[angle],
                strength=0.35,  # 较低的强度保持特征
                guidance_scale=7.5,
                num_inference_steps=50,
                seed=42 + angle  # 不同角度使用不同种子
            )
            
            results[angle] = result
            print(f"  ✅ 角度 {angle}° 生成完成")
        
        return results


def create_generator(device="cuda"):
    """
    便捷函数：创建图像生成器
    
    Args:
        device: 计算设备
        
    Returns:
        FaceImageGenerator: 生成器实例
    """
    generator = FaceImageGenerator(device)
    generator.load_models()
    return generator
