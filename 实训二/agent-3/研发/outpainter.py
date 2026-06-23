"""
扩图模块 - Outpainting
工单编号：人工智能NLP-Agent 数字人项目-文生图智能体任务

使用Stable Diffusion Inpainting进行扩图
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionInpaintPipeline


class ImageOutpainter:
    """图像扩图器"""
    
    def __init__(self, device="cuda"):
        """
        初始化扩图器
        
        Args:
            device: 计算设备
        """
        self.device = device
        self.pipe = None
        print(f"🔧 扩图器使用设备: {device}")
    
    def load_model(self):
        """加载Inpainting模型"""
        print("📥 加载Inpainting模型...")
        
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.pipe.enable_model_cpu_offload()
        
        print("✅ Inpainting模型加载完成")
    
    def create_expansion_mask(self, image_size, expand_ratio=1.5, direction="all"):
        """
        创建扩图遮罩
        
        Args:
            image_size: 原始图片尺寸 (width, height)
            expand_ratio: 扩展比例
            direction: 扩展方向 ("all", "horizontal", "vertical")
            
        Returns:
            tuple: (expanded_image_size, mask, paste_position)
        """
        width, height = image_size
        
        # 计算新尺寸
        if direction == "all":
            new_width = int(width * expand_ratio)
            new_height = int(height * expand_ratio)
        elif direction == "horizontal":
            new_width = int(width * expand_ratio)
            new_height = height
        else:  # vertical
            new_width = width
            new_height = int(height * expand_ratio)
        
        # 创建遮罩（白色=保留原图，黑色=需要生成）
        mask = Image.new("L", (new_width, new_height), 0)
        draw = ImageDraw.Draw(mask)
        
        # 计算原图粘贴位置（居中）
        paste_x = (new_width - width) // 2
        paste_y = (new_height - height) // 2
        
        # 绘制保留区域（白色）
        draw.rectangle([paste_x, paste_y, paste_x + width, paste_y + height], fill=255)
        
        # 边缘羽化（渐变）
        feather_size = min(30, width // 10, height // 10)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_size))
        
        return (new_width, new_height), mask, (paste_x, paste_y)
    
    def expand_image(
        self,
        image,
        expand_ratio=1.5,
        prompt="natural background, high quality, consistent lighting, seamless",
        negative_prompt="blurry, low quality, distorted, artifacts, seams",
        num_inference_steps=50,
        guidance_scale=7.5
    ):
        """
        扩图
        
        Args:
            image: 原始图片
            expand_ratio: 扩展比例
            prompt: 提示词
            negative_prompt: 负面提示词
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            
        Returns:
            PIL Image: 扩展后的图片
        """
        if self.pipe is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        print("🔄 开始扩图...")
        
        original_size = image.size
        
        # 创建扩展遮罩
        new_size, mask, paste_pos = self.create_expansion_mask(
            original_size, expand_ratio
        )
        
        # 创建新画布并粘贴原图
        expanded_image = Image.new("RGB", new_size, (128, 128, 128))  # 灰色填充
        expanded_image.paste(image, paste_pos)
        
        # 调整尺寸到512x512用于生成
        expanded_512 = expanded_image.resize((512, 512), Image.Resampling.LANCZOS)
        mask_512 = mask.resize((512, 512), Image.Resampling.LANCZOS)
        
        # 使用Inpainting填充
        result = self.pipe(
            prompt=prompt,
            image=expanded_512,
            mask_image=mask_512,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt
        ).images[0]
        
        # 恢复到新尺寸
        result = result.resize(new_size, Image.Resampling.LANCZOS)
        
        print("✅ 扩图完成")
        return result
    
    def expand_with_context(
        self,
        image,
        expand_ratio=1.5,
        context_prompt="same person, same background style, consistent lighting"
    ):
        """
        带上下文的智能扩图
        
        Args:
            image: 原始图片
            expand_ratio: 扩展比例
            context_prompt: 上下文提示词
            
        Returns:
            PIL Image: 扩展后的图片
        """
        # 分析原图特征（简化版）
        prompt = f"{context_prompt}, natural, high quality, detailed"
        
        return self.expand_image(
            image=image,
            expand_ratio=expand_ratio,
            prompt=prompt,
            num_inference_steps=50
        )


def create_outpainter(device="cuda"):
    """
    便捷函数：创建扩图器
    
    Args:
        device: 计算设备
        
    Returns:
        ImageOutpainter: 扩图器实例
    """
    outpainter = ImageOutpainter(device)
    outpainter.load_model()
    return outpainter
