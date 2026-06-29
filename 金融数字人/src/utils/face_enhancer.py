# face_enhancer.py - 人脸增强模块
# 本模块使用GFPGAN/RestoreFormer/CodeFormer等模型对人脸进行超分辨率增强
# 提高人脸区域的清晰度和质量，用于视频生成后的后处理

import os
import torch 


from tqdm import tqdm

from src.utils.videoio import load_video_to_cv2  # 导入视频加载工具

import cv2


class GeneratorWithLen(object):
    """带长度信息的生成器包装类
    
    普通生成器没有__len__方法，无法直接传给需要调用len()的函数。
    这个包装类为生成器添加了长度信息。
    参考: https://stackoverflow.com/a/7460929
    """

    def __init__(self, gen, length):
        """
        Args:
            gen: 原始生成器对象
            length: 生成器的总长度（元素个数）
        """
        self.gen = gen
        self.length = length

    def __len__(self):
        """返回生成器的长度"""
        return self.length

    def __iter__(self):
        """返回迭代器"""
        return self.gen


def enhancer_list(images, method='gfpgan', bg_upsampler='realesrgan'):
    """将所有增强后的图片一次性加载到列表中
    
    注意：此方法会将所有增强结果存储在内存中，适用于图片数量较少的场景。
    
    Args:
        images: 输入图片列表或视频文件路径
        method: 增强方法，支持'gfpgan'、'RestoreFormer'、'codeformer'
        bg_upsampler: 背景上采样器，默认使用RealESRGAN
    
    Returns:
        增强后的图片列表
    """
    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    return list(gen)


def enhancer_generator_with_len(images, method='gfpgan', bg_upsampler='realesrgan'):
    """提供带长度信息的生成器
    
    适用于需要调用len()的函数场景。
    使用生成器方式可以节省内存，避免一次性加载所有增强结果。

    Args:
        images: 输入图片列表或视频文件路径
        method: 增强方法
        bg_upsampler: 背景上采样器
    
    Returns:
        带长度信息的生成器对象
    """
    if os.path.isfile(images): # 如果输入是视频文件，先转换为图片序列
        # TODO: 创建load_video_to_cv2的生成器版本
        images = load_video_to_cv2(images)

    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    gen_with_len = GeneratorWithLen(gen, len(images))  # 包装为带长度的生成器
    return gen_with_len


def enhancer_generator_no_len(images, method='gfpgan', bg_upsampler='realesrgan'):
    """核心人脸增强生成器函数
    
    使用生成器方式逐帧处理，避免所有增强图片同时存储在内存中，
    相比enhancer_list可以节省大量内存。
    
    Args:
        images: 输入图片列表或视频文件路径
        method: 增强方法，可选值：
            - 'gfpgan': GFPGANv1.4（默认）
            - 'RestoreFormer': RestoreFormer模型
            - 'codeformer': CodeFormer模型
        bg_upsampler: 背景上采样器，'realesrgan'或None
    
    Yields:
        增强后的图片（numpy数组，RGB格式）
    """
    try:
        from gfpgan import GFPGANer  # 导入GFPGAN人脸增强器
    except ImportError:
        print("GFPGAN library not found. Installing...")
        try:
            # 使用pip自动安装GFPGAN库
            import subprocess
            subprocess.check_call(["pip", "install", "gfpgan"])
            
            # 安装后重试导入
            from gfpgan import GFPGANer
            print("GFPGAN library installed successfully!")
        except Exception as e:
            print(f"Failed to install GFPGAN library. Error: {e}")
            # 处理安装失败的情况
        
    print('face enhancer....')  # 提示开始人脸增强
    # 如果输入是视频文件路径，先加载为图片序列
    if not isinstance(images, list) and os.path.isfile(images): # handle video to images
        images = load_video_to_cv2(images)

    # ------------------------ 设置GFPGAN恢复器 ------------------------
    # 根据选择的方法配置不同的模型参数
    if  method == 'gfpgan':
        arch = 'clean'  # GFPGAN的网络架构
        channel_multiplier = 2  # 通道倍增系数
        model_name = 'GFPGANv1.4'  # 模型名称
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'  # 模型下载地址
    elif method == 'RestoreFormer':
        arch = 'RestoreFormer'  # RestoreFormer架构
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    elif method == 'codeformer': # TODO: 待完善
        arch = 'CodeFormer'  # CodeFormer架构
        channel_multiplier = 2
        model_name = 'CodeFormer'
        url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    else:
        raise ValueError(f'Wrong model version {method}.')  # 不支持的方法


    # ------------------------ 设置背景上采样器 ------------------------
    # RealESRGAN用于增强背景区域，提升整体画质
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU模式下不使用RealESRGAN（速度太慢）
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            # GPU模式下初始化RealESRGAN背景上采样器
            from basicsr.archs.rrdbnet_arch import RRDBNet  # 导入RRDB网络架构
            from realesrgan import RealESRGANer  # 导入RealESRGAN增强器
            # 定义RRDBNet网络（3通道输入输出，64特征，23个残差块，2倍超分辨率）
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,  # 2倍超分辨率
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,  # 分块处理大小
                tile_pad=10,  # 分块边距
                pre_pad=0,  # 预填充
                half=True)  # 使用半精度推理（GPU模式）
    else:
        bg_upsampler = None  # 不使用背景上采样器

    # 确定模型权重文件路径（按优先级查找）
    model_path = os.path.join('gfpgan/weights', model_name + '.pth')  # 首先在gfpgan/weights目录查找
    
    if not os.path.isfile(model_path):
        model_path = os.path.join('checkpoints', model_name + '.pth')  # 其次在checkpoints目录查找
    
    if not os.path.isfile(model_path):
        # 如果本地找不到，使用在线下载链接
        model_path = url

    # 初始化GFPGAN人脸增强器
    restorer = GFPGANer(
        model_path=model_path,  # 模型权重路径
        upscale=2,  # 2倍超分辨率
        arch=arch,  # 网络架构
        channel_multiplier=channel_multiplier,  # 通道倍增系数
        bg_upsampler=bg_upsampler)  # 背景上采样器

    # ------------------------ 逐帧增强人脸 ------------------------
    for idx in tqdm(range(len(images)), 'Face Enhancer:'):  # 显示进度条
        
        img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)  # RGB转BGR（OpenCV格式）
        
        # 执行人脸增强：裁剪人脸 -> 增强 -> 粘贴回原图
        cropped_faces, restored_faces, r_img = restorer.enhance(
            img,
            has_aligned=False,  # 输入人脸未对齐
            only_center_face=False,  # 处理所有人脸
            paste_back=True)  # 将增强后的人脸粘贴回原图
        
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)  # BGR转回RGB
        yield r_img  # 生成器输出增强后的图片
