# -*- coding: utf-8 -*-
# pirender_animate.py - PIRender人脸动画渲染模块
# 本文件实现了基于PIRender的人脸动画生成主类（AnimateFromCoeff_PIRender），
# 负责加载模型、生成动画帧、合成视频并处理音频同步。
# PIRender通过光流变形和图像编辑，根据3DMM姿态参数驱动人脸图像生成动画。

import os
import cv2
from tqdm import tqdm
import yaml
import numpy as np
import warnings
from skimage import img_as_ubyte
import safetensors
import safetensors.torch 
warnings.filterwarnings('ignore')


import imageio
import torch

from src.facerender.pirender.config import Config
from src.facerender.pirender.face_model import FaceGenerator

from pydub import AudioSegment 
from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
from src.utils.paste_pic import paste_pic
from src.utils.videoio import save_video_with_watermark

# 尝试导入webui模块，判断是否在WebUI环境中运行
try:
    import webui  # 在WebUI环境中运行
    in_webui = True
except:
    in_webui = False

class AnimateFromCoeff_PIRender():
    """基于PIRender的人脸动画生成类
    该类负责：
    1. 加载PIRender预训练模型
    2. 根据3DMM系数生成人脸动画帧
    3. 将动画帧合成为视频
    4. 音视频同步
    5. 可选的面部增强处理
    """

    def __init__(self, sadtalker_path, device):
        """初始化PIRender动画生成器
        
        参数:
            sadtalker_path: 包含模型路径的字典，包括：
                - pirender_yaml_path: PIRender配置文件路径
                - pirender_checkpoint: PIRender模型权重路径
            device: 计算设备（如 'cuda:0' 或 'cpu'）
        """
        # 加载PIRender配置
        opt = Config(sadtalker_path['pirender_yaml_path'], None, is_train=False)
        opt.device = device
        # 创建人脸生成器模型并移到指定设备
        self.net_G_ema = FaceGenerator(**opt.gen.param).to(opt.device)
        # 加载预训练权重
        checkpoint_path = sadtalker_path['pirender_checkpoint']
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.net_G_ema.load_state_dict(checkpoint['net_G_ema'], strict=False)
        print('load [net_G] and [net_G_ema] from {}'.format(checkpoint_path))
        # 设置为评估模式
        self.net_G = self.net_G_ema.eval()
        self.device = device
    

    def generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess='crop', img_size=256):
        """生成人脸动画视频
        
        参数:
            x: 输入数据字典，包含：
                - source_image: 源人脸图像（张量）
                - source_semantics: 源图像的语义/姿态参数
                - target_semantics_list: 目标姿态参数序列（驱动动画）
                - frame_num: 帧数
                - video_name: 视频名称
                - audio_path: 音频文件路径
            video_save_dir: 视频保存目录
            pic_path: 原始图片路径（用于full模式回贴）
            crop_info: 裁剪信息（包含原始尺寸等）
            enhancer: 面部增强方法（如 'gfpgan'）
            background_enhancer: 背景增强方法
            preprocess: 预处理模式（'crop' 或 'full'）
            img_size: 生成图像尺寸（默认256x256）
        返回:
            return_path: 最终视频文件路径
        """
        # 将输入数据转换为FloatTensor并移到计算设备
        source_image=x['source_image'].type(torch.FloatTensor)
        source_semantics=x['source_semantics'].type(torch.FloatTensor)
        target_semantics=x['target_semantics_list'].type(torch.FloatTensor) 
        source_image=source_image.to(self.device)
        source_semantics=source_semantics.to(self.device)
        target_semantics=target_semantics.to(self.device)
        frame_num = x['frame_num']
        
        # 在无梯度模式下生成动画帧
        with torch.no_grad():
            predictions_video = []
            # 逐帧生成：对每个目标姿态，用源图像生成对应的动画帧
            for i in tqdm(range(target_semantics.shape[1]), 'FaceRender:'):
                 predictions_video.append(self.net_G(source_image, target_semantics[:, i])['fake_image'])
        
        # 将所有帧堆叠并reshape为 (T, C, H, W)
        predictions_video = torch.stack(predictions_video, dim=1)
        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])

        # 将张量转换为numpy数组列表
        video = []
        for idx in range(len(predictions_video)):
            image = predictions_video[idx]
            # 转置为 (H, W, C) 格式
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)
        # 将float32转换为uint8
        result = img_as_ubyte(video)

        ### 生成的视频是256x256，需要保持原始宽高比进行缩放
        original_size = crop_info[0]
        if original_size:
            result = [ cv2.resize(result_i,(img_size, int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]
        
        video_name = x['video_name']  + '.mp4'
        # 保存临时视频文件（无音频）
        path = os.path.join(video_save_dir, 'temp_'+video_name)
        
        imageio.mimsave(path, result,  fps=float(25))

        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path 
        
        # ===== 音频处理：截取与视频帧数匹配的音频段 =====
        audio_path =  x['audio_path'] 
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name+'.wav')
        start_time = 0
        # cog will not keep the .mp3 filename
        sound = AudioSegment.from_file(audio_path)
        frames = frame_num 
        # 根据帧数计算结束时间（毫秒）
        end_time = start_time + frames*1/25*1000
        # 设置采样率为16000Hz并截取对应时间段的音频
        word1=sound.set_frame_rate(16000)
        word = word1[start_time:end_time]
        word.export(new_audio_path, format="wav")

        # 将视频和音频合并
        save_video_with_watermark(path, new_audio_path, av_path, watermark= False)
        print(f'The generated video is named {video_save_dir}/{video_name}') 

        # 如果使用full预处理模式，将生成的人脸回贴到原始完整图像中
        if 'full' in preprocess.lower():
            # 仅对完整图像添加水印
            video_name_full = x['video_name']  + '_full.mp4'
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            paste_pic(path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop= True if 'ext' in preprocess.lower() else False)
            print(f'The generated video is named {video_save_dir}/{video_name_full}') 
        else:
            full_video_path = av_path 

        #### 面部增强处理（可选）
        if enhancer:
            video_name_enhancer = x['video_name']  + '_enhanced.mp4'
            enhanced_path = os.path.join(video_save_dir, 'temp_'+video_name_enhancer)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer) 
            return_path = av_path_enhancer

            try:
                # 尝试使用生成器方式逐帧增强（节省内存）
                enhanced_images_gen_with_len = enhancer_generator_with_len(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            except:
                # 如果失败，使用列表方式加载所有帧进行增强
                enhanced_images_gen_with_len = enhancer_list(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            
            # 将增强后的视频与音频合并
            save_video_with_watermark(enhanced_path, new_audio_path, av_path_enhancer, watermark= False)
            print(f'The generated video is named {video_save_dir}/{video_name_enhancer}') 
            # 删除临时增强视频文件
            os.remove(enhanced_path)

        # 清理临时文件
        os.remove(path)
        os.remove(new_audio_path)

        return return_path

