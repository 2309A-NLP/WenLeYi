"""
animate.py - SadTalker 面部动画生成主模块
该文件负责协调整个面部动画渲染流程，包括模型加载、动画生成和视频保存。
是 SadTalker 系统中 face rendering 的核心入口文件。
"""
import os
import cv2
import yaml
import numpy as np
import warnings
from skimage import img_as_ubyte
import safetensors
import safetensors.torch 
warnings.filterwarnings('ignore')


import torch

# 导入面部渲染相关的核心网络模块
from src.facerender.modules.keypoint_detector import HEEstimator, KPDetector  # 关键点检测器和头部姿态估计器
from src.facerender.modules.mapping import MappingNet  # 语义特征到头部姿态参数的映射网络
from src.facerender.modules.generator import  OcclusionAwareSPADEGenerator  # 基于遮挡感知和SPADE的图像生成器
from src.facerender.modules.make_animation import make_animation  # 动画合成函数

# from pydub import AudioSegment 
from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list  # 人脸增强工具（如GFPGAN）
from src.utils.paste_pic import paste_pic  # 将生成的人脸贴回原始图片的工具
from src.utils.videoio import save_video_with_watermark  # 带水印的视频保存工具
import imageio
try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False

def opencv_save_video(path, videos, fps = 25, img_size = 256):
    """
    使用 OpenCV 将帧序列保存为 MP4 视频文件。
    
    参数:
        path (str): 视频文件保存路径
        videos (list): 帧图像列表，每个元素为 numpy 数组
        fps (int): 帧率，默认 25
        img_size (int): 视频分辨率，默认 256x256
    """
    # 初始化视频编码器，使用 mp4v 编码格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(path, fourcc, float(fps), (img_size, img_size))

    # 逐帧写入视频文件
    for frame in videos:
        video_writer.write(frame)
        # cv2.imshow("Stream", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 释放视频写入器并关闭文件
    video_writer.release()

class AnimateFromCoeff():
    """
    基于系数的面部动画生成类。
    
    该类是 SadTalker 的核心推理类，负责：
    1. 初始化和加载所有子模型（生成器、关键点检测器、头部姿态估计器、映射网络）
    2. 根据驱动音频的语义系数生成面部动画视频
    3. 支持视频增强和全图贴回
    """

    def __init__(self, sadtalker_path, device):
        """
        初始化动画生成器，加载所有预训练模型。
        
        参数:
            sadtalker_path (dict): 包含各模型检查点路径的字典
            device (torch.device): 计算设备（CPU/GPU）
        """
        # 从 YAML 配置文件加载模型参数
        with open(sadtalker_path['facerender_yaml']) as f:
            config = yaml.safe_load(f)

        # 实例化各子网络模型，从配置文件读取参数
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                    **config['model_params']['common_params'])
        kp_extractor = KPDetector(**config['model_params']['kp_detector_params'],
                                    **config['model_params']['common_params'])
        he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
        mapping = MappingNet(**config['model_params']['mapping_params'])

        # 将所有模型移到指定设备（GPU/CPU）
        generator.to(device)
        kp_extractor.to(device)
        he_estimator.to(device)
        mapping.to(device)
        
        # 冻结所有模型参数，推理时不需要梯度更新
        for param in generator.parameters():
            param.requires_grad = False
        for param in kp_extractor.parameters():
            param.requires_grad = False 
        for param in he_estimator.parameters():
            param.requires_grad = False
        for param in mapping.parameters():
            param.requires_grad = False

        # 加载预训练模型权重
        if sadtalker_path is not None:
            if 'checkpoint' in sadtalker_path: # 使用 safe tensor 格式的检查点
                self.load_cpk_facevid2vid_safetensor(sadtalker_path['checkpoint'], kp_detector=kp_extractor, generator=generator, he_estimator=None)
            else:
                # 使用标准 PyTorch 检查点格式
                self.load_cpk_facevid2vid(sadtalker_path['free_view_checkpoint'], kp_detector=kp_extractor, generator=generator, he_estimator=he_estimator)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.")

        # 加载映射网络的预训练权重
        if  sadtalker_path['mappingnet_checkpoint'] is not None:
            self.load_cpk_mapping(sadtalker_path['mappingnet_checkpoint'], mapping=mapping)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.") 

        # 保存各子模型的引用到实例属性
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.he_estimator = he_estimator
        self.mapping = mapping

        # 将所有模型设置为评估模式（关闭 Dropout 和 BatchNorm 的训练行为）
        self.kp_extractor.eval()
        self.generator.eval()
        self.he_estimator.eval()
        self.mapping.eval()
         
        self.device = device
    
    def load_cpk_facevid2vid_safetensor(self, checkpoint_path, generator=None, 
                        kp_detector=None, he_estimator=None,  
                        device="cpu"):
        """
        从 SafeTensors 格式加载 face-vid2vid 的预训练模型权重。
        SafeTensors 是一种更安全的模型序列化格式，避免了 pickle 反序列化的安全风险。
        
        参数:
            checkpoint_path (str): SafeTensors 检查点文件路径
            generator: 图像生成器网络
            kp_detector: 关键点检测器网络
            he_estimator: 头部姿态估计器网络
            device (str): 加载设备，默认 CPU
        """
        # 加载 SafeTensors 文件
        checkpoint = safetensors.torch.load_file(checkpoint_path)

        # 根据模型名称前缀，分别加载各子模型的权重
        if generator is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'generator' in k:
                    # 去掉前缀，提取生成器自身的参数名
                    x_generator[k.replace('generator.', '')] = v
            generator.load_state_dict(x_generator)
        if kp_detector is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'kp_extractor' in k:
                    x_generator[k.replace('kp_extractor.', '')] = v
            kp_detector.load_state_dict(x_generator)
        if he_estimator is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'he_estimator' in k:
                    x_generator[k.replace('he_estimator.', '')] = v
            he_estimator.load_state_dict(x_generator)
        
        return None

    def load_cpk_facevid2vid(self, checkpoint_path, generator=None, discriminator=None, 
                        kp_detector=None, he_estimator=None, optimizer_generator=None, 
                        optimizer_discriminator=None, optimizer_kp_detector=None, 
                        optimizer_he_estimator=None, device="cpu"):
        """
        从标准 PyTorch 格式加载 face-vid2vid 的预训练模型权重。
        除了模型参数，还可以加载优化器状态（用于继续训练）。
        
        参数:
            checkpoint_path (str): PyTorch 检查点文件路径
            generator: 图像生成器网络
            discriminator: 判别器网络
            kp_detector: 关键点检测器网络
            he_estimator: 头部姿态估计器网络
            optimizer_*: 各模型对应的优化器
            device (str): 加载设备
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        # 加载生成器权重
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        # 加载关键点检测器权重
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        # 加载头部姿态估计器权重
        if he_estimator is not None:
            he_estimator.load_state_dict(checkpoint['he_estimator'])
        # 加载判别器权重（可能不存在）
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        # 加载生成器优化器状态
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_he_estimator is not None:
            optimizer_he_estimator.load_state_dict(checkpoint['optimizer_he_estimator'])

        return checkpoint['epoch']
    
    def load_cpk_mapping(self, checkpoint_path, mapping=None, discriminator=None,
                 optimizer_mapping=None, optimizer_discriminator=None, device='cpu'):
        """
        加载映射网络（MappingNet）的预训练权重。
        
        参数:
            checkpoint_path (str): 映射网络检查点文件路径
            mapping: 映射网络模型
            discriminator: 判别器模型
            optimizer_mapping: 映射网络优化器
            optimizer_discriminator: 判别器优化器
            device (str): 加载设备
        """
        checkpoint = torch.load(checkpoint_path,  map_location=torch.device(device))
        if mapping is not None:
            mapping.load_state_dict(checkpoint['mapping'])
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if optimizer_mapping is not None:
            optimizer_mapping.load_state_dict(checkpoint['optimizer_mapping'])
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])

        return checkpoint['epoch']

    def generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess='crop', img_size=256, fps = 25):
        """
        生成面部动画视频的主函数。
        
        流程：
        1. 将源图像和语义目标送入动画生成流水线
        2. 将生成的帧序列保存为视频
        3. 合并音频，生成最终视频
        4. 可选：将人脸贴回原始图片、使用增强器提升画质
        
        参数:
            x (dict): 包含源图像、语义特征、目标语义等的输入字典
            video_save_dir (str): 视频保存目录
            pic_path (str): 原始图片路径（用于全图贴回）
            crop_info (list): 裁剪信息 [原始尺寸, 裁剪框等]
            enhancer (str): 增强方法名，如 'gfpgan'
            background_enhancer: 背景增强器
            preprocess (str): 预处理方式，'crop' 或 'full'（全图模式）
            img_size (int): 输出图像尺寸
            fps (int): 视频帧率
        """
        # 提取并转换输入数据为浮点张量
        source_image=x['source_image'].type(torch.FloatTensor)
        source_semantics=x['source_semantics'].type(torch.FloatTensor)
        target_semantics=x['target_semantics_list'].type(torch.FloatTensor) 
        # 将数据移到指定设备
        source_image=source_image.to(self.device)
        source_semantics=source_semantics.to(self.device)
        target_semantics=target_semantics.to(self.device)
        frame_num = x['frame_num']

        # 调用 make_animation 函数执行核心动画生成
        # 参数：源图像、源语义、目标语义、各子网络、是否使用表情参数
        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor, self.he_estimator, self.mapping, 
                                        None, None, None, use_exp = True)

        # 调整输出形状：从 (batch, frames, C, H, W) 变为 (total_frames, C, H, W)
        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])
        # 截取到目标帧数
        predictions_video = predictions_video[:frame_num]

        # 将张量转换为 numpy 图像列表
        video = []
        for idx in range(predictions_video.shape[0]):
            image = predictions_video[idx]
            # 从 CHW 格式转为 HWC 格式，并转为 numpy
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)
        # 将浮点图像转换为 0-255 的 uint8 格式
        result = img_as_ubyte(video)

        ### 生成的视频为 256x256，需要保持原始宽高比进行缩放
        original_size = crop_info[0]
        if original_size:
            # 根据原始图像的宽高比调整输出尺寸
            result = [ cv2.resize(result_i,(img_size, int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]
            # result = [ cv2.cvtColor(cv2.resize(result_i, (img_size, int(img_size * original_size[1]/original_size[0]) )), cv2.COLOR_BGR2RGB) for result_i in result ]
            # result = [ cv2.cvtColor(result_i, cv2.COLOR_BGR2RGB) for result_i in result ]
            
        # 构建临时视频保存路径
        video_name = x['video_name']  + '.mp4'
        path = os.path.join(video_save_dir, 'temp_'+video_name)
        print("fps: ", fps, len(result))
        # imageio.mimsave(path, result,  fps=float(fps))
        
        # 使用 imageio 保存视频帧序列
        # save_video(path, result, fps, img_size)
        print(path)
        # mimsave
        imageio.mimsave(path, result,  fps=float(fps))
        
        # 构建最终视频路径（带音频）
        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path 
        
        audio_path =  x['audio_path'] 
        # audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        # new_audio_path = os.path.join(video_save_dir, 'new_answer.wav')
        # start_time = 0
        # # cog will not keep the .mp3 filename
        # sound = AudioSegment.from_file(audio_path)
        # frames = frame_num 
        # end_time = start_time + frames*1/fps*1000
        # word1=sound.set_frame_rate(16000)
        # word = word1[start_time:end_time]
        # word.export(new_audio_path, format="wav")

        # 将视频和音频合并，生成最终视频文件
        save_video_with_watermark(path, audio_path, av_path, watermark= False)

        # 如果预处理模式为 'full'，将生成的人脸贴回原始完整图片
        if 'full' in preprocess.lower():
            # 在全图模式下添加水印
            video_name_full = x['video_name']  + '_full.mp4'
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            # 将人脸视频贴回原始图片并合并音频
            paste_pic(path, pic_path, crop_info, audio_path, full_video_path, extended_crop= True if 'ext' in preprocess.lower() else False)
            print(f'The generated video is named {video_save_dir}/{video_name_full}') 
        else:
            full_video_path = av_path 

        ### 人脸增强处理（如使用 GFPGAN 提升画质）
        if enhancer:
            video_name_enhancer = x['video_name']  + '_enhanced.mp4'
            enhanced_path = os.path.join(video_save_dir, 'temp_'+video_name_enhancer)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer) 
            return_path = av_path_enhancer

            try:
                # 逐帧增强（生成器方式，内存效率高）
                enhanced_images_gen_with_len = enhancer_generator_with_len(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(fps))
                # print(enhanced_images_gen_with_len.shape)
                # save_video(enhanced_path, enhanced_images_gen_with_len, fps, img_size)
            except:
                # 如果生成器方式失败，使用列表方式（内存占用更高）
                enhanced_images_gen_with_len = enhancer_list(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                # print(enhanced_images_gen_with_len.shape)
                # save_video(enhanced_path, enhanced_images_gen_with_len, fps, img_size)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(fps))
            
            # 将增强后的视频与音频合并
            save_video_with_watermark(enhanced_path, audio_path, av_path_enhancer, watermark= False)
            # print(f'The generated video is named {video_save_dir}/{video_name_enhancer}')
            # 删除临时增强视频文件
            os.remove(enhanced_path)

        # 删除临时视频文件
        os.remove(path)

        return return_path
