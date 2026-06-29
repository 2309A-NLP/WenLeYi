# 把当前目录加入Python搜索路径，确保能找到src下的子模块
import sys
sys.path.append('./')

# 导入PyTorch（深度学习框架）、uuid（生成唯一ID）
import torch, uuid
# 导入文件操作、系统功能、目录复制、平台检测
import os, sys, shutil, platform
# 导入人脸裁剪和3DMM系数提取模块
from src.utils.preprocess import CropAndExtract
# 导入音频到表情系数的转换模块
from src.test_audio2coeff import Audio2Coeff
# 导入面部动画生成模块（核心渲染器）
from src.facerender.animate import AnimateFromCoeff
# 导入数据批处理模块（把音频和图片打包成模型输入）
from src.generate_batch import get_data
# 导入面部渲染数据准备模块
from src.generate_facerender_batch import get_facerender_data
# 导入路径初始化模块（配置所有模型和配置文件的路径）
from src.utils.init_path import init_path


# ==================== SadTalker核心类 ====================
# SadTalker是数字人视频生成的核心引擎
# 它的工作：给一张照片+一段音频，输出一个口型同步的说话视频
class SadTalker():

    # 初始化方法：加载模型和配置
    # checkpoint_path: 模型文件目录（存放预训练权重）
    # config_path: 配置文件目录（网络结构参数）
    # lazy_load: 是否延迟加载（暂未使用）
    def __init__(self, checkpoint_path='checkpoints', config_path='src/config', lazy_load=False):
        import platform
        # 自动检测计算设备：优先用NVIDIA GPU，其次Apple Silicon，最后CPU
        if torch.cuda.is_available():
            device = "cuda"  # 有NVIDIA显卡就用GPU加速
        elif platform.system() == 'Darwin':  # macOS
            device = "mps"   # Apple Silicon芯片
        else:
            device = "cpu"   # 纯CPU，速度最慢
        
        self.device = device  # 保存设备信息，后续所有计算都用这个设备

        # 设置PyTorch的模型缓存目录
        os.environ['TORCH_HOME']= checkpoint_path

        # 保存路径配置
        self.checkpoint_path = checkpoint_path  # 模型权重路径
        self.config_path = config_path          # 配置文件路径
        
        # 初始化所有模型路径（找到SadTalker需要的各个模型文件在哪里）
        # 参数：模型目录、配置目录、输出尺寸256、不使用反向传播、裁剪模式
        self.sadtalker_paths = init_path(checkpoint_path, self.config_path, 256, False, 'crop')
        
        # 创建面部动画渲染器（负责把表情系数驱动到人脸上生成视频）
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)
        
        # 创建音频到表情系数的转换器（负责从音频中提取表情和姿态信息）
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)

    # ==================== 核心方法：生成说话视频 ====================
    # 输入：一张照片 + 一段音频
    # 输出：一段口型同步的说话视频
    # 参数说明：
    #   pic_path - 用户照片的3DMM系数路径（已预处理）
    #   crop_pic_path - 裁剪后的人脸照片路径
    #   first_coeff_path - 第一帧的3DMM系数文件
    #   crop_info - 裁剪信息（位置、大小等）
    #   source_image - 原始用户照片路径
    #   driven_audio - 驱动音频路径（TTS合成的语音）
    #   preprocess - 预处理方式（'crop'裁剪模式）
    #   still_mode - 静态模式（True=头部不动，只动嘴巴）
    #   use_enhancer - 是否使用GFPGAN人脸增强（提高清晰度）
    #   batch_size - 批处理大小（一次处理几帧）
    #   size - 输出视频尺寸（256x256像素）
    #   pose_style - 头部姿态风格（0=自然）
    #   facerender - 渲染器类型（'facevid2vid'=面部视频到视频）
    #   exp_scale - 表情缩放系数（1.0=正常，越大表情越夸张）
    #   fps - 视频帧率（20帧/秒）
    #   result_dir - 输出目录
    def test(self, 
            pic_path,#
            crop_pic_path,
            first_coeff_path,
            crop_info,
            source_image, driven_audio, preprocess='crop', 
            still_mode=False,  use_enhancer=False, batch_size=1, size=256, 
            pose_style = 0, 
            facerender='facevid2vid',
            exp_scale=1.0, 
            use_ref_video = False,
            ref_video = None,
            ref_info = None,
            use_idle_mode = False,
            length_of_audio = 0, use_blink=True, fps=20,
            result_dir='./results/'):


        # 创建输出目录
        save_dir = result_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 参考视频的姿态系数和眨眼系数（当前版本不使用参考视频，设为None）
        ref_pose_coeff_path = None
        ref_eyeblink_coeff_path = None
        
        # 音频路径就是驱动音频（TTS合成的语音文件）
        audio_path = driven_audio
        
        # ========== 第一步：准备数据 ==========
        # get_data把照片系数和音频打包成模型需要的输入格式
        # 参数：第一帧系数、音频路径、设备、是否静态模式、是否空闲模式、帧率等
        batch = get_data(first_coeff_path, audio_path, self.device, ref_eyeblink_coeff_path=ref_eyeblink_coeff_path, still=still_mode, \
            idlemode=use_idle_mode, length_of_audio=length_of_audio, use_blink=use_blink, fps = fps)
        
        # ========== 第二步：音频→表情系数 ==========
        # audio_to_coeff根据音频特征，生成每帧的表情系数和头部姿态系数
        # coeff包含：表情系数（嘴巴张合、眉毛运动）+ 姿态系数（头部转动角度）
        coeff = self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        # ========== 第三步：准备渲染数据 ==========
        # get_facerender_data把表情系数、裁剪信息、音频等打包成渲染器需要的格式
        # 包括：逐帧的表情系数、面部位置、图片尺寸、表情缩放等
        data = get_facerender_data(coeff, crop_pic_path, first_coeff_path, audio_path, batch_size, still_mode=still_mode, \
            preprocess=preprocess, size=size, expression_scale = exp_scale, facemodel=facerender)
        
        # ========== 第四步：渲染生成视频 ==========
        # animate_from_coeff是核心渲染器
        # 它拿着表情系数，把静态照片逐帧"画"成会说话的视频
        # enhancer='gfpgan'表示可选的人脸增强（提高清晰度）
        return_path = self.animate_from_coeff.generate(data, save_dir,  pic_path, crop_info, enhancer='gfpgan' if use_enhancer else None, preprocess=preprocess, img_size=size, fps = fps)
        
        # ========== 第五步：清理显存 ==========
        # GPU显存是有限的，生成完视频后要释放，否则后续操作会显存不足
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # 清空GPU缓存
            torch.cuda.synchronize()   # 等待GPU所有操作完成
        
        # 垃圾回收：释放Python内存中的临时对象
        import gc; gc.collect()
        
        # 返回生成的视频文件路径
        return return_path
    

    # ==================== test2方法：完整流程版本 ====================
    # 与test方法的区别：test2包含完整的人脸检测和预处理流程
    # test方法假设输入已经预处理好了（由webui.py提前处理）
    # test2方法从原始照片开始，自己完成所有预处理步骤
    def test2(self, source_image, driven_audio, preprocess='crop', 
        still_mode=False,  use_enhancer=False, batch_size=1, size=256, 
        pose_style = 0, 
        facerender='facevid2vid',
        exp_scale=1.0, 
        use_ref_video = False,
        ref_video = None,
        ref_info = None,
        use_idle_mode = False,
        length_of_audio = 0, use_blink=True, fps = 20,
        result_dir='./results/'):
        
        # 创建输出目录
        os.makedirs(result_dir, exist_ok=True)
        
        # 重新初始化路径（支持不同的输出尺寸）
        self.sadtalker_paths = init_path(self.checkpoint_path, self.config_path, size, False, preprocess)
        print(self.sadtalker_paths)
        
        # 创建各个处理模块
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)  # 音频→表情系数
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)  # 人脸裁剪+3DMM提取
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)  # 面部动画渲染

        # 用UUID生成唯一的时间戳目录名（避免文件冲突）
        time_tag = str(uuid.uuid4())
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)

        # 创建输入文件的临时存放目录
        input_dir = os.path.join(save_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        # 复制原始照片到临时目录
        print(source_image)
        pic_path = os.path.join(input_dir, os.path.basename(source_image)) 
        shutil.copy(source_image, input_dir)

        # 处理音频输入
        if driven_audio is not None and os.path.isfile(driven_audio):
            # 有音频文件：复制到临时目录
            audio_path = os.path.join(input_dir, os.path.basename(driven_audio))  
            shutil.copy(driven_audio, input_dir)
        elif use_idle_mode:
            # 空闲模式：生成一段静音音频（用于测试）
            audio_path = os.path.join(input_dir, 'idlemode_'+str(length_of_audio)+'.wav')
            from pydub import AudioSegment
            one_sec_segment = AudioSegment.silent(duration=1000*length_of_audio)  # 生成指定时长的静音
            one_sec_segment.export(audio_path, format="wav")
        else:
            # 既没有音频也不是空闲模式：报错
            assert driven_audio is not None, "No audio is given"
            print(use_ref_video, ref_info)
            assert use_ref_video == True and ref_info == 'all'

        # 如果使用参考视频，从视频中提取音频
        if use_ref_video and ref_info == 'all':
            ref_video_videoname = os.path.basename(ref_video)
            audio_path = os.path.join(save_dir, ref_video_videoname+'.wav')
            print('new audiopath:',audio_path)
            # 用ffmpeg从视频中提取音频轨道
            cmd = r"ffmpeg -y -hide_banner -loglevel error -i %s %s"%(ref_video, audio_path)
            os.system(cmd)        

        os.makedirs(save_dir, exist_ok=True)
        
        # ========== 关键步骤：从照片中提取人脸3DMM系数 ==========
        # CropAndExtract做三件事：
        # 1. 检测照片中的人脸位置
        # 2. 裁剪出人脸区域
        # 3. 提取3DMM系数（身份系数+表情系数）
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(pic_path, first_frame_dir, preprocess, True, size)
        print(first_coeff_path, crop_info)
        
        # 如果照片中没有检测到人脸，报错
        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        # 如果使用参考视频，从视频中提取姿态系数
        if use_ref_video:
            print('using ref video for genreation')
            ref_video_videoname = os.path.splitext(os.path.split(ref_video)[-1])[0]
            ref_video_frame_dir = os.path.join(save_dir, ref_video_videoname)
            os.makedirs(ref_video_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            # 从参考视频逐帧提取3DMM系数
            ref_video_coeff_path, _, _ =  self.preprocess_model.generate(ref_video, ref_video_frame_dir, preprocess, source_image_flag=False)
        else:
            ref_video_coeff_path = None

        # 根据参考信息类型，决定用参考视频的哪些系数
        if use_ref_video:
            if ref_info == 'pose':
                # 只用参考视频的姿态（头部转动）
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = None
            elif ref_info == 'blink':
                # 只用参考视频的眨眼
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'pose+blink':
                # 用参考视频的姿态+眨眼
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'all':            
                # 全部用自动生成（不用参考视频）
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = None
            else:
                raise('error in refinfo')
        else:
            # 不使用参考视频，所有系数自动生成
            ref_pose_coeff_path = None
            ref_eyeblink_coeff_path = None

        # 音频→表情系数转换
        if use_ref_video and ref_info == 'all':
            # 使用参考视频的系数作为表情系数
            coeff_path = ref_video_coeff_path
        else:
            # 从音频中自动生成表情系数
            batch = get_data(first_coeff_path, audio_path, self.device, ref_eyeblink_coeff_path=ref_eyeblink_coeff_path, still=still_mode, \
                idlemode=use_idle_mode, length_of_audio=length_of_audio, use_blink=use_blink, fps = fps)
            coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        # 表情系数→视频渲染
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, still_mode=still_mode, \
            preprocess=preprocess, size=size, expression_scale = exp_scale, facemodel=facerender)
        return_path = self.animate_from_coeff.generate(data, save_dir,  pic_path, crop_info, enhancer='gfpgan' if use_enhancer else None, preprocess=preprocess, img_size=size, fps = fps)
        print(f'The generated video is saved in {return_path}')

        # 释放预处理模型占用的显存
        del self.preprocess_model

        # 清理GPU显存和Python内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        import gc; gc.collect()
        
        # 返回生成的视频文件路径
        return return_path
    

# 测试代码：直接运行此文件时执行
if __name__ == '__main__':
    # 创建SadTalker实例（会自动加载模型）
    sadtalker = SadTalker()
    # 指定输入照片和音频
    source_image = "inputs/girl.png"   # 输入照片
    source_audio = "answer.wav"        # 输入音频（TTS合成的语音）
    # 调用test2生成说话视频
    # use_idle_mode=True表示使用空闲模式（测试用）
    # length_of_audio=5表示生成5秒视频
    sadtalker.test2(source_image, source_audio, use_idle_mode=True, length_of_audio=5, result_dir='results/')
