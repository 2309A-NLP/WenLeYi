"""
数据批处理生成模块
负责将音频和图像数据预处理为模型输入所需的批次格式。
包括音频加载、Mel频谱图计算、帧对齐、眨眼序列生成等功能。
是SadTalker数字人系统中音频驱动部分的数据准备模块。
"""

import os

# from tqdm import tqdm
import torch
import numpy as np
import random
import scipy.io as scio
import src.utils.audio as audio


def crop_pad_audio(wav, audio_length):
    """
    裁剪或填充音频至指定长度。
    
    如果音频过长则截断，过短则用零填充（zero-padding），
    确保输出音频长度精确等于audio_length。
    
    参数：
        wav (np.ndarray): 输入音频波形数据
        audio_length (int): 目标音频长度（采样点数）
    
    返回：
        np.ndarray: 调整后的音频波形数据
    """
    if len(wav) > audio_length:
        # 音频过长，截断到指定长度
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        # 音频过短，用零值填充到指定长度
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav

def parse_audio_length(audio_length, sr, fps):
    """
    根据音频长度、采样率和帧率计算帧数。
    
    将音频的采样点数转换为视频帧数，并确保音频长度与帧数对齐。
    
    参数：
        audio_length (int): 音频长度（采样点数）
        sr (int): 采样率（如16000Hz）
        fps (int): 视频帧率（如25fps）
    
    返回：
        tuple: (对齐后的音频长度, 视频帧数)
    """
    # 计算每个视频帧对应的音频采样点数
    bit_per_frames = sr / fps

    # 计算总帧数（向下取整）
    num_frames = int(audio_length / bit_per_frames)
    # 重新计算与帧数对齐的音频长度
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames

def generate_blink_seq(num_frames):
    """
    生成固定模式的眨眼序列。
    
    以固定间隔（每80帧一次）生成眨眼动画的权重序列。
    眨眼过程模拟真实眨眼的渐变效果：
    从0→0.5→0.6→0.7→0.9→1.0→0.9→0.7→0.6→0.5→0
    
    参数：
        num_frames (int): 总帧数
    
    返回：
        np.ndarray: 眨眼权重序列，形状为 [num_frames, 1]
    """
    # 初始化全零的眨眼权重数组
    ratio = np.zeros((num_frames,1))
    frame_id = 0
    while frame_id in range(num_frames):
        start = 80  # 每80帧触发一次眨眼
        if frame_id+start+9<=num_frames - 1:
            # 在指定位置设置眨眼渐变权重
            # 模拟真实眨眼：快速闭眼→缓慢睁眼
            ratio[frame_id+start:frame_id+start+9, 0] = [0.5,0.6,0.7,0.9,1, 0.9, 0.7,0.6,0.5]
            # 移动到下一次眨眼的位置
            frame_id = frame_id+start+9
        else:
            break
    return ratio 

def generate_blink_seq_randomly(num_frames):
    """
    生成随机间隔的眨眼序列。
    
    与固定模式不同，该函数以随机间隔生成眨眼动画，
    使眨眼行为看起来更加自然和真实。
    
    参数：
        num_frames (int): 总帧数
    
    返回：
        np.ndarray: 眨眼权重序列，形状为 [num_frames, 1]
    """
    # 初始化全零的眨眼权重数组
    ratio = np.zeros((num_frames,1))
    # 帧数过少（≤20帧）时不需要眨眼
    if num_frames<=20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        # 随机选择两次眨眼之间的间隔帧数
        # 范围为[min(10, num_frames)]到[min(num_frames/2, 70)]
        start = random.choice(range(min(10,num_frames), min(int(num_frames/2), 70))) 
        if frame_id+start+5<=num_frames - 1:
            # 设置眨眼渐变权重（5帧的眨眼动画）
            ratio[frame_id+start:frame_id+start+5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
            # 移动到下一次眨眼的位置
            frame_id = frame_id+start+5
        else:
            break
    return ratio

def get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=False, idlemode=False, length_of_audio=False, use_blink=True, fps=25):
    """
    获取模型输入数据。
    
    该函数是数据准备的核心函数，负责：
        1. 加载并处理音频文件（计算Mel频谱图）
        2. 加载参考3DMM系数
        3. 生成眨眼序列
        4. 将所有数据整理为模型输入格式
    
    参数：
        first_coeff_path (str): 参考图像的3DMM系数文件路径（.mat格式）
        audio_path (str): 输入音频文件路径
        device: 计算设备（CPU/GPU）
        ref_eyeblink_coeff_path (str): 参考眨眼系数文件路径（可选）
        still (bool): 是否为静态模式（无头部运动），默认False
        idlemode (bool): 是否为空闲模式，默认False
        length_of_audio (bool/int): 音频长度（仅idle模式使用），默认False
        use_blink (bool): 是否启用眨眼效果，默认True
        fps (int): 视频帧率，默认25fps
    
    返回：
        dict: 包含以下键的字典：
            - 'indiv_mels': 每帧的Mel频谱特征 [1, T, 1, 80, 16]
            - 'ref': 参考3DMM系数 [1, 1, 70]
            - 'num_frames': 视频总帧数
            - 'ratio_gt': 眨眼权重序列 [1, T]
            - 'audio_name': 音频文件名（不含扩展名）
            - 'pic_name': 图片文件名（不含扩展名）
    """
    # SyncNet使用的Mel频谱时间步大小（16帧对应的时间窗口）
    syncnet_mel_step_size = 16

    # 从文件路径中提取图片名称（不含扩展名）
    pic_name = os.path.splitext(os.path.split(first_coeff_path)[-1])[0]
    # 从文件路径中提取音频名称（不含扩展名）
    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]

    
    # if idlemode:
    #     num_frames = int(length_of_audio * 25)
    #     indiv_mels = np.zeros((num_frames, 80, 16))
    # else:
    
    # ===== 步骤1：加载并处理音频 =====
    # 以16000Hz采样率加载WAV音频文件
    wav = audio.load_wav(audio_path, 16000) 
    # 计算与帧数对齐的音频长度和总帧数
    wav_length, num_frames = parse_audio_length(len(wav), 16000, fps)
    # 裁剪或填充音频到指定长度
    wav = crop_pad_audio(wav, wav_length)
    # 计算Mel频谱图并转置，形状变为 [时间帧数, 80]
    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()  # 保留原始频谱图副本，形状：[nframes, 80]
    indiv_mels = []  # 存储每帧对应的Mel频谱特征

    # ===== 步骤2：为每帧提取局部Mel频谱特征 =====
    for i in range(num_frames):
        # 计算当前帧对应的Mel频谱起始帧（向前偏移2帧以获取上下文）
        start_frame_num = i-2
        # 将帧索引转换为Mel频谱的采样点索引
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        # 生成帧索引序列（16个时间步）
        seq = list(range(start_idx, end_idx))
        # 边界裁剪：确保索引不超出频谱图范围
        seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
        # 提取对应时间步的Mel频谱
        m = spec[seq, :]
        # 转置后添加到列表：[80, 16]
        indiv_mels.append(m.T)
    indiv_mels = np.asarray(indiv_mels)  # 形状：[T, 80, 16]

    # ===== 步骤3：生成眨眼序列 =====
    ratio = generate_blink_seq_randomly(num_frames)  # 形状：[T]
    
    # ===== 步骤4：加载参考3DMM系数 =====
    source_semantics_path = first_coeff_path
    source_semantics_dict = scio.loadmat(source_semantics_path)
    # 提取参考系数的前70维（64维表情 + 6维姿态）
    ref_coeff = source_semantics_dict['coeff_3dmm'][:1,:70]  # 形状：[1, 70]
    # 将单帧参考系数扩展为与视频帧数相同的序列（所有帧使用相同的参考系数）
    ref_coeff = np.repeat(ref_coeff, num_frames, axis=0)

    # （可选）使用参考眨眼系数替换自动生成的眨眼序列
    # if ref_eyeblink_coeff_path is not None:
    #     ratio[:num_frames] = 0
    #     refeyeblink_coeff_dict = scio.loadmat(ref_eyeblink_coeff_path)
    #     refeyeblink_coeff = refeyeblink_coeff_dict['coeff_3dmm'][:,:64]
    #     refeyeblink_num_frames = refeyeblink_coeff.shape[0]
    #     if refeyeblink_num_frames<num_frames:
    #         div = num_frames//refeyeblink_num_frames
    #         re = num_frames%refeyeblink_num_frames
    #         refeyeblink_coeff_list = [refeyeblink_coeff for i in range(div)]
    #         refeyeblink_coeff_list.append(refeyeblink_coeff[:re, :64])
    #         refeyeblink_coeff = np.concatenate(refeyeblink_coeff_list, axis=0)
    #         print(refeyeblink_coeff.shape[0])

    #     ref_coeff[:, :64] = refeyeblink_coeff[:num_frames, :64] 
    
    # ===== 步骤5：转换为PyTorch张量并移至设备 =====
    # 将Mel频谱特征转换为张量并增加维度以匹配模型输入格式
    # 原始形状 [T, 80, 16] → [T, 1, 80, 16] → [1, T, 1, 80, 16]（添加batch维度）
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0) # bs T 1 80 16

    if use_blink:
        # 如果启用眨眼，将眨眼权重序列转换为张量并添加batch维度
        ratio = torch.FloatTensor(ratio).unsqueeze(0)  # 形状：[1, T]
    else:
        # 如果禁用眨眼，将所有眨眼权重设为0（不眨眼）
        ratio = torch.FloatTensor(ratio).unsqueeze(0).fill_(0.) 
                               # bs T
    # 将参考系数转换为张量并添加batch维度
    ref_coeff = torch.FloatTensor(ref_coeff).unsqueeze(0)  # 形状：[1, 1, 70]

    # 将所有张量移动到指定设备（GPU/CPU）
    indiv_mels = indiv_mels.to(device)
    ratio = ratio.to(device)
    ref_coeff = ref_coeff.to(device)

    # 返回整理好的批次数据
    return {'indiv_mels': indiv_mels,  # 每帧的Mel频谱特征
            'ref': ref_coeff,          # 参考3DMM系数
            'num_frames': num_frames,  # 视频总帧数
            'ratio_gt': ratio,         # 眨眼权重序列
            'audio_name': audio_name,  # 音频名称
            'pic_name': pic_name}      # 图片名称
