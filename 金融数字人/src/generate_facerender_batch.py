"""
人脸渲染数据批处理生成模块
负责将3DMM系数和参考图像转换为面部渲染模型所需的输入格式。
包括图像预处理、语义特征提取、目标系数变换、相机姿态生成等功能。
是SadTalker数字人系统中面部渲染部分的数据准备模块。
"""

import numpy as np
from PIL import Image
from skimage import  img_as_float32, transform
import torch
import scipy.io as scio
import os

def get_facerender_data(coeff, pic_path, first_coeff_path, audio_path, 
                        batch_size, input_yaw_list=None, input_pitch_list=None, input_roll_list=None, 
                        expression_scale=1.0, still_mode = False, preprocess='crop', size = 256, facemodel='facevid2vid'):
    """
    获取人脸渲染所需的全部数据。
    
    该函数将3DMM系数、参考图像、音频等信息整合为渲染模型所需的输入字典。
    主要步骤：
        1. 加载并预处理参考图像（缩放、归一化、维度转换）
        2. 加载参考图像的3DMM系数作为源语义特征
        3. 对预测的3DMM系数应用表情缩放和静态模式处理
        4. 为每帧生成目标语义特征序列
        5. 整理所有数据为模型输入格式
    
    参数：
        coeff (np.ndarray): 预测的3DMM系数序列 [T, 70]，包含表情和姿态系数
        pic_path (str): 参考图像文件路径
        first_coeff_path (str): 参考图像的3DMM系数文件路径（.mat格式）
        audio_path (str): 输入音频文件路径
        batch_size (int): 批次大小
        input_yaw_list (list): 自定义偏航角序列（可选，默认None使用预测值）
        input_pitch_list (list): 自定义俯仰角序列（可选）
        input_roll_list (list): 自定义翻滚角序列（可选）
        expression_scale (float): 表情缩放系数，默认1.0（不缩放）
        still_mode (bool): 静态模式，禁用头部运动，默认False
        preprocess (str): 预处理模式，'crop'或'full'，默认'crop'
        size (int): 输出图像尺寸，默认256×256
        facemodel (str): 人脸模型类型，默认'facevid2vid'
    
    返回：
        dict: 包含以下键的字典：
            - 'source_image': 参考图像张量 [batch_size, 3, size, size]
            - 'source_semantics': 源语义特征 [batch_size, 70, 27]
            - 'target_semantics_list': 目标语义特征 [batch_size, T, 70, 27]
            - 'frame_num': 视频总帧数
            - 'video_name': 输出视频名称
            - 'audio_path': 音频文件路径
    """
    # 语义特征的邻域半径（用于构建时间窗口），2*radius+1 = 27
    semantic_radius = 13
    # 生成视频名称（图片名_音频名的组合）
    video_name = f"{os.path.basename(pic_path).split('.')[0]}_{os.path.basename(audio_path).split('.')[0]}"
    # txt_path = os.path.splitext(coeff_path)[0]

    data={}  # 存储所有渲染数据的字典

    # ===== 步骤1：加载并预处理参考图像 =====
    img1 = Image.open(pic_path)  # 使用PIL打开参考图像
    source_image = np.array(img1)  # 转换为NumPy数组
    source_image = img_as_float32(source_image)  # 转换为float32类型（归一化到[0,1]范围）
    source_image = transform.resize(source_image, (size, size, 3))  # 缩放到目标尺寸
    source_image = source_image.transpose((2, 0, 1))  # 维度转换：HWC → CHW（通道在前）
    source_image_ts = torch.FloatTensor(source_image).unsqueeze(0)  # 转为张量并添加batch维度
    source_image_ts = source_image_ts.repeat(batch_size, 1, 1, 1)  # 扩展到批次大小
    data['source_image'] = source_image_ts  # 存入数据字典
 
    # ===== 步骤2：加载参考3DMM系数 =====
    source_semantics_dict = scio.loadmat(first_coeff_path)
    # generated_dict = scio.loadmat(coeff_path)

    # 根据预处理模式和人脸模型选择不同的系数维度
    if 'full' not in preprocess.lower() and facemodel != 'pirender':
        # 标准模式：使用前70维系数（64维表情 + 6维姿态）
        source_semantics = source_semantics_dict['coeff_3dmm'][:1,:70]  # 形状：[1, 70]
        generated_3dmm = coeff[:,:70]
    else:
        # full模式或pirender模型：使用前73维系数（额外3维用于完整头部模型）
        source_semantics = source_semantics_dict['coeff_3dmm'][:1,:73]  # 形状：[1, 73]
        generated_3dmm = coeff[:,:70]

    # 对源语义特征进行时间窗口变换
    source_semantics_new = transform_semantic_1(source_semantics, semantic_radius)
    source_semantics_ts = torch.FloatTensor(source_semantics_new).unsqueeze(0)
    source_semantics_ts = source_semantics_ts.repeat(batch_size, 1, 1)
    data['source_semantics'] = source_semantics_ts

    # ===== 步骤3：处理目标3DMM系数 =====
    # 应用表情缩放系数，增强或减弱面部表情幅度
    generated_3dmm[:, :64] = generated_3dmm[:, :64] * expression_scale

    # 在full模式或pirender模型下，补充额外的系数维度
    if 'full' in preprocess.lower() or facemodel == 'pirender':
        generated_3dmm = np.concatenate([generated_3dmm, np.repeat(source_semantics[:,70:], generated_3dmm.shape[0], axis=0)], axis=1)

    # 在静态模式下，将姿态系数替换为源图像的姿态（保持头部不动）
    if still_mode:
        generated_3dmm[:, 64:] = np.repeat(source_semantics[:, 64:], generated_3dmm.shape[0], axis=0)

    # （可选）将系数保存为文本文件用于调试
    # with open(txt_path+'.txt', 'w') as f:
    #     for coeff in generated_3dmm:
    #         for i in coeff:
    #             f.write(str(i)[:7]   + '  '+'\t')
    #         f.write('\n')

    # ===== 步骤4：为每帧生成目标语义特征 =====
    target_semantics_list = []  # 存储每帧的目标语义特征
    frame_num = generated_3dmm.shape[0]
    data['frame_num'] = frame_num
    
    for frame_idx in range(frame_num):
        # 对每一帧，以其为中心构建时间窗口的语义特征
        target_semantics = transform_semantic_target(generated_3dmm, frame_idx, semantic_radius)
        target_semantics_list.append(target_semantics)

    # 填充帧数使其成为batch_size的整数倍（padding）
    remainder = frame_num%batch_size
    if remainder!=0:
        # 使用最后一帧的特征填充剩余位置
        for _ in range(batch_size-remainder):
            target_semantics_list.append(target_semantics)

    # 将列表转换为NumPy数组并重塑为批次格式
    # 形状：[frame_num, 70, 27] → [batch_size, T, 70, 27]
    target_semantics_np = np.array(target_semantics_list)  # 形状：[frame_num, 70, semantic_radius*2+1]
    target_semantics_np = target_semantics_np.reshape(batch_size, -1, target_semantics_np.shape[-2], target_semantics_np.shape[-1])
    data['target_semantics_list'] = torch.FloatTensor(target_semantics_np)
    data['video_name'] = video_name
    data['audio_path'] = audio_path
    
    # （可选）自定义相机姿态角度（当前已禁用，使用模型预测的姿态）
    # if input_yaw_list is not None:
    #     yaw_c_seq = gen_camera_pose(input_yaw_list, frame_num, batch_size)
    #     data['yaw_c_seq'] = torch.FloatTensor(yaw_c_seq)
    # if input_pitch_list is not None:
    #     pitch_c_seq = gen_camera_pose(input_pitch_list, frame_num, batch_size)
    #     data['pitch_c_seq'] = torch.FloatTensor(pitch_c_seq)
    # if input_roll_list is not None:
    #     roll_c_seq = gen_camera_pose(input_roll_list, frame_num, batch_size) 
    #     data['roll_c_seq'] = torch.FloatTensor(roll_c_seq)
 
    return data

def transform_semantic_1(semantic, semantic_radius):
    """
    将源语义特征扩展为时间窗口格式。
    
    将单帧的源3DMM系数复制(2*semantic_radius+1)次，
    并转置为模型所需的 [系数维度, 时间窗口] 格式。
    这样源特征在整个时间窗口内保持不变。
    
    参数：
        semantic (np.ndarray): 源3DMM系数 [1, 70]
        semantic_radius (int): 时间窗口半径（总窗口大小 = 2*radius+1）
    
    返回：
        np.ndarray: 扩展后的语义特征 [70, 27]
    """
    # 将源系数重复(2*radius+1)次以构建时间窗口
    semantic_list =  [semantic for i in range(0, semantic_radius*2+1)]
    coeff_3dmm = np.concatenate(semantic_list, 0)
    # 转置：[窗口数, 70] → [70, 窗口数]
    return coeff_3dmm.transpose(1,0)

def transform_semantic_target(coeff_3dmm, frame_index, semantic_radius):
    """
    为目标帧生成时间窗口内的语义特征。
    
    以当前帧为中心，取左右各semantic_radius帧的3DMM系数，
    构建时间上下文窗口。超出边界的部分使用边界值填充。
    
    参数：
        coeff_3dmm (np.ndarray): 所有帧的3DMM系数 [T, 70]
        frame_index (int): 当前帧的索引
        semantic_radius (int): 时间窗口半径
    
    返回：
        np.ndarray: 时间窗口内的语义特征 [70, 27]
    """
    num_frames = coeff_3dmm.shape[0]
    # 生成以当前帧为中心的时间窗口索引序列
    seq = list(range(frame_index- semantic_radius, frame_index + semantic_radius+1))
    # 边界裁剪：确保索引不超出[0, num_frames-1]范围
    index = [ min(max(item, 0), num_frames-1) for item in seq ] 
    # 提取窗口内的3DMM系数
    coeff_3dmm_g = coeff_3dmm[index, :]
    # 转置：[27, 70] → [70, 27]
    return coeff_3dmm_g.transpose(1,0)

def gen_camera_pose(camera_degree_list, frame_num, batch_size):
    """
    生成相机姿态角度序列。
    
    将用户指定的相机角度列表插值为逐帧的角度序列，
    用于控制生成视频中的虚拟相机运动。
    支持固定角度和多角度切换两种模式。
    
    参数：
        camera_degree_list (list): 相机角度列表。
            - 单元素列表：全程固定该角度
            - 多元素列表：在各角度之间进行线性插值
        frame_num (int): 视频总帧数
        batch_size (int): 批次大小
    
    返回：
        np.ndarray: 角度序列，形状为 [batch_size, T/batch_size]
    """
    new_degree_list = [] 
    
    # 情况1：只有一个角度值，全程使用固定角度
    if len(camera_degree_list) == 1:
        for _ in range(frame_num):
            new_degree_list.append(camera_degree_list[0]) 
        # 填充剩余位置使其成为batch_size的整数倍
        remainder = frame_num%batch_size
        if remainder!=0:
            for _ in range(batch_size-remainder):
                new_degree_list.append(new_degree_list[-1])
        # 重塑为批次格式
        new_degree_np = np.array(new_degree_list).reshape(batch_size, -1) 
        return new_degree_np

    # 情况2：多个角度值，进行线性插值
    # 计算相邻角度之间的总变化量
    degree_sum = 0.
    for i, degree in enumerate(camera_degree_list[1:]):
        degree_sum += abs(degree-camera_degree_list[i])
    
    # 计算每帧的平均角度变化量
    degree_per_frame = degree_sum/(frame_num-1)
    
    # 在各角度之间生成线性插值序列
    for i, degree in enumerate(camera_degree_list[1:]):
        degree_last = camera_degree_list[i]
        # 计算当前段的角度步长（考虑正负方向）
        degree_step = degree_per_frame * abs(degree-degree_last)/(degree-degree_last)
        new_degree_list =  new_degree_list + list(np.arange(degree_last, degree, degree_step))
    
    # 调整序列长度以匹配总帧数
    if len(new_degree_list) > frame_num:
        # 过长则截断
        new_degree_list = new_degree_list[:frame_num]
    elif len(new_degree_list) < frame_num:
        # 过短则用最后一个值填充
        for _ in range(frame_num-len(new_degree_list)):
            new_degree_list.append(new_degree_list[-1])
    print(len(new_degree_list))
    print(frame_num)

    # 填充剩余位置使其成为batch_size的整数倍
    remainder = frame_num%batch_size
    if remainder!=0:
        for _ in range(batch_size-remainder):
            new_degree_list.append(new_degree_list[-1])
    # 重塑为批次格式
    new_degree_np = np.array(new_degree_list).reshape(batch_size, -1) 
    return new_degree_np
