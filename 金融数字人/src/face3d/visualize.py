"""
3DMM特征与音频同步可视化脚本

该脚本用于生成组合视频，将3D人脸模型渲染结果与音频同步。
主要功能包括：加载系数、渲染3D人脸、合成视频和音频。
"""

# 导入必要的库
import cv2
import numpy as np
from src.face3d.models.bfm import ParametricFaceModel
from src.face3d.models.facerecon_model import FaceReconModel
import torch
import subprocess, platform
import scipy.io as scio
from tqdm import tqdm 

# 草稿版本的视频生成函数
def gen_composed_video(args, device, first_frame_coeff, coeff_path, audio_path, save_path, exp_dim=64):
    """生成组合视频
    
    将3D人脸模型渲染结果与音频合成为最终视频。
    
    Args:
        args: 配置参数
        device: 计算设备（如'cuda:0'）
        first_frame_coeff: 第一帧的完整3DMM系数文件路径
        coeff_path: 预测的3DMM系数文件路径
        audio_path: 音频文件路径
        save_path: 输出视频保存路径
        exp_dim: 表情维度，默认64
    """
    
    # 加载第一帧的完整3DMM系数
    coeff_first = scio.loadmat(first_frame_coeff)['full_3dmm']

    # 加载预测的3DMM系数（表情和头部运动）
    coeff_pred = scio.loadmat(coeff_path)['coeff_3dmm']

    # 复制第一帧系数到所有帧（作为基础系数）
    # coeff_full包含257维系数：身份80 + 表情64 + 纹理80 + 旋转3 + 光照27 + 平移3
    coeff_full = np.repeat(coeff_first, coeff_pred.shape[0], axis=0) # 257

    # 用预测的系数替换对应的维度
    coeff_full[:, 80:144] = coeff_pred[:, 0:64]  # 表情系数
    coeff_full[:, 224:227]  = coeff_pred[:, 64:67] # 旋转角度（3维）
    coeff_full[:, 254:]  = coeff_pred[:, 67:] # 平移向量（3维）

    # 临时视频文件路径
    tmp_video_path = '/tmp/face3dtmp.mp4'

    # 初始化面部重建模型
    facemodel = FaceReconModel(args)
    
    # 创建视频写入器
    # 参数：输出路径、编码器、帧率、分辨率
    video = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (224, 224))

    # 逐帧渲染3D人脸
    for k in tqdm(range(coeff_pred.shape[0]), 'face3d rendering:'):
        # 将当前帧的系数转换为torch张量并移到指定设备
        cur_coeff_full = torch.tensor(coeff_full[k:k+1], device=device)

        # 前向传播，渲染3D人脸
        facemodel.forward(cur_coeff_full, device)

        # 获取预测的关键点（可用于后续可视化）
        predicted_landmark = facemodel.pred_lm # TODO.
        predicted_landmark = predicted_landmark.cpu().numpy().squeeze()

        # 获取渲染的人脸图像
        rendered_img = facemodel.pred_face
        # 将渲染结果转换为numpy数组（0-255范围）
        rendered_img = 255. * rendered_img.cpu().numpy().squeeze().transpose(1,2,0)
        # 转换为uint8类型
        out_img = rendered_img[:, :, :3].astype(np.uint8)

        # 将RGB格式转换为BGR格式（OpenCV使用BGR），写入视频
        video.write(np.uint8(out_img[:,:,::-1]))

    # 释放视频写入器
    video.release()

    # 使用ffmpeg将视频和音频合成为最终文件
    # 参数：静默输出、覆盖已有文件、输入视频和音频、合并输出
    command = 'ffmpeg -v quiet -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, tmp_video_path, save_path)
    # 根据操作系统选择shell参数
    subprocess.call(command, shell=platform.system() != 'Windows')
