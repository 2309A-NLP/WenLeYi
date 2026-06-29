# videoio.py - 视频输入输出模块
# 本模块提供视频文件的读取和写入功能
# 包括视频帧加载、音视频合并、水印添加等操作

import shutil
import uuid
import subprocess
import os

import cv2


def load_video_to_cv2(input_path):
    """使用OpenCV加载视频文件为图片帧列表
    
    逐帧读取视频，将BGR格式转换为RGB格式后返回。
    
    Args:
        input_path: 视频文件路径
    
    Returns:
        RGB格式的视频帧列表（numpy数组）
    """
    video_stream = cv2.VideoCapture(input_path)  # 打开视频文件
    fps = video_stream.get(cv2.CAP_PROP_FPS)  # 获取视频帧率（未使用，仅读取）
    full_frames = []  # 存储所有视频帧
    while 1:
        still_reading, frame = video_stream.read()  # 逐帧读取
        if not still_reading:
            video_stream.release()  # 读取完毕，释放资源
            break 
        full_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # BGR转RGB并添加到列表
    return full_frames


def save_video_with_watermark(video, audio, save_path, watermark=False):
    """将视频和音频合并并保存
    
    使用FFmpeg命令行工具将视频和音频流合并。
    支持添加水印（当前未实现）。
    
    Args:
        video: 输入视频文件路径
        audio: 输入音频文件路径
        save_path: 输出文件路径
        watermark: 是否添加水印（当前未使用）
    """
    temp_file = str(uuid.uuid4())+'.mp4'  # 创建临时文件（UUID避免命名冲突）
    # os.system(cmd)
    try:
        # 首先尝试使用H.264编码器重新编码视频
        cmd = r'ffmpeg -y -hide_banner -loglevel error -i \"%s\" -i \"%s\" -vcodec h264 \"%s\"' % (video, audio, temp_file)
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        # 如果H.264编码失败，使用直接复制模式（不重新编码）
        cmd = r'ffmpeg -y -hide_banner -loglevel error -i \"%s\" -i \"%s\" -vcodec copy \"%s\"' % (video, audio, temp_file)
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not os.path.exists(temp_file):
        print("FFmpeg error")  # FFmpeg处理失败
    shutil.move(temp_file, save_path)  # 将临时文件移动到目标路径
