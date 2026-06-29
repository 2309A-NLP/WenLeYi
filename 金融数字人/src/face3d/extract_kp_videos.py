"""
从视频中提取面部关键点（landmark）的脚本。

该脚本用于从输入目录中的视频文件中提取68个面部关键点，
支持多GPU并行处理，将结果保存为txt文件。
"""

import os
import cv2
import time
import glob
import argparse
import face_alignment
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import cycle

# 导入多进程相关模块，用于并行处理多个视频
from torch.multiprocessing import Pool, Process, set_start_method

class KeypointExtractor():
    """面部关键点提取器类
    
    使用face_alignment库进行2D面部关键点检测。
    """
    
    def __init__(self, device):
        """初始化关键点提取器
        
        Args:
            device: 计算设备，如 'cuda:0' 或 'cpu'
        """
        # 使用face_alignment库初始化2D面部关键点检测器
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                                                     device=device)   

    def extract_keypoint(self, images, name=None, info=True):
        """提取面部关键点
        
        支持两种输入模式：
        1. 单张图片：直接提取关键点
        2. 图片列表：批量提取并保存结果
        
        Args:
            images: 单张图片（PIL.Image）或图片列表
            name: 输出文件名（不含扩展名），用于保存关键点
            info: 是否显示进度条
        
        Returns:
            keypoints: numpy数组，形状为 (N, 68, 2) 或 (68, 2)
        """
        if isinstance(images, list):
            # 批量处理模式：对列表中的每张图片提取关键点
            keypoints = []
            # 根据info参数决定是否显示进度条
            if info:
                i_range = tqdm(images,desc='landmark Det:')
            else:
                i_range = images

            for image in i_range:
                # 递归调用自身处理单张图片
                current_kp = self.extract_keypoint(image)
                # 如果检测失败（返回-1），则使用上一帧的关键点（相邻帧一致性）
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            # 合并所有关键点并保存到txt文件
            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints
        else:
            # 单张图片处理模式
            while True:
                try:
                    # 调用face_alignment检测器获取关键点
                    keypoints = self.detector.get_landmarks_from_image(np.array(images))[0]
                    break
                except RuntimeError as e:
                    # CUDA内存不足时等待1秒后重试
                    if str(e).startswith('CUDA'):
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break   
                except TypeError:
                    # 未检测到人脸时，返回全-1的数组
                    print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)                    
                    break
            # 如果指定了文件名，保存关键点到txt文件
            if name is not None:
                np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints

def read_video(filename):
    """读取视频文件并返回所有帧的列表
    
    Args:
        filename: 视频文件路径
    
    Returns:
        frames: PIL.Image列表，包含视频的所有帧
    """
    frames = []
    # 使用OpenCV打开视频文件
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # 将BGR格式转换为RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为PIL.Image格式
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            break
    # 释放视频捕获资源
    cap.release()
    return frames

def run(data):
    """多进程处理函数，用于处理单个视频文件
    
    Args:
        data: 包含文件名、配置参数和设备ID的元组
    """
    filename, opt, device = data
    # 设置当前进程使用的GPU设备
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    # 创建关键点提取器实例
    kp_extractor = KeypointExtractor()
    # 读取视频帧
    images = read_video(filename)
    # 获取视频文件名（目录名和文件名）
    name = filename.split('/')[-2:]
    # 创建输出目录
    os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
    # 提取关键点并保存
    kp_extractor.extract_keypoint(
        images, 
        name=os.path.join(opt.output_dir, name[-2], name[-1])
    )

if __name__ == '__main__':
    # 设置多进程启动方法为'spawn'，避免CUDA相关问题
    set_start_method('spawn')
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, help='the folder of the input files')
    parser.add_argument('--output_dir', type=str, help='the folder of the output files')
    parser.add_argument('--device_ids', type=str, default='0,1')
    parser.add_argument('--workers', type=int, default=4)

    opt = parser.parse_args()
    filenames = list()
    # 定义支持的视频文件扩展名
    VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS
    
    # 遍历所有支持的扩展名，收集视频文件列表
    for ext in extensions:
        os.listdir(f'{opt.input_dir}')
        print(f'{opt.input_dir}/*.{ext}')
        filenames = sorted(glob.glob(f'{opt.input_dir}/*.{ext}'))
    print('Total number of videos:', len(filenames))
    
    # 创建进程池进行并行处理
    pool = Pool(opt.workers)
    args_list = cycle([opt])
    # 解析GPU设备ID列表，循环使用
    device_ids = opt.device_ids.split(",")
    device_ids = cycle(device_ids)
    # 使用imap_unordered进行并行处理，tqdm显示进度
    for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids))):
        None
