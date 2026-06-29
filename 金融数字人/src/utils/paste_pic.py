# paste_pic.py - 人脸粘贴模块
# 本模块将增强后的人脸区域无缝融合回原始背景图片
# 使用OpenCV的seamlessClone实现自然的人脸融合效果

import cv2, os
import numpy as np
from tqdm import tqdm
import uuid

from src.utils.videoio import save_video_with_watermark  # 导入视频保存工具


def paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False):
    """将生成的人脸视频无缝粘贴回原始背景图片
    
    流程：
    1. 加载原始背景图片（或视频第一帧）
    2. 加载生成的人脸视频帧
    3. 根据裁剪信息调整人脸位置
    4. 使用seamlessClone将每帧人脸融合到背景中
    5. 合并音频并保存最终视频
    
    Args:
        video_path: 生成的人脸视频路径
        pic_path: 原始背景图片或视频路径
        crop_info: 裁剪信息元组，包含 (裁剪尺寸, 裁剪区域, 对齐四边形)
        new_audio_path: 新的音频文件路径
        full_video_path: 最终输出视频路径
        extended_crop: 是否使用扩展裁剪区域（使用更大的裁剪范围）
    
    Raises:
        ValueError: 如果pic_path不是有效文件路径
    """
    # 验证输入文件是否存在
    if not os.path.isfile(pic_path):
        raise ValueError('pic_path must be a valid path to video/image file')
    elif pic_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        # 输入为图片文件，直接读取
        full_img = cv2.imread(pic_path)
    else:
        # 输入为视频文件，读取第一帧作为背景
        video_stream = cv2.VideoCapture(pic_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = [] 
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break 
            break  # 只取第一帧
        full_img = frame
    # 获取背景图片的尺寸
    frame_h = full_img.shape[0]  # 高度
    frame_w = full_img.shape[1]  # 宽度

    # 加载生成的人脸视频的所有帧
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    crop_frames = []  # 存储裁剪后的人脸帧
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        crop_frames.append(frame)
    
    # 验证裁剪信息的完整性
    if len(crop_info) != 3:
        print("you didn't crop the image")  # 未裁剪则直接返回
        return
    else:
        r_w, r_h = crop_info[0]  # 裁剪后的尺寸 (宽, 高)
        clx, cly, crx, cry = crop_info[1]  # 裁剪区域的四个边界
        lx, ly, rx, ry = crop_info[2]  # 对齐四边形的四个边界
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        # 计算在原图中的实际位置（裁剪区域 + 对齐偏移）
        # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
        # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

        if extended_crop:
            # 扩展裁剪模式：使用更大的裁剪范围
            oy1, oy2, ox1, ox2 = cly, cry, clx, crx
        else:
            # 标准模式：使用精确的对齐区域
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

    # 创建临时视频文件用于保存融合结果
    tmp_path = str(uuid.uuid4())+'.mp4'  # 使用UUID作为临时文件名
    out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_w, frame_h))
    # 逐帧进行人脸融合
    for crop_frame in tqdm(crop_frames, 'seamlessClone:'):  # 显示进度条
        # 将人脸帧缩放到目标区域大小
        p = cv2.resize(crop_frame.astype(np.uint8), (ox2-ox1, oy2 - oy1)) 
        
        # 创建全白掩码（255表示需要融合的区域）
        mask = 255*np.ones(p.shape, p.dtype)
        # 计算融合位置（目标区域的中心点）
        location = ((ox1+ox2) // 2, (oy1+oy2) // 2)
        # 使用seamlessClone将人脸无缝融合到背景图片中
        # NORMAL_CLONE模式：混合颜色并保持自然的边缘过渡
        gen_img = cv2.seamlessClone(p, full_img, mask, location, cv2.NORMAL_CLONE)
        out_tmp.write(gen_img)  # 写入融合后的帧

    out_tmp.release()  # 释放视频写入器

    # 将视频和音频合并，保存最终结果
    save_video_with_watermark(tmp_path, new_audio_path, full_video_path, watermark=False)
    os.remove(tmp_path)  # 删除临时视频文件
