# preprocess.py - 预处理模块
# 本模块负责视频/图片的预处理流程，包括人脸检测、裁剪、
# 关键点提取和3D人脸模型参数（3DMM）提取
# 是SadTalker流水线中的核心预处理步骤

import numpy as np
import cv2, os, sys, torch
from tqdm import tqdm
from PIL import Image 

# 3D人脸模型参数提取相关模块
import safetensors
import safetensors.torch 
from src.face3d.util.preprocess import align_img  # 人脸对齐预处理
from src.face3d.util.load_mats import load_lm3d  # 加载3D人脸模型的3D关键点标准模板
from src.face3d.models import networks  # 3D人脸重建网络模型

from scipy.io import loadmat, savemat  # 用于保存/加载MATLAB格式的系数文件
from src.utils.croper import Preprocesser  # 人脸裁剪预处理器


import warnings

from src.utils.safetensor_helper import load_x_from_safetensor  # safetensor格式的模型加载辅助函数
warnings.filterwarnings("ignore")  # 忽略警告信息


def split_coeff(coeffs):
        """将3DMM系数向量拆分为各个分量
        
        3DMM（3D Morphable Model）系数是一个256维向量，
        包含身份、表情、纹理、姿态等多个维度的信息。
        
        Args:
            coeffs: 3DMM系数向量，shape=(B, 256)
        
        Returns:
            字典，包含各个分量：
            - 'id': 身份系数（0-80维），描述人脸的身份特征
            - 'exp': 表情系数（80-144维），描述面部表情
            - 'tex': 纹理系数（144-224维），描述皮肤纹理
            - 'angle': 旋转角度（224-227维），描述头部姿态
            - 'gamma': 光照系数（227-254维），描述光照条件
            - 'trans': 平移量（254-256维），描述人脸在图像中的位置
        """
        id_coeffs = coeffs[:, :80]  # 身份系数：描述"这是谁的脸"
        exp_coeffs = coeffs[:, 80: 144]  # 表情系数：描述面部表情变化
        tex_coeffs = coeffs[:, 144: 224]  # 纹理系数：描述皮肤纹理和颜色
        angles = coeffs[:, 224: 227]  # 旋转角度：描述头部的三维旋转
        gammas = coeffs[:, 227: 254]  # 光照系数：描述环境光照条件
        translations = coeffs[:, 254:]  # 平移量：描述人脸在二维图像中的位置
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }


class CropAndExtract():
    """人脸裁剪与3DMM参数提取类
    
    整合了人脸检测、裁剪、关键点提取和3D人脸模型参数提取的完整流程。
    从输入的视频/图片中提取人脸的3D形态参数，用于后续的表情驱动。
    """

    def __init__(self, sadtalker_path, device):
        """初始化裁剪与提取器
        
        Args:
            sadtalker_path: SadTalker模型路径字典
            device: 计算设备（'cuda'或'cpu'）
        """
        self.propress = Preprocesser(device)  # 初始化人脸预处理器
        # 定义并初始化3D人脸重建网络（ResNet50架构）
        self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(device)
        
        # 加载3D人脸重建模型的预训练权重
        if sadtalker_path['use_safetensor']:
            # 从safetensor格式加载模型权重
            checkpoint = safetensors.torch.load_file(sadtalker_path['checkpoint'])    
            self.net_recon.load_state_dict(load_x_from_safetensor(checkpoint, 'face_3drecon'))
        else:
            # 从PyTorch标准格式加载模型权重
            checkpoint = torch.load(sadtalker_path['path_of_net_recon_model'], map_location=torch.device(device))    
            self.net_recon.load_state_dict(checkpoint['net_recon'])

        self.net_recon.eval()  # 设置模型为评估模式（关闭Dropout和BatchNorm的训练行为）
        self.lm3d_std = load_lm3d(sadtalker_path['dir_of_BFM_fitting'])  # 加载3D人脸关键点标准模板
        self.device = device  # 记录计算设备
    
    def generate(self, input_path, save_dir, crop_or_resize='crop', source_image_flag=False, pic_size=256):
        """主处理函数：从视频/图片中提取人脸3DMM参数
        
        完整流程：
        1. 加载输入视频/图片
        2. 人脸检测和裁剪
        3. 提取人脸关键点
        4. 使用3D重建网络提取3DMM参数
        5. 保存结果到指定目录
        
        Args:
            input_path: 输入视频或图片的路径
            save_dir: 结果保存目录
            crop_or_resize: 预处理模式，可选：
                - 'crop': 裁剪人脸区域（默认）
                - 'full': 使用完整面部渲染模式
                - 'resize': 仅缩放不裁剪
            source_image_flag: 是否为源图片模式（仅处理第一帧）
            pic_size: 输出图片尺寸（默认256）
        
        Returns:
            (系数文件路径, 裁剪后图片路径, 裁剪信息) 或 (None, None) 如果未检测到人脸
        """
        # 生成输出文件名（不含扩展名）
        pic_name = os.path.splitext(os.path.split(input_path)[-1])[0]  

        # 构建输出文件路径
        landmarks_path =  os.path.join(save_dir, pic_name+'_landmarks.txt')  # 关键点文件
        coeff_path =  os.path.join(save_dir, pic_name+'.mat')  # 3DMM系数文件（MATLAB格式）
        png_path =  os.path.join(save_dir, pic_name+'.png')  # 裁剪后的人脸图片

        # 加载输入文件
        if not os.path.isfile(input_path):
            raise ValueError('input_path must be a valid path to video/image file')
        elif input_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            # 输入为图片文件，直接读取
            full_frames = [cv2.imread(input_path)]
            fps = 25  # 图片默认帧率设为25
        else:
            # 输入为视频文件，逐帧读取
            video_stream = cv2.VideoCapture(input_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
            full_frames = [] 
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break 
                full_frames.append(frame) 
                if source_image_flag:
                    break  # 源图片模式只取第一帧

        # 将BGR格式转换为RGB格式（OpenCV默认BGR，模型需要RGB）
        x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames] 

        #### 根据模式进行人脸裁剪或缩放
        if 'crop' in crop_or_resize.lower(): # 默认裁剪模式
            # 调用预处理器进行人脸裁剪，'ext'表示扩展裁剪（静态模式）
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop  # 裁剪区域的四个边界
            lx, ly, rx, ry = quad  # 对齐四边形的四个边界
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            # 计算裁剪区域在原图中的绝对坐标
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            # 保存裁剪信息：(裁剪尺寸, 裁剪区域, 对齐四边形)
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        elif 'full' in crop_or_resize.lower():
            # 完整面部渲染模式：同样进行裁剪但使用不同的渲染器配置
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        else: # 缩放模式：不裁剪，直接缩放到目标尺寸
            oy1, oy2, ox1, ox2 = 0, x_full_frames[0].shape[0], 0, x_full_frames[0].shape[1] 
            crop_info = ((ox2 - ox1, oy2 - oy1), None, None)

        # 将所有帧缩放到统一尺寸并转换为PIL Image格式
        frames_pil = [Image.fromarray(cv2.resize(frame,(pic_size, pic_size))) for frame in x_full_frames]
        if len(frames_pil) == 0:
            print('No face is detected in the input file')  # 未检测到人脸
            return None, None

        # 保存裁剪后的人脸图片
        for frame in frames_pil:
            cv2.imwrite(png_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        # 2. 根据检测到的人脸提取关键点
        if not os.path.isfile(landmarks_path): 
            # 如果关键点文件不存在，使用模型提取
            lm = self.propress.predictor.extract_keypoint(frames_pil, landmarks_path)
        else:
            print(' Using saved landmarks.')  # 使用已保存的关键点
            lm = np.loadtxt(landmarks_path).astype(np.float32)  # 从文件加载关键点
            lm = lm.reshape([len(x_full_frames), -1, 2])  # 重塑为(帧数, 关键点数, 2)

        if not os.path.isfile(coeff_path):
            # 如果3DMM系数文件不存在，使用3D重建网络提取
            # 从Deep3DFaceRecon_pytorch加载3DMM参数生成器
            video_coeffs, full_coeffs = [],  []
            for idx in tqdm(range(len(frames_pil)), desc='3DMM Extraction In Video:'):  # 逐帧提取，显示进度条
                frame = frames_pil[idx]
                W,H = frame.size  # 图片尺寸
                lm1 = lm[idx].reshape([-1, 2])  # 当前帧的关键点
            
                if np.mean(lm1) == -1:
                    # 如果关键点全为-1（未检测到），使用标准模板的关键点
                    lm1 = (self.lm3d_std[:, :2]+1)/2.
                    lm1 = np.concatenate(
                        [lm1[:, :1]*W, lm1[:, 1:2]*H], 1
                    )
                else:
                    # 翻转y坐标（坐标系转换：从图像坐标到标准坐标）
                    lm1[:, -1] = H - 1 - lm1[:, -1]

                # 对齐人脸到标准位置（使用仿射变换）
                trans_params, im1, lm1, _ = align_img(frame, lm1, self.lm3d_std)
 
                # 将变换参数展平为一维数组
                trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
                # 将图片转换为模型输入格式（归一化、转置、添加batch维度）
                im_t = torch.tensor(np.array(im1)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
                
                with torch.no_grad():  # 推理时禁用梯度计算
                    full_coeff = self.net_recon(im_t)  # 3D重建网络前向传播
                    coeffs = split_coeff(full_coeff)  # 拆分3DMM系数

                # 将系数从GPU转移到CPU并转为numpy数组
                pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
 
                # 拼接最终的系数向量：表情 + 旋转角度 + 平移量 + 变换参数
                pred_coeff = np.concatenate([
                    pred_coeff['exp'],  # 表情系数
                    pred_coeff['angle'],  # 旋转角度
                    pred_coeff['trans'],  # 平移量
                    trans_params[2:][None],  # 仿射变换参数
                    ], 1)
                video_coeffs.append(pred_coeff)  # 保存语义系数
                full_coeffs.append(full_coeff.cpu().numpy())  # 保存完整系数

            # 取第一帧的系数作为语义系数（用于驱动）
            semantic_npy = np.array(video_coeffs)[:,0] 

            # 保存3DMM系数到MATLAB格式文件
            savemat(coeff_path, {'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]})

        return coeff_path, png_path, crop_info
