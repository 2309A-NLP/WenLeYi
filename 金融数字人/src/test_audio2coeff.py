"""
音频到3DMM系数转换模块（Audio to Coefficients）
该模块负责将音频信号转换为3D人脸模型的系数（3DMM coefficients），
包括表情系数（expression）和姿态系数（pose）。
是SadTalker数字人系统中的核心推理模块之一。
"""

import os 
import torch
import numpy as np
from scipy.io import savemat, loadmat
from yacs.config import CfgNode as CN
from scipy.signal import savgol_filter

import safetensors
import safetensors.torch 

# 导入音频到姿态（Audio2Pose）模型
from src.audio2pose_models.audio2pose import Audio2Pose
# 导入音频到表情（Audio2Exp）模型的网络结构
from src.audio2exp_models.networks import SimpleWrapperV2 
# 导入音频到表情模型
from src.audio2exp_models.audio2exp import Audio2Exp
# 导入safetensor格式的模型加载辅助函数
from src.utils.safetensor_helper import load_x_from_safetensor  

def load_cpk(checkpoint_path, model=None, optimizer=None, device="cpu"):
    """
    加载模型检查点（checkpoint）。
    
    从指定路径加载模型权重和优化器状态，
    用于恢复训练或进行推理。
    
    参数：
        checkpoint_path (str): 检查点文件路径
        model: 要加载权重的模型实例（可选）
        optimizer: 要加载状态的优化器实例（可选）
        device (str): 设备类型，默认为"cpu"
    
    返回：
        int: 检查点中保存的训练轮次（epoch）编号
    """
    # 加载检查点文件，将数据映射到指定设备
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    # 如果提供了模型实例，加载模型权重
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    # 如果提供了优化器实例，加载优化器状态
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 返回保存的训练轮次
    return checkpoint['epoch']

class Audio2Coeff():
    """
    音频到3DMM系数转换类。
    
    该类封装了两个核心子模型：
        1. Audio2Pose：将音频特征转换为头部姿态系数（6维：3维旋转 + 3维平移）
        2. Audio2Exp：将音频特征转换为面部表情系数（64维）
    
    最终将两个模型的输出拼接为70维的3DMM系数（64维表情 + 6维姿态），
    用于后续的面部渲染。
    
    使用流程：
        1. 初始化：加载预训练模型权重
        2. generate()：输入音频批次数据，输出3DMM系数序列
    """

    def __init__(self, sadtalker_path, device):
        """
        初始化Audio2Coeff模型。
        
        参数：
            sadtalker_path (dict): 包含各种模型路径的字典，键包括：
                - 'audio2pose_yaml_path': Audio2Pose模型的YAML配置文件路径
                - 'audio2exp_yaml_path': Audio2Exp模型的YAML配置文件路径
                - 'audio2pose_checkpoint': Audio2Pose模型的检查点路径
                - 'audio2exp_checkpoint': Audio2Exp模型的检查点路径
                - 'use_safetensor': 是否使用safetensor格式加载权重
                - 'checkpoint': safetensor格式的统一检查点路径
            device: 计算设备（CPU或GPU）
        """
        # 加载Audio2Pose模型的配置文件
        fcfg_pose = open(sadtalker_path['audio2pose_yaml_path'])
        cfg_pose = CN.load_cfg(fcfg_pose)
        cfg_pose.freeze()  # 冻结配置，防止后续意外修改

        # 加载Audio2Exp模型的配置文件
        fcfg_exp = open(sadtalker_path['audio2exp_yaml_path'])
        cfg_exp = CN.load_cfg(fcfg_exp)
        cfg_exp.freeze()

        # ===== 加载Audio2Pose模型 =====
        # 创建Audio2Pose模型实例，输入为音频特征，输出为姿态系数
        self.audio2pose_model = Audio2Pose(cfg_pose, None, device=device)
        # 将模型移动到指定设备（GPU/CPU）
        self.audio2pose_model = self.audio2pose_model.to(device)
        # 设置为评估模式（禁用Dropout等训练行为）
        self.audio2pose_model.eval()
        # 冻结所有参数，不计算梯度（推理时不需要）
        for param in self.audio2pose_model.parameters():
            param.requires_grad = False 
        
        try:
            # 根据配置选择加载方式
            if sadtalker_path['use_safetensor']:
                # 使用safetensor格式加载（更安全、更高效）
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                self.audio2pose_model.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2pose'))
            else:
                # 使用传统checkpoint格式加载
                load_cpk(sadtalker_path['audio2pose_checkpoint'], model=self.audio2pose_model, device=device)
        except:
            raise Exception("Failed in loading audio2pose_checkpoint")

        # ===== 加载Audio2Exp模型 =====
        # 创建SimpleWrapperV2网络（Audio2Exp的底层生成器）
        netG = SimpleWrapperV2()
        netG = netG.to(device)
        # 冻结参数
        for param in netG.parameters():
            netG.requires_grad = False
        netG.eval()
        try:
            # 加载网络权重
            if sadtalker_path['use_safetensor']:
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                netG.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))
            else:
                load_cpk(sadtalker_path['audio2exp_checkpoint'], model=netG, device=device)
        except:
            raise Exception("Failed in loading audio2exp_checkpoint")
        
        # 创建Audio2Exp模型包装器，封装网络和配置
        self.audio2exp_model = Audio2Exp(netG, cfg_exp, device=device, prepare_training_loss=False)
        self.audio2exp_model = self.audio2exp_model.to(device)
        # 冻结参数
        for param in self.audio2exp_model.parameters():
            param.requires_grad = False
        self.audio2exp_model.eval()
 
        # 保存设备引用
        self.device = device

    def generate(self, batch, coeff_save_dir, pose_style, ref_pose_coeff_path=None):
        """
        生成3DMM系数（表情 + 姿态）。
        
        对输入的音频批次数据进行推理，生成每帧对应的3DMM系数，
        包括64维表情系数和6维姿态系数，共70维。
        
        参数：
            batch (dict): 输入批次数据，包含：
                - 'indiv_mels': 音频Mel频谱特征 [bs, T, 1, 80, 16]
                - 'ref': 参考系数 [bs, 1, 70]
                - 'ratio_gt': 眨眼比例 [bs, T]
                - 'class': 姿态风格类别ID
                - 'pic_name': 图片名称
                - 'audio_name': 音频名称
            coeff_save_dir (str): 系数保存目录（当前未使用）
            pose_style (int): 姿态风格编号（0-45，共46种可选风格）
            ref_pose_coeff_path (str): 参考姿态系数文件路径（可选）
        
        返回：
            np.ndarray: 3DMM系数数组，形状为 [T, 70]，其中前64维为表情系数，
                       后6维为姿态系数
        """
        # 使用torch.no_grad()上下文管理器，禁用梯度计算（节省内存，加速推理）
        with torch.no_grad():
            # ===== 步骤1：预测表情系数 =====
            # 调用Audio2Exp模型进行推理
            #test
            results_dict_exp= self.audio2exp_model.test(batch)
            exp_pred = results_dict_exp['exp_coeff_pred']  # 表情系数预测结果，形状：[bs, T, 64]

            # ===== 步骤2：预测姿态系数 =====
            # 设置姿态风格类别
            batch['class'] = torch.LongTensor([pose_style]).to(self.device)
            # 调用Audio2Pose模型进行推理
            results_dict_pose = self.audio2pose_model.test(batch) 
            pose_pred = results_dict_pose['pose_pred']  # 姿态系数预测结果，形状：[bs, T, 6]

            # ===== 步骤3：对姿态系数进行平滑处理 =====
            # 使用Savitzky-Golay滤波器平滑姿态轨迹，减少抖动
            pose_len = pose_pred.shape[1]
            if pose_len<13: 
                # 如果帧数较少，使用动态窗口大小
                pose_len = int((pose_len-1)/2)*2+1  # 确保窗口大小为奇数
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), pose_len, 2, axis=1)).to(self.device)
            else:
                # 帧数足够时，使用固定窗口大小13进行平滑
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), 13, 2, axis=1)).to(self.device) 
            
            # ===== 步骤4：拼接表情和姿态系数 =====
            # 将表情系数（64维）和姿态系数（6维）在最后一维拼接，得到70维系数
            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)  # 形状：[bs, T, 70]

            # 将结果从GPU移到CPU，并转换为NumPy数组
            coeffs_pred_numpy = coeffs_pred[0].clone().detach().cpu().numpy() 

            # （可选）如果指定了参考姿态路径，可以使用参考姿态进行相对姿态计算
            # if ref_pose_coeff_path is not None: 
            #     coeffs_pred_numpy = self.using_refpose(coeffs_pred_numpy, ref_pose_coeff_path)
        
            # （可选）将系数保存为MAT文件用于调试
            # savemat(os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['pic_name'], batch['audio_name'])),  
            #         {'coeff_3dmm': coeffs_pred_numpy})

            return coeffs_pred_numpy
    
    def using_refpose(self, coeffs_pred_numpy, ref_pose_coeff_path):
        """
        使用参考姿态进行相对姿态计算。
        
        通过参考姿态文件中的头部姿态信息，计算相对于参考姿态的
        相对头部运动，使生成的姿态变化更加自然。
        
        参数：
            coeffs_pred_numpy (np.ndarray): 预测的3DMM系数 [T, 70]
            ref_pose_coeff_path (str): 参考姿态系数文件路径（.mat格式）
        
        返回：
            np.ndarray: 更新后的3DMM系数 [T, 70]
        """
        # 获取总帧数
        num_frames = coeffs_pred_numpy.shape[0]
        # 加载参考姿态系数文件
        refpose_coeff_dict = loadmat(ref_pose_coeff_path)
        # 提取参考姿态系数中的姿态部分（第64-70列，共6维）
        refpose_coeff = refpose_coeff_dict['coeff_3dmm'][:,64:70]
        refpose_num_frames = refpose_coeff.shape[0]
        
        # 如果参考姿态的帧数少于目标帧数，通过重复和截断来匹配
        if refpose_num_frames<num_frames:
            div = num_frames//refpose_num_frames  # 需要重复的完整次数
            re = num_frames%refpose_num_frames     # 剩余需要补充的帧数
            refpose_coeff_list = [refpose_coeff for i in range(div)]
            refpose_coeff_list.append(refpose_coeff[:re, :])
            refpose_coeff = np.concatenate(refpose_coeff_list, axis=0)

        # 计算相对头部姿态：当前姿态 + (参考姿态序列 - 参考姿态第一帧)
        # 这样可以保留预测的头部运动，同时加入参考姿态的偏移
        coeffs_pred_numpy[:, 64:70] = coeffs_pred_numpy[:, 64:70] + ( refpose_coeff[:num_frames, :] - refpose_coeff[0:1, :] )
        return coeffs_pred_numpy
