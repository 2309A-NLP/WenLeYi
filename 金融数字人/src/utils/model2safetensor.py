"""
模型格式转换模块
本模块用于将SadTalker的PyTorch模型检查点（.pth格式）转换为safetensors格式。
safetensors是一种更安全、更高效的模型存储格式，可以避免pickle反序列化带来的安全风险。

主要功能：
1. 加载各个子模型（3D人脸重建、人脸渲染、音频到姿态、音频到表情）
2. 将所有模型参数整合到一个统一的SadTalker模块中
3. 导出为safetensors格式
"""
import torch
import yaml
import os

import safetensors
from safetensors.torch import save_file
from yacs.config import CfgNode as CN
import sys

sys.path.append('/apdcephfs/private_shadowcun/SadTalker')

# 导入3D人脸重建网络
from src.face3d.models import networks

# 导入人脸渲染相关模块
from src.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from src.facerender.modules.mapping import MappingNet
from src.facerender.modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator

# 导入音频相关模型
from src.audio2pose_models.audio2pose import Audio2Pose
from src.audio2exp_models.networks import SimpleWrapperV2 
from src.test_audio2coeff import load_cpk

# ========== 基本配置参数 ==========
size = 256  # 输出图像尺寸
############ face vid2vid 配置
config_path = os.path.join('src', 'config', 'facerender.yaml')  # 人脸渲染配置文件路径
current_root_path = '.'  # 项目根目录

# ========== 加载3D人脸重建模型 ==========
path_of_net_recon_model = os.path.join(current_root_path, 'checkpoints', 'epoch_20.pth')  # 3D重建模型路径
net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='')  # 定义ResNet50用于3D人脸重建
checkpoint = torch.load(path_of_net_recon_model, map_location='cpu')    # 加载模型权重（在CPU上）
net_recon.load_state_dict(checkpoint['net_recon'])  # 加载重建网络的参数

# ========== 加载人脸渲染配置 ==========
with open(config_path) as f:
    config = yaml.safe_load(f)  # 读取YAML配置文件

# 初始化各个子模型
generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])  # 遮挡感知的SPADE生成器
kp_extractor = KPDetector(**config['model_params']['kp_detector_params'],
                            **config['model_params']['common_params'])  # 关键点检测器
he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                        **config['model_params']['common_params'])  # 头部姿态估计器
mapping = MappingNet(**config['model_params']['mapping_params'])  # 映射网络

def load_cpk_facevid2vid(checkpoint_path, generator=None, discriminator=None, 
                        kp_detector=None, he_estimator=None, optimizer_generator=None, 
                        optimizer_discriminator=None, optimizer_kp_detector=None, 
                        optimizer_he_estimator=None, device="cpu"):
    """
    加载face vid2vid模型的检查点（旧格式，.pth文件）
    
    参数:
        checkpoint_path: 检查点文件路径
        generator: 生成器模型
        discriminator: 判别器模型（可选）
        kp_detector: 关键点检测器
        he_estimator: 头部姿态估计器
        optimizer_*: 各模型的优化器（可选）
        device: 计算设备
    返回:
        epoch: 检查点中的训练轮数
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if generator is not None:
        generator.load_state_dict(checkpoint['generator'])
    if kp_detector is not None:
        kp_detector.load_state_dict(checkpoint['kp_detector'])
    if he_estimator is not None:
        he_estimator.load_state_dict(checkpoint['he_estimator'])
    if discriminator is not None:
        try:
            discriminator.load_state_dict(checkpoint['discriminator'])
        except:
            print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
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


def load_cpk_facevid2vid_safetensor(checkpoint_path, generator=None, 
                        kp_detector=None, he_estimator=None,  
                        device="cpu"):
    """
    加载face vid2vid模型的检查点（新格式，safetensors文件）
    
    safetensors格式的键名通常带有模块前缀，需要去除前缀后再加载到对应模型中
    
    参数:
        checkpoint_path: safetensors检查点文件路径
        generator: 生成器模型
        kp_detector: 关键点检测器
        he_estimator: 头部姿态估计器
        device: 计算设备
    返回:
        None
    """
    checkpoint = safetensors.torch.load_file(checkpoint_path)  # 使用safetensors安全加载

    # 加载生成器权重（去除'generator.'前缀）
    if generator is not None:
        x_generator = {}
        for k,v in checkpoint.items():
            if 'generator' in k:
                x_generator[k.replace('generator.', '')] = v
        generator.load_state_dict(x_generator)
    # 加载关键点检测器权重（去除'kp_extractor.'前缀）
    if kp_detector is not None:
        x_generator = {}
        for k,v in checkpoint.items():
            if 'kp_extractor' in k:
                x_generator[k.replace('kp_extractor.', '')] = v
        kp_detector.load_state_dict(x_generator)
    # 加载头部姿态估计器权重（去除'he_estimator.'前缀）
    if he_estimator is not None:
        x_generator = {}
        for k,v in checkpoint.items():
            if 'he_estimator' in k:
                x_generator[k.replace('he_estimator.', '')] = v
        he_estimator.load_state_dict(x_generator)
    
    return None

# ========== 加载face vid2vid检查点 ==========
free_view_checkpoint = '/apdcephfs/private_shadowcun/SadTalker/checkpoints/facevid2vid_'+str(size)+'-model.pth.tar'
load_cpk_facevid2vid(free_view_checkpoint, kp_detector=kp_extractor, generator=generator, he_estimator=he_estimator)

# ========== 加载其他模型检查点路径 ==========
wav2lip_checkpoint = os.path.join(current_root_path, 'checkpoints', 'wav2lip.pth')  # Wav2Lip唇部同步模型

# 音频到姿态模型相关路径
audio2pose_checkpoint = os.path.join(current_root_path, 'checkpoints', 'auido2pose_00140-model.pth')
audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')

# 音频到表情模型相关路径
audio2exp_checkpoint = os.path.join(current_root_path, 'checkpoints', 'auido2exp_00300-model.pth')
audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

# ========== 加载音频到姿态模型 ==========
fcfg_pose = open(audio2pose_yaml_path)
cfg_pose = CN.load_cfg(fcfg_pose)  # 从YAML加载配置
cfg_pose.freeze()  # 冻结配置，防止修改
audio2pose_model = Audio2Pose(cfg_pose, wav2lip_checkpoint)  # 初始化Audio2Pose模型
audio2pose_model.eval()  # 设置为评估模式
load_cpk(audio2pose_checkpoint, model=audio2pose_model, device='cpu')  # 加载预训练权重

# ========== 加载音频到表情模型 ==========
# load audio2exp_model
netG = SimpleWrapperV2()  # 初始化音频到表情的网络
netG.eval()  # 设置为评估模式
load_cpk(audio2exp_checkpoint, model=netG, device='cpu')  # 加载预训练权重

class SadTalker(torch.nn.Module):
    """
    SadTalker主模型类
    将所有子模型（关键点提取器、生成器、音频到表情、音频到姿态、3D人脸重建）整合为一个统一的模型
    """
    def __init__(self, kp_extractor, generator, netG, audio2pose, face_3drecon):
        """初始化SadTalker模型"""
        super(SadTalker, self).__init__()
        self.kp_extractor = kp_extractor      # 关键点提取器
        self.generator = generator             # 生成器
        self.audio2exp = netG                  # 音频到表情模型
        self.audio2pose = audio2pose           # 音频到姿态模型
        self.face_3drecon = face_3drecon       # 3D人脸重建模型


# 创建SadTalker模型实例并保存为safetensors格式
model = SadTalker(kp_extractor, generator, netG, audio2pose_model, net_recon)

# 将模型状态字典保存为safetensors格式（更安全的模型存储格式）
save_file(model.state_dict(), "checkpoints/SadTalker_V0.0.2_"+str(size)+".safetensors")

### 测试：验证safetensors格式的模型可以正确加载
load_cpk_facevid2vid_safetensor('checkpoints/SadTalker_V0.0.2_'+str(size)+'.safetensors', kp_detector=kp_extractor, generator=generator, he_estimator=None)