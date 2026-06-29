# init_path.py - 模型路径初始化模块
# 本模块负责配置SadTalker项目所需的所有模型文件和配置文件路径
# 包括3D人脸重建模型、音频到表情/姿态映射模型、面部渲染器等

import os


def init_path(checkpoint_dir, config_dir, size=512, old_version=False, preprocess='crop'):
    """初始化SadTalker模型的文件路径
    
    根据模型目录、配置目录、输出尺寸和预处理模式，
    构建所有需要的模型文件和配置文件的路径字典。
    
    Args:
        checkpoint_dir: 模型权重文件目录（包含预训练模型）
        config_dir: 配置文件目录（包含YAML配置文件）
        size: 输出图片尺寸（默认512）
        old_version: 是否使用旧版本（当前未使用）
        preprocess: 预处理模式，'crop'或'full'
    
    Returns:
        字典，包含所有模型和配置文件的路径
    """
    print('using safetensor as default')  # 提示默认使用safetensor格式
    # 构建基础路径字典
    sadtalker_paths = {
        # 主模型权重文件（SadTalker V0.0.2，根据尺寸选择对应版本）
        "checkpoint":os.path.join(checkpoint_dir, 'SadTalker_V0.0.2_'+str(size)+'.safetensors'),
        }
    use_safetensor = True  # 标记使用safetensor格式

    # 设置3D人脸模型拟合数据目录
    sadtalker_paths['dir_of_BFM_fitting'] = os.path.join(config_dir) # , 'BFM_Fitting'
    # 音频到姿态映射的YAML配置文件
    sadtalker_paths['audio2pose_yaml_path'] = os.path.join(config_dir, 'auido2pose.yaml')
    # 音频到表情映射的YAML配置文件
    sadtalker_paths['audio2exp_yaml_path'] = os.path.join(config_dir, 'auido2exp.yaml')
    # PI渲染器的YAML配置文件（用于面部渲染）
    sadtalker_paths['pirender_yaml_path'] = os.path.join(config_dir, 'facerender_pirender.yaml')
    # PI渲染器的模型权重文件
    sadtalker_paths['pirender_checkpoint'] =  os.path.join(checkpoint_dir, 'epoch_00190_iteration_000400000_checkpoint.pt')
    # 标记是否使用safetensor格式
    sadtalker_paths['use_safetensor'] =  use_safetensor # os.path.join(config_dir, 'auido2exp.yaml')
    
    # 根据预处理模式选择不同的映射网络和渲染器配置
    if 'full' in preprocess:
        # full模式：使用完整的面部渲染配置
        sadtalker_paths['mappingnet_checkpoint'] = os.path.join(checkpoint_dir, 'mapping_00109-model.pth.tar')  # 映射网络权重
        sadtalker_paths['facerender_yaml'] = os.path.join(config_dir, 'facerender_still.yaml')  # 静态面部渲染配置
    else:
        # crop模式（默认）：使用裁剪后的面部渲染配置
        sadtalker_paths['mappingnet_checkpoint'] = os.path.join(checkpoint_dir, 'mapping_00229-model.pth.tar')  # 映射网络权重
        sadtalker_paths['facerender_yaml'] = os.path.join(config_dir, 'facerender.yaml')  # 标准面部渲染配置

    # 以下为旧版本配置（已注释）
    # sadtalker_paths['mappingnet_checkpoint'] = os.path.join(checkpoint_dir, 'mapping_00229-model.pth.tar')
    # sadtalker_paths['facerender_yaml'] = os.path.join(config_dir, 'facerender.yaml')

    return sadtalker_paths
