"""
make_animation.py - 动画生成核心模块
该模块实现了从源面部图像生成驱动动画的核心逻辑。
主要包含：
1. normalize_kp：关键点归一化函数，用于调整关键点运动的尺度和方向
2. headpose_pred_to_degree：将分类预测转换为角度值
3. get_rotation_matrix：根据欧拉角生成 3D 旋转矩阵
4. keypoint_transformation：对关键点应用旋转、平移和表情变换
5. make_animation：主动画生成函数，逐帧合成面部动画
6. AnimateModel：用于多 GPU 训练的封装模型
"""
from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    """
    关键点归一化函数。
    
    对驱动面部的关键点进行归一化处理，使运动能够正确地迁移到源面部。
    
    主要功能：
    1. 自适应运动尺度：根据源和驱动面部的大小差异调整运动幅度
    2. 相对运动：使用相对于初始帧的位移，而非绝对坐标
    3. 相对雅可比：使用相对于初始帧的雅可比变换
    
    参数:
        kp_source (dict): 源面部关键点
        kp_driving (dict): 驱动面部关键点
        kp_driving_initial (dict): 驱动面部初始帧的关键点
        adapt_movement_scale (bool): 是否自适应调整运动尺度
        use_relative_movement (bool): 是否使用相对运动
        use_relative_jacobian (bool): 是否使用相对雅可比
    
    返回:
        dict: 归一化后的驱动关键点
    """
    # 自适应运动尺度：通过凸包面积比计算缩放因子
    if adapt_movement_scale:
        # 计算源面部关键点的凸包面积
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        # 计算驱动面部初始帧的凸包面积
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        # 缩放因子 = 源面积的平方根 / 驱动面积的平方根
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    # 复制驱动关键点字典
    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        # 计算相对于初始帧的位移
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        # 应用自适应缩放
        kp_value_diff *= adapt_movement_scale
        # 将相对位移加到源面部关键点上
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            # 计算相对雅可比变换
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            # 应用到源面部雅可比
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def headpose_pred_to_degree(pred):
    """
    将头部姿态的分类预测转换为角度值。
    
    使用软化 argmax（期望值）从 66 个 bin 的分类分布中提取角度。
    角度范围约为 [-99, 99] 度（每个 bin 宽 3 度）。
    
    参数:
        pred (Tensor): 分类预测，形状 (bs, 66)
    
    返回:
        Tensor: 角度值（度），形状 (bs,)
    """
    device = pred.device
    # 创建索引张量 [0, 1, 2, ..., 65]
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    # softmax 归一化为概率分布
    pred = F.softmax(pred)
    # 计算期望值：sum(probability * index) * 3 - 99
    # 这将 [0, 65] 的索引映射到 [-99, 96] 度的范围
    degree = torch.sum(pred*idx_tensor, 1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    """
    根据欧拉角（yaw, pitch, roll）生成 3D 旋转矩阵。
    
    分别构建绕 X 轴（pitch）、Y 轴（yaw）、Z 轴（roll）的旋转矩阵，
    然后将它们组合为最终的旋转矩阵。
    
    参数:
        yaw (Tensor): 偏航角（度），形状 (bs,)
        pitch (Tensor): 俯仰角（度），形状 (bs,)
        roll (Tensor): 翻滚角（度），形状 (bs,)
    
    返回:
        Tensor: 旋转矩阵，形状 (bs, 3, 3)
    """
    # 将角度从度转换为弧度
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    # 扩展维度以便构建矩阵
    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    # 构建绕 X 轴（俯仰）的旋转矩阵
    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    # 构建绕 Y 轴（偏航）的旋转矩阵
    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    # 构建绕 Z 轴（翻滚）的旋转矩阵
    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    # 组合旋转矩阵：pitch * yaw * roll
    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, wo_exp=False):
    """
    对规范关键点应用头部姿态和表情变换。
    
    变换步骤：
    1. 使用欧拉角生成旋转矩阵
    2. 对关键点进行旋转
    3. 应用平移
    4. 叠加表情偏移
    
    参数:
        kp_canonical (dict): 规范关键点（'value': (bs, k, 3)）
        he (dict): 头部姿态和表情参数，包含 yaw, pitch, roll, t, exp
        wo_exp (bool): 是否忽略表情参数（置零）
    
    返回:
        dict: 变换后的关键点 ('value': (bs, k, 3))
    """
    kp = kp_canonical['value']    # (bs, k, 3) 
    yaw, pitch, roll= he['yaw'], he['pitch'], he['roll']      
    # 将分类预测转换为角度值
    yaw = headpose_pred_to_degree(yaw) 
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    # 如果提供了直接的角度输入（训练时可能使用），则覆盖
    if 'yaw_in' in he:
        yaw = he['yaw_in']
    if 'pitch_in' in he:
        pitch = he['pitch_in']
    if 'roll_in' in he:
        roll = he['roll_in']

    # 生成旋转矩阵
    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    # 如果不需要表情，将表情参数置零
    if wo_exp:
        exp =  exp*0  
    
    # 步骤1：对关键点应用旋转（通过矩阵乘法）
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # 步骤2：应用平移（只使用 Y 方向的平移，X 和 Z 方向置零）
    t[:, 0] = t[:, 0]*0
    t[:, 2] = t[:, 2]*0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # 步骤3：叠加表情偏移
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {'value': kp_transformed}



def make_animation(source_image, source_semantics, target_semantics,
                            generator, kp_detector, he_estimator, mapping, 
                            yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                            use_exp=True, use_half=False):
    """
    主动画生成函数：根据源图像和目标语义系数生成面部动画。
    
    核心流程：
    1. 从源图像检测关键点和头部姿态
    2. 逐帧处理目标语义系数，生成驱动关键点
    3. 使用生成器合成每一帧的面部图像
    
    参数:
        source_image (Tensor): 源面部图像 (bs, C, H, W)
        source_semantics (Tensor): 源语义特征（3DMM系数）
        target_semantics (Tensor): 目标语义特征序列 (bs, num_frames, feature_dim)
        generator: 图像生成器
        kp_detector: 关键点检测器
        he_estimator: 头部姿态估计器（此处未直接使用，通过 mapping 代替）
        mapping: 映射网络（将语义特征映射为头部姿态参数）
        yaw_c_seq (Tensor): 外部提供的偏航角序列（可选）
        pitch_c_seq (Tensor): 外部提供的俯仰角序列（可选）
        roll_c_seq (Tensor): 外部提供的翻滚角序列（可选）
        use_exp (bool): 是否使用表情参数
        use_half (bool): 是否使用半精度浮点（未使用）
    
    返回:
        Tensor: 生成的动画帧序列 (bs, num_frames, C, H, W)
    """
    with torch.no_grad():  # 推理时不需要梯度
        predictions = []

        # 从源图像检测规范关键点
        kp_canonical = kp_detector(source_image)
        # 通过映射网络将源语义特征转换为头部姿态参数
        he_source = mapping(source_semantics)
        # 对源关键点应用头部姿态变换
        kp_source = keypoint_transformation(kp_canonical, he_source)
    
        # 逐帧处理目标语义系数
        for frame_idx in tqdm(range(target_semantics.shape[1]), 'Face Renderer:'):
            # still check the dimension
            # print(target_semantics.shape, source_semantics.shape)
            target_semantics_frame = target_semantics[:, frame_idx]
            # 将目标语义特征映射为头部姿态参数
            he_driving = mapping(target_semantics_frame)
            # 如果提供了外部角度序列，使用外部角度
            if yaw_c_seq is not None:
                he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
            if pitch_c_seq is not None:
                he_driving['pitch_in'] = pitch_c_seq[:, frame_idx] 
            if roll_c_seq is not None:
                he_driving['roll_in'] = roll_c_seq[:, frame_idx] 
            
            # 对规范关键点应用驱动面部的头部姿态变换
            kp_driving = keypoint_transformation(kp_canonical, he_driving)
            
            kp_norm = kp_driving
            # 使用生成器合成新帧
            out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
            '''
            source_image_new = out['prediction'].squeeze(1)
            kp_canonical_new =  kp_detector(source_image_new)
            he_source_new = he_estimator(source_image_new) 
            kp_source_new = keypoint_transformation(kp_canonical_new, he_source_new, wo_exp=True)
            kp_driving_new = keypoint_transformation(kp_canonical_new, he_driving, wo_exp=True)
            out = generator(source_image_new, kp_source=kp_source_new, kp_driving=kp_driving_new)
            '''
            predictions.append(out['prediction'])
        # 将所有帧堆叠为一个张量
        predictions_ts = torch.stack(predictions, dim=1)
    return predictions_ts

class AnimateModel(torch.nn.Module):
    """
    动画生成模型的封装类。
    
    将生成器、关键点检测器和映射网络合并为一个模型，
    便于多 GPU 训练时的模型并行化。
    
    在训练时，该模型可以作为整体参与梯度计算和更新。
    """

    def __init__(self, generator, kp_extractor, mapping):
        """
        参数:
            generator: 图像生成器
            kp_extractor: 关键点检测器
            mapping: 映射网络
        """
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        # 将所有子模型设置为评估模式
        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        """
        前向传播：完整的动画生成流程。
        
        参数:
            x (dict): 包含以下键值对：
                - 'source_image': 源图像
                - 'source_semantics': 源语义特征
                - 'target_semantics': 目标语义特征
                - 'yaw_c_seq': 偏航角序列
                - 'pitch_c_seq': 俯仰角序列
                - 'roll_c_seq': 翻滚角序列
        
        返回:
            Tensor: 生成的动画帧序列
        """
        source_image = x['source_image']
        source_semantics = x['source_semantics']
        target_semantics = x['target_semantics']
        yaw_c_seq = x['yaw_c_seq']
        pitch_c_seq = x['pitch_c_seq']
        roll_c_seq = x['roll_c_seq']

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor,
                                        self.mapping, use_exp = True,
                                        yaw_c_seq=yaw_c_seq, pitch_c_seq=pitch_c_seq, roll_c_seq=roll_c_seq)
        
        return predictions_video