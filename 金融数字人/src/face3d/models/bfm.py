"""Deep3DFaceRecon_pytorch的参数化3D人脸模型定义脚本

该脚本定义了基于BFM（Basel Face Model）的参数化3D人脸模型，
支持面部形状、纹理、法向量、光照等的计算，以及透视投影和关键点提取。
"""

import numpy as np
import  torch
import torch.nn.functional as F
from scipy.io import loadmat
from src.face3d.util.load_mats import transferBFM09
import os

def perspective_projection(focal, center):
    """计算透视投影矩阵
    
    Args:
        focal: 相机焦距
        center: 图像中心点坐标
    
    Returns:
        3x3的透视投影矩阵（转置后）
    """
    # return p.T (N, 3) @ (3, 3) 
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()

class SH:
    """球谐函数（Spherical Harmonics）类
    
    定义了用于光照计算的球谐函数系数。
    用于模拟面部在不同光照条件下的颜色变化。
    """
    def __init__(self):
        """初始化球谐函数系数"""
        # 球谐函数的系数a
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        # 球谐函数的系数c
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]



class ParametricFaceModel:
    """参数化面部模型类
    
    基于BFM（Basel Face Model）实现的参数化3D人脸模型。
    支持从系数向量生成3D人脸形状、纹理、法向量和颜色。
    """
    
    def __init__(self, 
                bfm_folder='./BFM', 
                recenter=True,
                camera_distance=10.,
                init_lit=np.array([
                    0.8, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                focal=1015.,
                center=112.,
                is_train=True,
                default_name='BFM_model_front.mat'):
        """初始化参数化面部模型
        
        Args:
            bfm_folder: BFM模型文件夹路径
            recenter: 是否重新中心化面部模型
            camera_distance: 相机到面部的距离
            init_lit: 初始光照参数
            focal: 相机焦距
            center: 图像中心点
            is_train: 是否为训练模式
            default_name: BFM模型文件名
        """
        
        # 如果BFM模型文件不存在，进行转换
        if not os.path.isfile(os.path.join(bfm_folder, default_name)):
            transferBFM09(bfm_folder)
            
        # 加载BFM模型文件
        model = loadmat(os.path.join(bfm_folder, default_name))
        # 平均面部形状 [3*N, 1]，N为顶点数
        self.mean_shape = model['meanshape'].astype(np.float32)
        # 身份基 [3*N, 80]，用于表示不同人的面部特征
        self.id_base = model['idBase'].astype(np.float32)
        # 表情基 [3*N, 64]，用于表示不同的面部表情
        self.exp_base = model['exBase'].astype(np.float32)
        # 平均面部纹理 [3*N, 1]（0-255）
        self.mean_tex = model['meantex'].astype(np.float32)
        # 纹理基 [3*N, 80]，用于表示不同的皮肤颜色
        self.tex_base = model['texBase'].astype(np.float32)
        # 顶点所属的面片索引 [N, 8]，从0开始
        self.point_buf = model['point_buf'].astype(np.int64) - 1
        # 每个面片的顶点索引 [F, 3]，从0开始
        self.face_buf = model['tri'].astype(np.int64) - 1
        # 68个关键点对应的顶点索引 [68, 1]，从0开始
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1

        if is_train:
            # 训练模式专用的面部区域掩码
            # 小面部区域的顶点索引，用于计算光照误差
            self.front_mask = np.squeeze(model['frontmask2_idx']).astype(np.int64) - 1
            # 小面部区域的面片索引 [f, 3]
            self.front_face_buf = model['tri_mask2'].astype(np.int64) - 1
            # 预定义的皮肤区域索引，用于计算反射率损失
            self.skin_mask = np.squeeze(model['skinmask'])
        
        if recenter:
            # 重新中心化平均形状，使其以原点为中心
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

        # 计算透视投影矩阵
        self.persc_proj = perspective_projection(focal, center)
        self.device = 'cpu'
        self.camera_distance = camera_distance
        # 初始化球谐函数
        self.SH = SH()
        # 初始化光照参数
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)
        

    def to(self, device):
        """将模型移动到指定设备
        
        Args:
            device: 目标设备（如'cuda:0'或'cpu'）
        """
        self.device = device
        # 遍历所有属性，将numpy数组转换为torch张量并移动到目标设备
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))

    
    def compute_shape(self, id_coeff, exp_coeff):
        """根据身份系数和表情系数计算3D面部形状
        
        Args:
            id_coeff: 身份系数，torch.tensor，形状 (B, 80)
            exp_coeff: 表情系数，torch.tensor，形状 (B, 64)
        
        Returns:
            face_shape: 面部形状，torch.tensor，形状 (B, N, 3)
        """
        batch_size = id_coeff.shape[0]
        # 计算身份部分：id_base @ id_coeff
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        # 计算表情部分：exp_base @ exp_coeff
        exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        # 最终形状 = 身份部分 + 表情部分 + 平均形状
        face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])
    

    def compute_texture(self, tex_coeff, normalize=True):
        """根据纹理系数计算面部纹理
        
        Args:
            tex_coeff: 纹理系数，torch.tensor，形状 (B, 80)
            normalize: 是否归一化到0-1范围
        
        Returns:
            face_texture: 面部纹理，torch.tensor，形状 (B, N, 3)，RGB顺序
        """
        batch_size = tex_coeff.shape[0]
        # 计算纹理：tex_base @ tex_coeff + mean_tex
        face_texture = torch.einsum('ij,aj->ai', self.tex_base, tex_coeff) + self.mean_tex
        if normalize:
            # 归一化到0-1范围
            face_texture = face_texture / 255.
        return face_texture.reshape([batch_size, -1, 3])


    def compute_norm(self, face_shape):
        """计算面部顶点的法向量
        
        Args:
            face_shape: 面部形状，torch.tensor，形状 (B, N, 3)
        
        Returns:
            vertex_norm: 顶点法向量，torch.tensor，形状 (B, N, 3)
        """
        # 获取三角形的三个顶点
        v1 = face_shape[:, self.face_buf[:, 0]]
        v2 = face_shape[:, self.face_buf[:, 1]]
        v3 = face_shape[:, self.face_buf[:, 2]]
        # 计算两条边向量
        e1 = v1 - v2
        e2 = v2 - v3
        # 通过叉积计算面法向量
        face_norm = torch.cross(e1, e2, dim=-1)
        # 归一化面法向量
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        # 补充零向量以匹配顶点数
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
        
        # 将面法向量聚合到顶点（通过point_buf索引）
        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        # 归一化顶点法向量
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm


    def compute_color(self, face_texture, face_norm, gamma):
        """根据纹理、法向量和光照参数计算面部颜色
        
        使用球谐函数（SH）模拟光照效果
        
        Args:
            face_texture: 面部纹理，torch.tensor，形状 (B, N, 3)
            face_norm: 旋转后的面部法向量，torch.tensor，形状 (B, N, 3)
            gamma: 光照参数（球谐系数），torch.tensor，形状 (B, 27)
        
        Returns:
            face_color: 面部颜色，torch.tensor，形状 (B, N, 3)，RGB顺序，范围0-1
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        # 将gamma重塑为 (B, 3, 9)，每个颜色通道9个球谐系数
        gamma = gamma.reshape([batch_size, 3, 9])
        # 加上初始光照参数
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        # 计算球谐基函数值（9个基函数）
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),  # Y00
            -a[1] * c[1] * face_norm[..., 1:2],  # Y1-1
             a[1] * c[1] * face_norm[..., 2:],  # Y10
            -a[1] * c[1] * face_norm[..., :1],  # Y11
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],  # Y2-2
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],  # Y2-1
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),  # Y20
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],  # Y21
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)  # Y22
        ], dim=-1)
        # 计算RGB三个通道的颜色
        r = Y @ gamma[..., :1]  # 红色通道
        g = Y @ gamma[..., 1:2]  # 绿色通道
        b = Y @ gamma[..., 2:]  # 蓝色通道
        # 组合RGB通道并与纹理相乘
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color

    
    def compute_rotation(self, angles):
        """根据欧拉角计算旋转矩阵
        
        Args:
            angles: 欧拉角，torch.tensor，形状 (B, 3)，单位为弧度
        
        Returns:
            rot: 旋转矩阵，torch.tensor，形状 (B, 3, 3)
        """
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        # 绕X轴旋转矩阵
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        # 绕Y轴旋转矩阵
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        # 绕Z轴旋转矩阵
        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        # 组合旋转：先绕X轴，再绕Y轴，最后绕Z轴
        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)


    def to_camera(self, face_shape):
        """将面部形状从世界坐标系转换到相机坐标系
        
        Args:
            face_shape: 面部形状，torch.tensor，形状 (B, N, 3)
        
        Returns:
            face_shape: 相机坐标系下的面部形状
        """
        # 沿Z轴方向平移到相机位置
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """将3D面部形状投影到2D图像平面
        
        Args:
            face_shape: 3D面部形状，torch.tensor，形状 (B, N, 3)
        
        Returns:
            face_proj: 2D投影坐标，torch.tensor，形状 (B, N, 2)
        """
        # 应用透视投影矩阵
        face_proj = face_shape @ self.persc_proj
        # 透视除法得到2D坐标
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj


    def transform(self, face_shape, rot, trans):
        """对3D面部形状进行刚体变换（旋转和平移）
        
        Args:
            face_shape: 面部形状，torch.tensor，形状 (B, N, 3)
            rot: 旋转矩阵，torch.tensor，形状 (B, 3, 3)
            trans: 平移向量，torch.tensor，形状 (B, 3)
        
        Returns:
            face_shape: 变换后的面部形状，torch.tensor，形状 (B, N, 3)
        """
        # 旋转 + 平移
        return face_shape @ rot + trans.unsqueeze(1)


    def get_landmarks(self, face_proj):
        """从投影坐标中提取68个面部关键点
        
        Args:
            face_proj: 投影后的坐标，torch.tensor，形状 (B, N, 2)
        
        Returns:
            face_lms: 面部关键点，torch.tensor，形状 (B, 68, 2)
        """  
        # 使用keypoints索引提取关键点
        return face_proj[:, self.keypoints]

    def split_coeff(self, coeffs):
        """将系数向量分割为各个组成部分
        
        Args:
            coeffs: 系数向量，torch.tensor，形状 (B, 256)
        
        Returns:
            coeffs_dict: 包含各个系数组件的字典
                - 'id': 身份系数 [0:80]
                - 'exp': 表情系数 [80:144]
                - 'tex': 纹理系数 [144:224]
                - 'angle': 旋转角度 [224:227]
                - 'gamma': 光照参数 [227:254]
                - 'trans': 平移向量 [254:]
        """
        id_coeffs = coeffs[:, :80]  # 身份系数：80维
        exp_coeffs = coeffs[:, 80: 144]  # 表情系数：64维
        tex_coeffs = coeffs[:, 144: 224]  # 纹理系数：80维
        angles = coeffs[:, 224: 227]  # 旋转角度：3维
        gammas = coeffs[:, 227: 254]  # 光照参数：27维
        translations = coeffs[:, 254:]  # 平移向量：2维
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

    def compute_for_render(self, coeffs):
        """根据系数计算渲染所需的全部数据
        
        Args:
            coeffs: 系数向量，torch.tensor，形状 (B, 257)
        
        Returns:
            face_vertex: 面部顶点坐标（相机坐标系），torch.tensor，形状 (B, N, 3)
            face_texture: 面部纹理，torch.tensor，形状 (B, N, 3)
            face_color: 面部颜色（RGB），torch.tensor，形状 (B, N, 3)
            landmark: 面部关键点，torch.tensor，形状 (B, 68, 2)
        """
        # 分割系数
        coef_dict = self.split_coeff(coeffs)
        # 计算面部形状
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        # 计算旋转矩阵
        rotation = self.compute_rotation(coef_dict['angle'])

        # 应用刚体变换
        face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])
        # 转换到相机坐标系
        face_vertex = self.to_camera(face_shape_transformed)
        
        # 投影到2D图像平面
        face_proj = self.to_image(face_vertex)
        # 提取关键点
        landmark = self.get_landmarks(face_proj)

        # 计算纹理
        face_texture = self.compute_texture(coef_dict['tex'])
        # 计算法向量
        face_norm = self.compute_norm(face_shape)
        # 旋转法向量
        face_norm_roted = face_norm @ rotation
        # 计算颜色（考虑光照）
        face_color = self.compute_color(face_texture, face_norm_roted, coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark

    def compute_for_render_woRotation(self, coeffs):
        """根据系数计算渲染数据（不应用旋转）
        
        与compute_for_render相同，但不应用旋转矩阵。
        
        Args:
            coeffs: 系数向量，torch.tensor，形状 (B, 257)
        
        Returns:
            face_vertex: 面部顶点坐标，torch.tensor，形状 (B, N, 3)
            face_texture: 面部纹理，torch.tensor，形状 (B, N, 3)
            face_color: 面部颜色，torch.tensor，形状 (B, N, 3)
            landmark: 面部关键点，torch.tensor，形状 (B, 68, 2)
        """
        # 分割系数
        coef_dict = self.split_coeff(coeffs)
        # 计算面部形状
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        #rotation = self.compute_rotation(coef_dict['angle'])

        # 不应用旋转和平移
        #face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])
        face_vertex = self.to_camera(face_shape)
        
        # 投影到2D图像平面
        face_proj = self.to_image(face_vertex)
        # 提取关键点
        landmark = self.get_landmarks(face_proj)

        # 计算纹理
        face_texture = self.compute_texture(coef_dict['tex'])
        # 计算法向量
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm                                    # @ rotation
        # 计算颜色（不考虑旋转）
        face_color = self.compute_color(face_texture, face_norm_roted, coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark


if __name__ == '__main__':
    # 执行BFM模型转换
    transferBFM09()
