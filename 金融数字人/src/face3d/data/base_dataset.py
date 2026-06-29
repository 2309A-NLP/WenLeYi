"""数据集抽象基类模块

该模块实现了一个数据集的抽象基类 (ABC) 'BaseDataset'，
所有自定义数据集都应继承此基类。

同时包含常用的图像变换函数（如 get_transform、get_affine_mat 等），
可在子类中复用。
"""
import random                       # 随机数生成模块，用于数据增强
import numpy as np                  # NumPy 数值计算库
import torch.utils.data as data     # PyTorch 数据工具模块
from PIL import Image               # PIL 图像处理库
import torchvision.transforms as transforms  # PyTorch 图像变换工具
from abc import ABC, abstractmethod  # 抽象基类支持


class BaseDataset(data.Dataset, ABC):
    """数据集抽象基类 (ABC)。

    所有自定义数据集都应继承此类，并实现以下四个函数：
    -- <__init__>:                      初始化类，首先调用 BaseDataset.__init__(self, opt)
    -- <__len__>:                       返回数据集大小
    -- <__getitem__>:                   获取一个数据点
    -- <modify_commandline_options>:    （可选）添加数据集特定的选项并设置默认选项
    """

    def __init__(self, opt):
        """初始化类；将选项保存到类中。

        参数:
            opt (Option类) -- 存储所有实验标志位；需要是 BaseOptions 的子类
        """
        self.opt = opt              # 保存实验选项
        # self.root = opt.dataroot  # 数据根目录（当前未使用）
        self.current_epoch = 0      # 记录当前训练轮次，用于数据增强等场景

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的数据集特定选项，并重写已有选项的默认值。

        参数:
            parser          -- 原始的命令行参数解析器
            is_train (bool) -- 是否为训练阶段。可据此添加训练/测试特定的选项

        返回:
            修改后的解析器
        """
        return parser  # 基类默认不修改任何选项

    @abstractmethod
    def __len__(self):
        """返回数据集中图像的总数。"""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """返回一个数据点及其元数据信息。

        参数:
            index (int) -- 用于数据索引的随机整数

        返回:
            包含数据及其名称的字典。通常包含数据本身及其元数据信息。
        """
        pass


def get_transform(grayscale=False):
    """获取图像预处理变换流水线。

    参数:
        grayscale (bool) -- 是否转换为灰度图像，默认为 False

    返回:
        组合后的图像变换对象
    """
    transform_list = []             # 变换列表
    if grayscale:
        # 如果需要灰度图，添加灰度转换（输出1通道）
        transform_list.append(transforms.Grayscale(1))
    # 添加 ToTensor 变换：将 PIL Image 或 numpy ndarray 转为 Tensor，并归一化到 [0, 1]
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)  # 将所有变换组合成流水线


def get_affine_mat(opt, size):
    """生成用于数据增强的仿射变换矩阵。

    根据预处理选项（平移、缩放、旋转、翻转）生成相应的仿射变换矩阵。
    变换流程：中心化 -> 翻转 -> 平移 -> 旋转 -> 缩放 -> 移回中心

    参数:
        opt       -- 实验选项对象，包含预处理参数
        size (tuple): 图像尺寸 (宽, 高)

    返回:
        affine (ndarray): 仿射变换矩阵 (3x3)
        affine_inv (ndarray): 逆仿射变换矩阵 (3x3)，用于图像变换
        flip (bool): 是否进行了水平翻转
    """
    # 初始化变换参数：无平移、无缩放、无旋转、不翻转
    shift_x, shift_y, scale, rot_angle, flip = 0., 0., 1., 0., False
    w, h = size  # 图像宽高

    # 根据预处理选项随机生成变换参数
    if 'shift' in opt.preprocess:
        # 随机平移：在 [-shift_pixs, +shift_pixs] 范围内随机选择
        shift_pixs = int(opt.shift_pixs)
        shift_x = random.randint(-shift_pixs, shift_pixs)
        shift_y = random.randint(-shift_pixs, shift_pixs)
    if 'scale' in opt.preprocess:
        # 随机缩放：缩放因子在 [1-scale_delta, 1+scale_delta] 范围内
        scale = 1 + opt.scale_delta * (2 * random.random() - 1)
    if 'rot' in opt.preprocess:
        # 随机旋转：旋转角度在 [-rot_angle, +rot_angle] 范围内
        rot_angle = opt.rot_angle * (2 * random.random() - 1)
        rot_rad = -rot_angle * np.pi / 180  # 转换为弧度
    if 'flip' in opt.preprocess:
        # 50%概率进行水平翻转
        flip = random.random() > 0.5

    # 构建仿射变换矩阵（3x3齐次坐标）
    # 步骤1：将图像中心平移到原点
    shift_to_origin = np.array([1, 0, -w // 2, 0, 1, -h // 2, 0, 0, 1]).reshape([3, 3])
    # 步骤2：水平翻转（x轴取反）
    flip_mat = np.array([-1 if flip else 1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape([3, 3])
    # 步骤3：随机平移
    shift_mat = np.array([1, 0, shift_x, 0, 1, shift_y, 0, 0, 1]).reshape([3, 3])
    # 步骤4：旋转
    rot_mat = np.array([np.cos(rot_rad), np.sin(rot_rad), 0, -np.sin(rot_rad), np.cos(rot_rad), 0, 0, 0, 1]).reshape([3, 3])
    # 步骤5：缩放
    scale_mat = np.array([scale, 0, 0, 0, scale, 0, 0, 0, 1]).reshape([3, 3])
    # 步骤6：将中心移回
    shift_to_center = np.array([1, 0, w // 2, 0, 1, h // 2, 0, 0, 1]).reshape([3, 3])
    
    # 组合所有变换（注意矩阵乘法顺序与操作顺序相反）
    affine = shift_to_center @ scale_mat @ rot_mat @ shift_mat @ flip_mat @ shift_to_origin
    # 计算逆矩阵，用于对图像进行仿射变换
    affine_inv = np.linalg.inv(affine)
    return affine, affine_inv, flip


def apply_img_affine(img, affine_inv, method=Image.BICUBIC):
    """对图像应用仿射变换。

    参数:
        img (PIL.Image): 原始图像
        affine_inv (ndarray): 逆仿射变换矩阵 (3x3)
        method: 重采样方法，默认为双三次插值 (BICUBIC)

    返回:
        变换后的 PIL Image
    """
    # 使用 PIL 的 transform 方法应用仿射变换，取矩阵的前6个元素
    return img.transform(img.size, Image.AFFINE, data=affine_inv.flatten()[:6], resample=Image.BICUBIC)


def apply_lm_affine(landmark, affine, flip, size):
    """对关键点（landmark）应用仿射变换。

    关键点变换与图像变换需要保持一致，
    如果图像进行了水平翻转，关键点的对应关系也需要调整。

    参数:
        landmark (ndarray): 关键点坐标 (68, 2)
        affine (ndarray): 仿射变换矩阵 (3x3)
        flip (bool): 是否进行了水平翻转
        size (tuple): 图像尺寸 (宽, 高)

    返回:
        变换后的关键点坐标 (68, 2)
    """
    _, h = size       # 获取图像高度
    lm = landmark.copy()  # 复制以避免修改原始数据

    # 将关键点坐标转换为齐次坐标并进行翻转（y轴镜像）
    lm[:, 1] = h - 1 - lm[:, 1]
    lm = np.concatenate((lm, np.ones([lm.shape[0], 1])), -1)  # 添加齐次坐标列

    # 应用仿射变换
    lm = lm @ np.transpose(affine)
    # 从齐次坐标转换回笛卡尔坐标
    lm[:, :2] = lm[:, :2] / lm[:, 2:]
    lm = lm[:, :2]
    # 再次翻转y轴
    lm[:, 1] = h - 1 - lm[:, 1]

    # 如果进行了水平翻转，需要调整68个关键点的左右对应关系
    if flip:
        lm_ = lm.copy()
        # 下巴轮廓 (0-16)：左右镜像
        lm_[:17] = lm[16::-1]
        # 左眉毛 (17-21) <-> 右眉毛 (22-26)
        lm_[17:22] = lm[26:21:-1]
        lm_[22:27] = lm[21:16:-1]
        # 鼻子 (31-35)：左右镜像
        lm_[31:36] = lm[35:30:-1]
        # 左眼 (36-41) <-> 右眼 (42-47)
        lm_[36:40] = lm[45:41:-1]
        lm_[40:42] = lm[47:45:-1]
        lm_[42:46] = lm[39:35:-1]
        lm_[46:48] = lm[41:39:-1]
        # 外嘴唇 (48-54) <-> 内嘴唇 (55-59)
        lm_[48:55] = lm[54:47:-1]
        lm_[55:60] = lm[59:54:-1]
        # 内嘴唇 (60-64) <-> 外嘴唇 (65-67)
        lm_[60:65] = lm[64:59:-1]
        lm_[65:68] = lm[67:64:-1]
        lm = lm_
    return lm
