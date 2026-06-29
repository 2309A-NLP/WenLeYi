"""基于文件列表的自定义数据集

该脚本定义了 Deep3DFaceRecon_pytorch 使用的自定义数据集，
通过文件列表（flist）来加载训练/验证图像、掩码和关键点数据。
"""
import os.path                       # 路径处理模块
from data.base_dataset import BaseDataset, get_transform, get_affine_mat, apply_img_affine, apply_lm_affine  # 基类和工具函数
from data.image_folder import make_dataset  # 从图像文件夹模块导入生成数据集列表的函数
from PIL import Image               # PIL 图像处理库
import random                       # 随机数模块，用于数据增强
import util.util as util            # 工具函数模块
import numpy as np                  # NumPy 数值计算库
import json                         # JSON 解析模块
import torch                        # PyTorch 深度学习框架
from scipy.io import loadmat, savemat  # MATLAB 文件读写
import pickle                       # Python 序列化模块
from util.preprocess import align_img, estimate_norm  # 图像预处理和法线估计
from util.load_mats import load_lm3d  # 加载3D关键点标准数据


def default_flist_reader(flist):
    """默认的文件列表读取器。

    读取文本格式的文件列表，每行包含一个文件路径。
    格式类似于 Caffe 的 filelist 格式。

    参数:
        flist (str): 文件列表路径

    返回:
        imlist (list): 图像路径列表
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()       # 去除首尾空白字符
            imlist.append(impath)
    return imlist


def jason_flist_reader(flist):
    """JSON 格式的文件列表读取器。

    从 JSON 文件中读取文件列表信息。

    参数:
        flist (str): JSON 文件路径

    返回:
        info: JSON 文件中的数据
    """
    with open(flist, 'r') as fp:
        info = json.load(fp)
    return info


def parse_label(label):
    """将标签数据转换为 PyTorch 浮点张量。

    参数:
        label: 标签数据（numpy 数组或其他可转换格式）

    返回:
        torch.Tensor: float32 类型的张量
    """
    return torch.tensor(np.array(label).astype(np.float32))


class FlistDataset(BaseDataset):
    """基于文件列表的数据集类。

    该数据集需要一个目录来存放训练图像 '/path/to/data/train'，
    可以通过参数 '--dataroot /path/to/data' 来指定数据路径。

    支持的文件结构：
    - 图像文件：/path/to/data/images/xxx.jpg
    - 掩码文件：/path/to/data/mask/xxx.jpg
    - 关键点文件：/path/to/data/landmarks/xxx.txt
    """

    def __init__(self, opt):
        """初始化数据集类。

        参数:
            opt (Option类) -- 存储所有实验标志位；需要是 BaseOptions 的子类
        """
        # 调用父类的初始化方法
        BaseDataset.__init__(self, opt)
        
        # 加载3D关键点标准数据，用于后续的图像对齐
        self.lm3d_std = load_lm3d(opt.bfm_folder)
        
        # 从文件列表中读取所有掩码文件名
        msk_names = default_flist_reader(opt.flist)
        # 将相对路径转换为绝对路径
        self.msk_paths = [os.path.join(opt.data_root, i) for i in msk_names]

        # 记录数据集大小
        self.size = len(self.msk_paths) 
        self.opt = opt
        
        # 设置数据集名称标识（用于区分训练/验证集）
        self.name = 'train' if opt.isTrain else 'val'
        # 如果文件列表名包含下划线，将额外信息附加到数据集名称中
        if '_' in opt.flist:
            self.name += '_' + opt.flist.split(os.sep)[-1].split('_')[0]
        

    def __getitem__(self, index):
        """返回一个数据点及其元数据信息。

        参数:
            index (int) -- 用于数据索引的随机整数

        返回:
            字典，包含以下字段：
            - imgs (tensor): 输入图像
            - lms (tensor): 对应的3D关键点坐标
            - msks (tensor): 对应的注意力掩码
            - M (tensor): 仿射变换矩阵
            - im_paths (str): 图像路径
            - aug_flag (bool): 标识是否经过数据增强
            - dataset (str): 数据集名称
        """
        # 确保索引在数据集范围内（循环取值）
        msk_path = self.msk_paths[index % self.size]
        # 根据掩码路径推导图像路径和关键点路径
        img_path = msk_path.replace('mask/', '')  # 掩码在 mask/ 子目录下
        # 关键点路径：从 mask/ 改为 landmarks/，后缀改为 .txt
        lm_path = '.'.join(msk_path.replace('mask', 'landmarks').split('.')[:-1]) + '.txt'

        # 加载原始图像（RGB模式）
        raw_img = Image.open(img_path).convert('RGB')
        # 加载掩码图像（RGB模式）
        raw_msk = Image.open(msk_path).convert('RGB')
        # 加载关键点坐标（文本格式，浮点数组）
        raw_lm = np.loadtxt(lm_path).astype(np.float32)

        # 使用标准3D关键点进行图像对齐
        # align_img 会将人脸图像对齐到标准位置
        _, img, lm, msk = align_img(raw_img, raw_lm, self.lm3d_std, raw_msk)
        
        # 判断是否需要数据增强（仅训练阶段且开启了增强标志）
        aug_flag = self.opt.use_aug and self.opt.isTrain
        if aug_flag:
            # 执行数据增强：随机仿射变换（平移、缩放、旋转、翻转）
            img, lm, msk = self._augmentation(img, lm, self.opt, msk)
        
        # 获取图像高度并估算归一化仿射矩阵 M
        _, H = img.size
        M = estimate_norm(lm, H)

        # 将图像和掩码转换为张量格式
        transform = get_transform()
        img_tensor = transform(img)         # 图像转为 [C, H, W] 张量
        msk_tensor = transform(msk)[:1, ...]  # 掩码取单通道
        # 将关键点和变换矩阵转换为张量
        lm_tensor = parse_label(lm)
        M_tensor = parse_label(M)

        # 返回数据字典
        return {'imgs': img_tensor, 
                'lms': lm_tensor, 
                'msks': msk_tensor, 
                'M': M_tensor,
                'im_paths': img_path, 
                'aug_flag': aug_flag,
                'dataset': self.name}

    def _augmentation(self, img, lm, opt, msk=None):
        """执行数据增强：对图像、关键点和掩码应用相同的仿射变换。

        参数:
            img (PIL.Image): 原始图像
            lm (ndarray): 关键点坐标
            opt: 实验选项对象
            msk (PIL.Image, optional): 掩码图像，默认为 None

        返回:
            img (PIL.Image): 增强后的图像
            lm (ndarray): 变换后的关键点
            msk (PIL.Image or None): 变换后的掩码
        """
        # 生成仿射变换矩阵
        affine, affine_inv, flip = get_affine_mat(opt, img.size)
        # 对图像应用逆仿射变换
        img = apply_img_affine(img, affine_inv)
        # 对关键点应用正向仿射变换（同时处理翻转时的左右映射）
        lm = apply_lm_affine(lm, affine, flip, img.size)
        # 如果提供了掩码，也对其应用相同的仿射变换
        if msk is not None:
            msk = apply_img_affine(msk, affine_inv, method=Image.BILINEAR)
        return img, lm, msk
    

    def __len__(self):
        """返回数据集中图像的总数。"""
        return self.size
