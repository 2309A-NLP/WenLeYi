"""Deep3DFaceRecon_pytorch的基础工具函数脚本

该脚本包含用于3D面部重建的各种实用工具函数，
包括类型转换、图像处理、目录创建等功能。
"""

from __future__ import print_function
import numpy as np
import torch
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision


def str2bool(v):
    """将字符串转换为布尔值
    
    用于命令行参数解析，支持多种布尔值表示方式。
    
    Args:
        v: 输入字符串
    
    Returns:
        bool: 布尔值
    
    Raises:
        ArgumentTypeError: 如果输入不是有效的布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    """复制配置对象并更新指定属性
    
    Args:
        default_opt: 默认配置对象
        **kwargs: 要更新的属性键值对
    
    Returns:
        conf: 更新后的配置对象
    """
    # 创建配置对象的副本
    conf = Namespace(**vars(default_opt))
    # 更新指定属性
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf

def genvalconf(train_opt, **kwargs):
    """从训练配置生成验证配置
    
    将训练配置中的val_xxx属性映射到xxx属性。
    
    Args:
        train_opt: 训练配置对象
        **kwargs: 要更新的属性键值对
    
    Returns:
        conf: 验证配置对象
    """
    # 创建配置对象的副本
    conf = Namespace(**vars(train_opt))
    attr_dict = train_opt.__dict__
    # 将val_xxx属性映射到xxx属性
    for key, value in attr_dict.items():
        if 'val' in key and key.split('_')[0] in attr_dict:
            setattr(conf, key.split('_')[0], value)

    # 更新指定属性
    for key in kwargs:
        setattr(conf, key, kwargs[key])

    return conf
        
def find_class_in_module(target_cls_name, module):
    """在模块中查找指定名称的类
    
    Args:
        target_cls_name: 目标类名称
        module: 模块名称
    
    Returns:
        cls: 找到的类
    
    Raises:
        AssertionError: 如果未找到匹配的类
    """
    # 标准化类名称（移除下划线，转小写）
    target_cls_name = target_cls_name.replace('_', '').lower()
    # 动态导入模块
    clslib = importlib.import_module(module)
    cls = None
    # 遍历模块中的所有类，查找匹配的类
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8):
    """将Tensor数组转换为numpy图像数组
    
    Args:
        input_image (tensor): 输入图像张量数组，范围(0, 1)
        imtype (type): 转换后的numpy数组类型
    
    Returns:
        numpy数组形式的图像
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # 从变量中获取数据
            image_tensor = input_image.data
        else:
            return input_image
        # 转换为numpy数组，并将值限制在[0, 1]范围内
        image_numpy = image_tensor.clamp(0.0, 1.0).cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # 灰度图转RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # 后处理：转置维度并缩放到[0, 255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    else:  # 如果已经是numpy数组，直接返回
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """诊断网络梯度
    
    计算并打印网络参数梯度的平均绝对值。
    用于检测梯度消失或梯度爆炸问题。
    
    Args:
        net (torch network): PyTorch网络
        name (str): 网络名称
    """
    mean = 0.0
    count = 0
    # 遍历所有参数，计算梯度的平均绝对值
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """保存numpy图像到磁盘
    
    Args:
        image_numpy (numpy array): 输入numpy数组
        image_path (str): 图像保存路径
        aspect_ratio (float): 宽高比调整，默认1.0（不调整）
    """
    # 将numpy数组转换为PIL图像
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        # 如果宽高比大于1，增加宽度
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        # 如果宽高比小于1，增加高度
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    # 保存图像
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """打印numpy数组的统计信息
    
    Args:
        x: numpy数组
        val (bool): 是否打印值的统计信息
        shp (bool): 是否打印数组形状
    """
    # 转换为float64类型
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        # 展平数组并打印统计信息
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """批量创建目录
    
    Args:
        paths (str list): 目录路径列表
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """创建单个目录
    
    Args:
        path (str): 目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    """正确调整标签图像大小
    
    使用最近邻插值调整标签图像大小，避免引入新的标签值。
    
    Args:
        t: 输入张量
        size: 目标尺寸
    
    Returns:
        调整大小后的张量
    """
    device = t.device
    # 将张量移到CPU
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        # 处理每个样本
        one_t = t[i, :1]
        # 转换为numpy数组
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        # 使用最近邻插值调整大小
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        # 转换回张量
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    """正确调整图像大小
    
    使用双三次插值调整图像大小。
    
    Args:
        t: 输入张量
        size: 目标尺寸
        mode: 插值模式，默认Image.BICUBIC
    
    Returns:
        调整大小后的张量
    """
    device = t.device
    # 将张量移到CPU
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        # 处理每个样本
        one_t = t[i:i + 1]
        # 转换为PIL图像并调整大小
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        # 转换回张量并缩放到[-1, 1]范围
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)

def draw_landmarks(img, landmark, color='r', step=2):
    """在图像上绘制面部关键点
    
    Args:
        img: 输入图像，numpy.array (B, H, W, 3)，RGB顺序，范围(0, 255)
        landmark: 面部关键点，numpy.array (B, 68, 2)，y方向与v方向相反
        color: 颜色，'r'为红色，'b'为蓝色
        step: 关键点绘制半径
    
    Returns:
        img: 绘制了关键点的图像
    """
    # 设置颜色
    if color =='r':
        c = np.array([255., 0, 0])  # 红色
    else:
        c = np.array([0, 0, 255.])  # 蓝色

    _, H, W, _ = img.shape
    # 复制图像和关键点，避免修改原始数据
    img, landmark = img.copy(), landmark.copy()
    # 翻转y坐标（图像坐标系与关键点坐标系y方向相反）
    landmark[..., 1] = H - 1 - landmark[..., 1]
    # 四舍五入关键点坐标
    landmark = np.round(landmark).astype(np.int32)
    # 绘制每个关键点
    for i in range(landmark.shape[1]):
        x, y = landmark[:, i, 0], landmark[:, i, 1]
        # 绘制一个方形区域
        for j in range(-step, step):
            for k in range(-step, step):
                # 确保坐标在图像范围内
                u = np.clip(x + j, 0, W - 1)
                v = np.clip(y + k, 0, H - 1)
                # 为每个样本设置颜色
                for m in range(landmark.shape[0]):
                    img[m, v[m], u[m]] = c
    return img
