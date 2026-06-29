"""图像文件夹加载模块

修改自官方 PyTorch 图像文件夹类
(https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)，
使其能够从当前目录及其子目录中递归加载图像文件。
"""
import numpy as np                  # NumPy 数值计算库
import torch.utils.data as data     # PyTorch 数据工具模块

from PIL import Image               # PIL 图像处理库
import os                           # 操作系统接口模块
import os.path                      # 路径处理模块

# 支持的图像文件扩展名列表
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',       # JPEG 格式
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',  # PNG、PPM、BMP 格式
    '.tif', '.TIF', '.tiff', '.TIFF',       # TIFF 格式
]


def is_image_file(filename):
    """检查给定的文件名是否为支持的图像文件格式。

    参数:
        filename (str): 文件名

    返回:
        bool: 如果文件扩展名在支持列表中则返回 True
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    """递归遍历目录，生成所有图像文件路径列表。

    该函数会遍历指定目录（包括所有子目录），
    找到所有支持格式的图像文件并返回其完整路径。

    参数:
        dir (str): 图像根目录路径
        max_dataset_size (int/float): 最大数据集大小限制，默认为无穷大

    返回:
        images (list): 图像文件路径列表

    异常:
        AssertionError: 如果指定路径不是有效目录
    """
    images = []
    # 验证目录有效性（支持符号链接）
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    # 递归遍历目录树，followlinks=True 表示跟随符号链接
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)  # 构建完整路径
                images.append(path)
    # 根据最大数据集大小限制截断列表
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    """默认的图像加载函数。

    打开图像文件并转换为 RGB 模式。

    参数:
        path (str): 图像文件路径

    返回:
        PIL.Image: RGB 模式的图像
    """
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    """通用的图像文件夹数据集类。

    支持从目录中自动发现图像文件，
    并可选地返回图像路径。
    """

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        """初始化图像文件夹数据集。

        参数:
            root (str): 图像根目录路径
            transform: 图像预处理变换（可选）
            return_paths (bool): 是否返回图像路径，默认为 False
            loader: 图像加载函数，默认为 default_loader
        """
        # 从根目录中获取所有图像文件路径
        imgs = make_dataset(root)
        # 如果未找到任何图像，抛出异常
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root          # 根目录路径
        self.imgs = imgs          # 图像路径列表
        self.transform = transform  # 图像变换
        self.return_paths = return_paths  # 是否返回路径
        self.loader = loader      # 图像加载函数

    def __getitem__(self, index):
        """获取指定索引的图像数据。

        参数:
            index (int): 图像索引

        返回:
            如果 return_paths 为 True，返回 (图像, 路径)
            否则只返回图像
        """
        path = self.imgs[index]   # 获取图像路径
        img = self.loader(path)   # 加载图像
        if self.transform is not None:
            img = self.transform(img)  # 应用预处理变换
        if self.return_paths:
            return img, path      # 返回图像和路径
        else:
            return img            # 只返回图像

    def __len__(self):
        """返回数据集中图像的总数。"""
        return len(self.imgs)
