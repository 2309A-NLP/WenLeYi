"""可视化器模块

该脚本定义了 Deep3DFaceRecon_pytorch 的可视化器，
用于在训练/测试过程中显示和保存结果图像、损失曲线等信息。
支持 TensorBoard 和 HTML 页面两种可视化方式。
"""
import numpy as np                  # NumPy 数值计算库
import os                           # 操作系统接口模块
import sys                          # 系统模块
import ntpath                       # 路径处理模块（用于提取文件名）
import time                         # 时间模块
from . import util, html            # 工具函数和 HTML 生成模块
from subprocess import Popen, PIPE  # 子进程管理
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 写入器


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """将图像保存到磁盘并添加到 HTML 页面。

    参数:
        webpage (HTML类) -- 存储图像的 HTML 网页类（详见 html.py）
        visuals (OrderedDict) -- 存储 (名称, 图像) 对的有序字典
        image_path (str) -- 用于创建图像路径的字符串
        aspect_ratio (float) -- 保存图像的宽高比，默认 1.0
        width (int) -- 图像将被缩放到 width x width
    """
    image_dir = webpage.get_image_dir()  # 获取图像存储目录
    short_path = ntpath.basename(image_path[0])  # 提取文件名（去掉路径）
    name = os.path.splitext(short_path)[0]        # 去掉文件扩展名

    webpage.add_header(name)  # 添加 HTML 标题
    ims, txts, links = [], [], []  # 图像名、文本、链接列表

    for label, im_data in visuals.items():
        # 将张量转换为 NumPy 图像
        im = util.tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)  # 图像保存名
        # 创建子目录（如 'input/', 'output/' 等）
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        # 保存图像到磁盘
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    # 将图像添加到 HTML 页面
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """可视化器类。

    该类包含多个函数，可以显示/保存图像并打印/保存日志信息。
    使用 TensorBoard 进行实时显示，使用 'dominate' 库（封装在 'HTML' 中）
    创建包含图像的 HTML 文件。
    """

    def __init__(self, opt):
        """初始化可视化器类。

        参数:
            opt -- 存储所有实验标志位；需要是 BaseOptions 的子类

        步骤1: 缓存训练/测试选项
        步骤2: 创建 TensorBoard 写入器
        步骤3: 创建 HTML 对象用于保存可视化页面
        步骤4: 创建日志文件用于保存训练损失
        """
        self.opt = opt              # 缓存选项
        self.use_html = opt.isTrain and not opt.no_html  # 是否使用 HTML 可视化
        # 创建 TensorBoard 写入器，日志保存到 checkpoints_dir/logs/name/
        self.writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'logs', opt.name))
        self.win_size = opt.display_winsize  # 可视化窗口大小
        self.name = opt.name        # 实验名称
        self.saved = False          # 是否已保存结果标志

        if self.use_html:
            # 创建 HTML 目录结构：checkpoints_dir/name/web/ 和 web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        # 创建训练损失日志文件
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """重置已保存状态标志。"""
        self.saved = False


    def display_current_results(self, visuals, total_iters, epoch, save_result):
        """在 TensorBoard 上显示当前结果；将当前结果保存到 HTML 文件。

        参数:
            visuals (OrderedDict) -- 要显示或保存的图像字典
            total_iters (int) -- 总迭代次数
            epoch (int) -- 当前轮次
            save_result (bool) -- 是否将当前结果保存到 HTML 文件
        """
        # 将所有图像添加到 TensorBoard
        for label, image in visuals.items():
            self.writer.add_image(label, util.tensor2im(image), total_iters, dataformats='HWC')

        # 如果需要保存或尚未保存，保存到 HTML
        if self.use_html and (save_result or not self.saved):
            self.saved = True
            # 保存图像到磁盘
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # 更新 HTML 页面
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            # 从当前轮次到第1轮，按时间倒序排列
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, total_iters, losses):
        """将当前损失值记录到 TensorBoard。

        参数:
            total_iters (int): 总迭代次数
            losses (OrderedDict): 损失值字典
        """
        for name, value in losses.items():
            self.writer.add_scalar(name, value, total_iters)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """在控制台打印当前损失值，并保存到磁盘。

        参数:
            epoch (int) -- 当前轮次
            iters (int) -- 当前轮次中的训练迭代次数（每轮结束重置为0）
            losses (OrderedDict) -- 训练损失字典，格式为 (name, float) 对
            t_comp (float) -- 每个数据点的计算时间（按 batch_size 归一化）
            t_data (float) -- 每个数据点的数据加载时间（按 batch_size 归一化）
        """
        # 构建损失信息字符串
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # 打印到控制台
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # 保存到日志文件


class MyVisualizer:
    """自定义可视化器类（支持多数据集）。

    相比 Visualizer 类，增加了以下功能：
    - 支持区分训练集/验证集/测试集的结果
    - 支持按批次保存结果图像
    - 支持选择性地添加 TensorBoard 图像
    """

    def __init__(self, opt):
        """初始化可视化器类。

        参数:
            opt -- 存储所有实验标志位；需要是 BaseOptions 的子类

        步骤1: 缓存训练/测试选项
        步骤2: 创建 TensorBoard 写入器
        步骤3: 创建日志文件用于保存训练损失
        """
        self.opt = opt              # 缓存选项
        self.name = opt.name        # 实验名称
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'results')  # 结果图像目录
        
        if opt.phase != 'test':
            # 训练/验证阶段：创建 TensorBoard 写入器
            self.writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, 'logs'))
            # 创建训练损失日志文件
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)


    def display_current_results(self, visuals, total_iters, epoch, dataset='train', save_results=False, count=0, name=None,
            add_image=True):
        """在 TensorBoard 上显示当前结果；将当前结果保存到磁盘。

        参数:
            visuals (OrderedDict) -- 要显示或保存的图像字典
            total_iters (int) -- 总迭代次数
            epoch (int) -- 当前轮次
            dataset (str) -- 数据集类型：'train'、'val' 或 'test'
            save_results (bool) -- 是否将结果保存到磁盘
            count (int) -- 当前批次的起始计数
            name (str, optional) -- 自定义保存文件名
            add_image (bool) -- 是否添加到 TensorBoard
        """
        for label, image in visuals.items():
            for i in range(image.shape[0]):  # 遍历批次中的每张图像
                image_numpy = util.tensor2im(image[i])

                if add_image:
                    # 添加到 TensorBoard，包含数据集标识和索引
                    self.writer.add_image(label + '%s_%02d' % (dataset, i + count),
                            image_numpy, total_iters, dataformats='HWC')

                if save_results:
                    # 构建保存路径：results/dataset/epoch_X_XXXXXX/
                    save_path = os.path.join(self.img_dir, dataset, 'epoch_%s_%06d' % (epoch, total_iters))
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)

                    # 确定文件名
                    if name is not None:
                        img_path = os.path.join(save_path, '%s.png' % name)
                    else:
                        img_path = os.path.join(save_path, '%s_%03d.png' % (label, i + count))
                    util.save_image(image_numpy, img_path)


    def plot_current_losses(self, total_iters, losses, dataset='train'):
        """将当前损失值记录到 TensorBoard（包含数据集标识）。

        参数:
            total_iters (int): 总迭代次数
            losses (OrderedDict): 损失值字典
            dataset (str): 数据集类型标识
        """
        for name, value in losses.items():
            # 损失名称附加数据集标识，例如 "loss_G/train"
            self.writer.add_scalar(name + '/%s' % dataset, value, total_iters)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, dataset='train'):
        """在控制台打印当前损失值，并保存到磁盘。

        参数:
            epoch (int) -- 当前轮次
            iters (int) -- 当前轮次中的训练迭代次数（每轮结束重置为0）
            losses (OrderedDict) -- 训练损失字典，格式为 (name, float) 对
            t_comp (float) -- 每个数据点的计算时间（按 batch_size 归一化）
            t_data (float) -- 每个数据点的数据加载时间（按 batch_size 归一化）
            dataset (str) -- 数据集类型标识
        """
        # 构建包含数据集标识的损失信息字符串
        message = '(dataset: %s, epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (
            dataset, epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # 打印到控制台
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # 保存到日志文件
