"""
ArcFace训练数据加载模块
本模块提供了高效的数据加载功能，支持MXNet RecordIO格式的人脸数据集。

主要组件：
1. BackgroundGenerator: 后台数据预加载线程，实现数据加载与GPU计算的流水线
2. DataLoaderX: 增强版DataLoader，支持异步数据预加载到GPU
3. MXFaceDataset: MXNet RecordIO格式的人脸数据集
4. SyntheticDataset: 合成数据集，用于快速调试和性能测试

使用异步数据加载可以显著减少GPU等待数据的时间，提高训练效率。
"""
import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BackgroundGenerator(threading.Thread):
    """
    后台数据生成器
    
    在后台线程中异步预取数据，减少数据加载对GPU计算的阻塞。
    使用队列实现生产者-消费者模式。
    """
    def __init__(self, generator, local_rank, max_prefetch=6):
        """
        初始化后台数据生成器
        
        参数:
            generator: 数据生成器（如DataLoader的迭代器）
            local_rank: 当前GPU的本地编号
            max_prefetch: 最大预取数量（队列大小）
        """
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)  # 预取队列
        self.generator = generator  # 数据生成器
        self.local_rank = local_rank
        self.daemon = True  # 设置为守护线程，主线程结束时自动退出
        self.start()  # 立即启动线程

    def run(self):
        """线程运行函数：从生成器读取数据并放入队列"""
        torch.cuda.set_device(self.local_rank)  # 设置CUDA设备
        for item in self.generator:
            self.queue.put(item)  # 将数据放入队列
        self.queue.put(None)  # 放入None标记数据结束

    def next(self):
        """获取下一个数据项"""
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration  # 数据读取完毕
        return next_item

    def __next__(self):
        """支持next()内置函数"""
        return self.next()

    def __iter__(self):
        """支持迭代器协议"""
        return self


class DataLoaderX(DataLoader):
    """
    增强版DataLoader，支持异步GPU数据传输
    
    利用CUDA流实现数据预加载，将CPU到GPU的数据传输与模型计算重叠执行，
    从而提高GPU利用率和训练速度。
    """

    def __init__(self, local_rank, **kwargs):
        """
        初始化增强版DataLoader
        
        参数:
            local_rank: 当前GPU的本地编号
            **kwargs: 传递给父类DataLoader的参数
        """
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)  # 创建CUDA流用于异步传输
        self.local_rank = local_rank

    def __iter__(self):
        """创建迭代器，使用后台线程预取数据"""
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)  # 后台线程预取
        self.preload()  # 开始预加载第一个batch
        return self

    def preload(self):
        """预加载下一个batch到GPU"""
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):  # 在异步流中执行数据传输
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        """获取下一个batch，等待异步传输完成"""
        torch.cuda.current_stream().wait_stream(self.stream)  # 等待异步流完成
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()  # 预加载下一个batch
        return batch


class MXFaceDataset(Dataset):
    """
    MXNet RecordIO格式的人脸数据集
    
    支持从MXNet的RecordIO文件中读取人脸图片和标签。
    RecordIO是一种高效的二进制存储格式，支持随机访问。
    
    数据预处理流程：
    1. 转换为PIL图像
    2. 随机水平翻转（数据增强）
    3. 转换为Tensor
    4. 归一化到[-1, 1]范围
    """
    def __init__(self, root_dir, local_rank):
        """
        初始化数据集
        
        参数:
            root_dir: 数据集根目录（包含train.rec和train.idx文件）
            local_rank: 当前GPU的本地编号
        """
        super(MXFaceDataset, self).__init__()
        # 数据预处理流水线
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),          # 转为PIL图像
             transforms.RandomHorizontalFlip(),  # 随机水平翻转（数据增强）
             transforms.ToTensor(),            # 转为Tensor
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化到[-1, 1]
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        # 加载MXNet RecordIO文件
        path_imgrec = os.path.join(root_dir, 'train.rec')  # 图像数据文件
        path_imgidx = os.path.join(root_dir, 'train.idx')  # 索引文件
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # 创建索引读取器
        s = self.imgrec.read_idx(0)  # 读取头部信息
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))  # 头部记录的类别数和图片数
            self.imgidx = np.array(range(1, int(header.label[0])))  # 生成索引数组
        else:
            self.imgidx = np.array(list(self.imgrec.keys))  # 使用所有键作为索引

    def __getitem__(self, index):
        """
        获取单个样本
        
        参数:
            index: 样本索引
        返回:
            sample: 处理后的图像Tensor
            label: 身份标签（Long类型）
        """
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)  # 读取图像记录
        header, img = mx.recordio.unpack(s)  # 解包
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]  # 取第一个标签值
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()  # 解码图像为numpy数组
        if self.transform is not None:
            sample = self.transform(sample)  # 应用数据预处理
        return sample, label

    def __len__(self):
        """返回数据集大小"""
        return len(self.imgidx)


class SyntheticDataset(Dataset):
    """
    合成数据集
    
    使用随机生成的图像数据，用于快速调试模型训练流程和性能测试。
    所有样本共享同一张随机图像和标签1。
    """
    def __init__(self, local_rank):
        """
        初始化合成数据集，生成一张112x112的随机图像
        """
        super(SyntheticDataset, self).__init__()
        # 生成随机图像并预处理
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)  # 随机RGB图像
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = torch.from_numpy(img).squeeze(0).float()  # 转为Tensor
        img = ((img / 255) - 0.5) / 0.5  # 归一化到[-1, 1]
        self.img = img
        self.label = 1  # 固定标签

    def __getitem__(self, index):
        """获取样本（始终返回同一张随机图像）"""
        return self.img, self.label

    def __len__(self):
        """返回数据集大小（返回一个大数以模拟大数据集）"""
        return 1000000
