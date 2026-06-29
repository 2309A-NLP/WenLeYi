"""数据加载与预处理模块包

该包包含了所有与数据加载和预处理相关的模块。

要添加一个名为 'dummy' 的自定义数据集类，你需要：
1. 添加一个名为 'dummy_dataset.py' 的文件
2. 定义一个继承自 BaseDataset 的子类 'DummyDataset'
3. 实现以下四个函数：
   -- <__init__>:                      初始化类，首先调用 BaseDataset.__init__(self, opt)
   -- <__len__>:                       返回数据集的大小
   -- <__getitem__>:                   从数据加载器中获取一个数据点
   -- <modify_commandline_options>:    （可选）添加数据集特定的选项并设置默认选项

现在你可以通过指定参数 '--dataset_mode dummy' 来使用该数据集类。
请参考我们的模板数据集类 'template_dataset.py' 了解更多细节。
"""
import numpy as np                  # NumPy 数值计算库
import importlib                    # 动态导入模块的工具库
import torch.utils.data             # PyTorch 数据工具模块
from face3d.data.base_dataset import BaseDataset  # 从基类模块导入数据集基类


def find_dataset_using_name(dataset_name):
    """根据名称动态导入并查找数据集类。

    该函数通过动态导入 "data/[dataset_name]_dataset.py" 模块，
    在该模块中查找名为 DatasetNameDataset() 的类并实例化。
    该类必须是 BaseDataset 的子类，且查找不区分大小写。

    参数:
        dataset_name (str): 数据集名称，例如 'flist'

    返回:
        对应的数据集类

    异常:
        NotImplementedError: 如果在模块中找不到匹配的子类
    """
    # 构建模块文件名，例如 "data.flist_dataset"
    dataset_filename = "data." + dataset_name + "_dataset"
    # 动态导入模块
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    # 将数据集名称去掉下划线并加上 'dataset' 后缀作为目标类名
    # 例如 'flist' -> 'flistdataset'
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    # 遍历模块中所有属性，查找匹配的类
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    # 如果未找到匹配的类，抛出异常
    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """返回数据集类的静态方法 <modify_commandline_options>。

    该方法用于在命令行解析阶段修改数据集相关的参数选项。

    参数:
        dataset_name (str): 数据集名称

    返回:
        对应数据集类的 modify_commandline_options 方法
    """
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, rank=0):
    """根据给定的选项创建数据集。

    该函数封装了 CustomDatasetDataLoader 类，
    是本包与 'train.py'/'test.py' 之间的主要接口。

    参数:
        opt: 实验选项对象
        rank (int): 分布式训练中的进程编号，默认为 0

    返回:
        数据集加载器实例

    使用示例:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    # 创建自定义数据集加载器实例
    data_loader = CustomDatasetDataLoader(opt, rank=rank)
    # 加载数据并返回
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """数据集类的包装器，执行多线程数据加载。

    该类封装了 PyTorch 的 DataLoader，提供了统一的数据加载接口，
    支持普通模式和分布式数据并行(DDP)模式。
    """

    def __init__(self, opt, rank=0):
        """初始化该类。

        步骤1: 根据 [dataset_mode] 名称创建数据集实例
        步骤2: 创建多线程数据加载器

        参数:
            opt: 实验选项对象，包含数据集模式等配置
            rank (int): 分布式训练中的进程编号
        """
        self.opt = opt
        # 步骤1: 根据选项中的 dataset_mode 查找并实例化对应的数据集类
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        self.sampler = None  # 分布式采样器，DDP模式下使用
        print("rank %d %s dataset [%s] was created" % (rank, self.dataset.name, type(self.dataset).__name__))

        # 步骤2: 根据是否使用分布式数据并行(DDP)创建不同的数据加载器
        if opt.use_ddp and opt.isTrain:
            # 分布式训练模式：使用 DistributedSampler 确保每个进程获得不同的数据子集
            world_size = opt.world_size
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                    self.dataset,
                    num_replicas=world_size,  # 总进程数
                    rank=rank,                 # 当前进程编号
                    shuffle=not opt.serial_batches  # 是否打乱数据顺序
                )
            # 创建分布式数据加载器，batch_size 和 num_workers 按世界大小分配
            self.dataloader = torch.utils.data.DataLoader(
                        self.dataset,
                        sampler=self.sampler,
                        num_workers=int(opt.num_threads / world_size), 
                        batch_size=int(opt.batch_size / world_size), 
                        drop_last=True)  # 丢弃最后一个不完整的批次
        else:
            # 普通模式：直接使用 DataLoader，训练时可打乱顺序
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=(not opt.serial_batches) and opt.isTrain,
                num_workers=int(opt.num_threads),
                drop_last=True
            )

    def set_epoch(self, epoch):
        """设置当前轮次（epoch），用于分布式采样器的随机种子更新。

        参数:
            epoch (int): 当前训练轮次
        """
        self.dataset.current_epoch = epoch
        # 如果使用了分布式采样器，需要更新其 epoch 以保证每轮数据顺序不同
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def load_data(self):
        """返回自身实例，作为数据加载的统一接口。"""
        return self

    def __len__(self):
        """返回数据集中数据的数量（受 max_dataset_size 限制）。"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """迭代返回一个批次的数据。

        该方法支持 Python 的迭代器协议，可用于 for 循环。
        当累计处理的数据量超过 max_dataset_size 时自动停止。
        """
        for i, data in enumerate(self.dataloader):
            # 检查是否超过最大数据集大小限制
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data  # 使用 yield 生成器逐批返回数据
