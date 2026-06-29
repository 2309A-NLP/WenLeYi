"""数据集类模板

该模块提供了一个数据集模板，供用户实现自定义数据集。
你可以通过指定 '--dataset_mode template' 来使用此数据集。

类名应与文件名和 dataset_mode 选项保持一致：
- 文件名格式：<dataset_mode>_dataset.py
- 类名格式：<Dataset_mode>Dataset.py

你需要实现以下函数：
    -- <modify_commandline_options>: 添加数据集特定的选项并重写已有选项的默认值
    -- <__init__>: 初始化数据集类
    -- <__getitem__>: 返回一个数据点及其元数据信息
    -- <__len__>: 返回图像数量
"""
from data.base_dataset import BaseDataset, get_transform  # 从基类导入数据集基类和变换函数
# from data.image_folder import make_dataset  # 可选：从图像文件夹模块导入
# from PIL import Image  # 可选：图像处理库


class TemplateDataset(BaseDataset):
    """自定义数据集的模板类。

    该类展示了如何继承 BaseDataset 实现自己的数据集，
    包含了所有必要方法的框架代码和详细注释。
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的数据集特定选项，并重写已有选项的默认值。

        参数:
            parser          -- 原始的命令行参数解析器
            is_train (bool) -- 是否为训练阶段。可据此添加训练/测试特定的选项

        返回:
            修改后的解析器
        """
        # 添加数据集特定的新选项
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # 设置数据集特定的默认值
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)
        return parser

    def __init__(self, opt):
        """初始化数据集类。

        参数:
            opt (Option类) -- 存储所有实验标志位；需要是 BaseOptions 的子类

        在此方法中可以执行以下操作：
        - 保存选项（已在 BaseDataset 中完成）
        - 获取图像路径和数据集的元信息
        - 定义图像变换
        """
        # 保存选项和数据根目录（调用父类初始化）
        BaseDataset.__init__(self, opt)
        # 获取数据集中的图像路径列表
        # 提示：可以调用 sorted(make_dataset(self.root, opt.max_dataset_size)) 获取目录下所有图像路径
        self.image_paths = []
        # 定义默认的图像变换函数
        # 可以使用 base_dataset.get_transform，也可以自定义变换函数
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """返回一个数据点及其元数据信息。

        参数:
            index -- 用于数据索引的随机整数

        返回:
            包含数据及其名称的字典。通常包含数据本身及其元数据信息。

        实现步骤：
        步骤1：获取随机图像路径，例如：path = self.image_paths[index]
        步骤2：从磁盘加载数据，例如：image = Image.open(path).convert('RGB')
        步骤3：将数据转换为 PyTorch 张量，例如：data = self.transform(image)
        步骤4：以字典形式返回数据点
        """
        path = 'temp'    # 需要是字符串类型
        data_A = None    # 需要是张量类型
        data_B = None    # 需要是张量类型
        return {'data_A': data_A, 'data_B': data_B, 'path': path}

    def __len__(self):
        """返回数据集中图像的总数。"""
        return len(self.image_paths)
