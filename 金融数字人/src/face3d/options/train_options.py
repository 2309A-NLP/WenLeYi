"""Deep3DFaceRecon_pytorch 训练选项配置

该类定义了训练阶段使用的特定选项，
继承自 BaseOptions 并添加数据集、可视化、模型保存和训练相关的参数。
"""
from .base_options import BaseOptions  # 导入基础选项类
from util import util                   # 工具函数模块


class TrainOptions(BaseOptions):
    """训练选项类。

    继承了 BaseOptions 中定义的所有共享选项，
    并添加了训练阶段特有的参数。
    """

    def initialize(self, parser):
        """初始化训练选项。

        参数:
            parser -- 命令行参数解析器

        返回:
            添加了训练选项的解析器
        """
        # 首先调用父类的初始化方法，定义共享选项
        parser = BaseOptions.initialize(self, parser)

        # ============ 数据集参数（训练） ============
        # 数据根目录
        parser.add_argument('--data_root', type=str, default='./', help='dataset root')
        # 训练集掩码文件列表路径
        parser.add_argument('--flist', type=str, default='datalist/train/masks.txt', help='list of mask names of training set')
        # 训练批处理大小
        parser.add_argument('--batch_size', type=int, default=32)
        # 数据集加载模式
        parser.add_argument('--dataset_mode', type=str, default='flist', help='chooses how datasets are loaded. [None | flist]')
        # 是否按顺序加载图像（不打乱），默认随机加载
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        # 数据加载的线程数
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        # 每个数据集允许的最大样本数
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        # 图像预处理方式：平移、缩放、旋转、翻转的组合
        parser.add_argument('--preprocess', type=str, default='shift_scale_rot_flip', help='scaling and cropping of images at load time [shift_scale_rot_flip | shift_scale | shift | shift_rot_flip ]')
        # 是否使用数据增强
        parser.add_argument('--use_aug', type=util.str2bool, nargs='?', const=True, default=True, help='whether use data augmentation')

        # ============ 数据集参数（验证） ============
        # 验证集掩码文件列表路径
        parser.add_argument('--flist_val', type=str, default='datalist/val/masks.txt', help='list of mask names of val set')
        # 验证批处理大小
        parser.add_argument('--batch_size_val', type=int, default=32)


        # ============ 可视化参数 ============
        # 在屏幕上显示训练结果的频率（每N个批次）
        parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results on screen')
        # 在控制台打印训练结果的频率
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        
        # ============ 模型保存和加载参数 ============
        # 保存最新模型的频率（每N个批次）
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        # 每隔多少个轮次保存检查点
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        # 评估频率
        parser.add_argument('--evaluation_freq', type=int, default=5000, help='evaluation freq')
        # 是否按迭代次数保存模型
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        # 是否继续训练（加载最新模型）
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        # 起始轮次计数
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        # 运行阶段：固定为 'train'
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # 从另一个检查点恢复训练
        parser.add_argument('--pretrained_name', type=str, default=None, help='resume training from another checkpoint')

        # ============ 训练参数 ============
        # 使用初始学习率训练的轮次数
        parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
        # Adam 优化器的初始学习率
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        # 学习率策略：[linear | step | plateau | cosine]
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
        # 学习率衰减周期：每隔多少个轮次乘以衰减系数
        parser.add_argument('--lr_decay_epochs', type=int, default=10, help='multiply by a gamma every lr_decay_epochs epoches')

        # 标记为训练模式
        self.isTrain = True
        return parser
