"""Deep3DFaceRecon_pytorch 基础选项配置

该脚本包含训练和测试共用的基础命令行参数选项定义。
"""
import argparse                     # 命令行参数解析模块
import os                           # 操作系统接口模块
from util import util               # 工具函数模块
import numpy as np                  # NumPy 数值计算库
import torch                        # PyTorch 深度学习框架
import face3d.models as models      # 模型包
import face3d.data as data          # 数据包


class BaseOptions():
    """基础选项类，定义训练和测试共用的选项。

    该类实现了多个辅助函数，如解析、打印和保存选项。
    同时收集在数据集类和模型类的 <modify_commandline_options> 函数中定义的额外选项。
    """

    def __init__(self, cmd_line=None):
        """初始化类；表示该类尚未完成初始化。"""
        self.initialized = False    # 是否已初始化标志
        self.cmd_line = None        # 命令行参数
        if cmd_line is not None:
            # 将字符串命令行参数分割为列表
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """定义训练和测试共用的基础选项。"""
        # ============ 基础参数 ============
        # 实验名称：决定样本和模型的存储位置
        parser.add_argument('--name', type=str, default='face_recon', help='name of the experiment. It decides where to store samples and models')
        # GPU 设备 ID：例如 '0', '0,1,2', '0,2'；使用 '-1' 表示 CPU
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # 模型检查点保存目录
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # 可视化的批次数
        parser.add_argument('--vis_batch_nums', type=float, default=1, help='batch nums of images for visulization')
        # 评估的批次数（默认无穷大，即全部评估）
        parser.add_argument('--eval_batch_nums', type=float, default=float('inf'), help='batch nums of images for evaluation')
        # 是否使用分布式数据并行 (DDP)
        parser.add_argument('--use_ddp', type=util.str2bool, nargs='?', const=True, default=True, help='whether use distributed data parallel')
        # DDP 通信端口
        parser.add_argument('--ddp_port', type=str, default='12355', help='ddp port')
        # 是否按批次显示损失信息
        parser.add_argument('--display_per_batch', type=util.str2bool, nargs='?', const=True, default=True, help='whether use batch to show losses')
        # 是否将图像添加到 TensorBoard
        parser.add_argument('--add_image', type=util.str2bool, nargs='?', const=True, default=True, help='whether add image to tensorboard')
        # 分布式训练的世界大小（进程数）
        parser.add_argument('--world_size', type=int, default=1, help='batch nums of images for evaluation')

        # ============ 模型参数 ============
        # 选择使用的模型
        parser.add_argument('--model', type=str, default='facerecon', help='chooses which model to use.')

        # ============ 附加参数 ============
        # 加载哪个轮次的模型？设置为 'latest' 使用最新的缓存模型
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        # 是否打印更多调试信息
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        # 自定义后缀：opt.name = opt.name + suffix
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        self.initialized = True     # 标记为已初始化
        return parser

    def gather_options(self):
        """初始化解析器并收集所有选项。

        步骤：
        1. 使用基本选项初始化解析器（仅执行一次）
        2. 添加模型特定和数据集特定的选项
        这些选项在模型和数据集类的 <modify_commandline_options> 函数中定义
        """
        if not self.initialized:  # 检查是否已初始化
            # 创建参数解析器，自动显示参数默认值
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # 获取基本选项
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()  # 解析已知参数，忽略未知参数
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # 设置 CUDA 可见设备
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

        # 修改模型相关的解析器选项
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)  # 获取模型的选项修改方法
        parser = model_option_setter(parser, self.isTrain)
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()  # 使用新的默认值重新解析
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # 修改数据集相关的解析器选项
        if opt.dataset_mode:
            dataset_name = opt.dataset_mode
            dataset_option_setter = data.get_option_setter(dataset_name)  # 获取数据集的选项修改方法
            parser = dataset_option_setter(parser, self.isTrain)

        # 保存解析器并返回最终解析结果
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        """打印并保存选项。

        将打印当前选项和默认值（如果不同）。
        选项将保存到文本文件 / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            # 如果当前值与默认值不同，添加注释说明默认值
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # 保存到磁盘
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)  # 创建实验目录
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        try:
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        except PermissionError as error:
            print("permission error {}".format(error))
            pass

    def parse(self):
        """解析选项，创建检查点目录后缀，并设置 GPU 设备。"""
        opt = self.gather_options()  # 收集所有选项
        opt.isTrain = self.isTrain   # 设置训练/测试标志

        # 处理选项后缀
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # 设置 GPU ID
        str_ids = opt.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        opt.world_size = len(gpu_ids)

        # 如果只有1个GPU，自动关闭分布式训练
        if opt.world_size == 1:
            opt.use_ddp = False

        # 非测试阶段：自动检测是否需要继续训练
        if opt.phase != 'test':
            # 确定模型目录
            if opt.pretrained_name is None:
                model_dir = os.path.join(opt.checkpoints_dir, opt.name)
            else:
                model_dir = os.path.join(opt.checkpoints_dir, opt.pretrained_name)

            # 检查是否已有检查点文件
            if os.path.isdir(model_dir):
                model_pths = [i for i in os.listdir(model_dir) if i.endswith('pth')]
                if os.path.isdir(model_dir) and len(model_pths) != 0:
                    opt.continue_train = True  # 自动设置继续训练标志
        
            # 更新最新的轮次计数
            if opt.continue_train:
                if opt.epoch == 'latest':
                    # 获取所有轮次编号（排除 latest 文件），找到最大值
                    epoch_counts = [int(i.split('.')[0].split('_')[-1]) for i in model_pths if 'latest' not in i]
                    if len(epoch_counts) != 0:
                        opt.epoch_count = max(epoch_counts) + 1  # 从下一个轮次开始
                else:
                    opt.epoch_count = int(opt.epoch) + 1
                    

        self.print_options(opt)     # 打印并保存选项
        self.opt = opt              # 保存选项到实例
        return self.opt
