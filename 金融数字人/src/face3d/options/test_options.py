"""Deep3DFaceRecon_pytorch 测试选项配置

该类定义了测试阶段使用的特定选项，
继承自 BaseOptions 并添加测试相关的参数。
"""
from .base_options import BaseOptions  # 导入基础选项类


class TestOptions(BaseOptions):
    """测试选项类。

    继承了 BaseOptions 中定义的所有共享选项，
    并添加了测试阶段特有的参数。
    """

    def initialize(self, parser):
        """初始化测试选项。

        参数:
            parser -- 命令行参数解析器

        返回:
            添加了测试选项的解析器
        """
        # 首先调用父类的初始化方法，定义共享选项
        parser = BaseOptions.initialize(self, parser)  # define shared options

        # ============ 测试特定选项 ============
        # 运行阶段：固定为 'test'
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # 数据集加载模式
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')
        # 测试图像所在的文件夹
        parser.add_argument('--img_folder', type=str, default='examples', help='folder for test images.')

        # Dropout 和 BatchNorm 在训练和测试时行为不同
        # 测试模式下关闭 Dropout，BatchNorm 使用运行统计量
        self.isTrain = False
        return parser
