"""Deep3DFaceRecon_pytorch 推理选项配置

该类定义了推理（Inference）阶段使用的特定选项，
继承自 BaseOptions 并添加推理相关的参数。
"""
from face3d.options.base_options import BaseOptions  # 导入基础选项类


class InferenceOptions(BaseOptions):
    """推理选项类。

    继承了 BaseOptions 中定义的所有共享选项，
    并添加了推理阶段特有的参数。
    """

    def initialize(self, parser):
        """初始化推理选项。

        参数:
            parser -- 命令行参数解析器

        返回:
            添加了推理选项的解析器
        """
        # 首先调用父类的初始化方法，定义共享选项
        parser = BaseOptions.initialize(self, parser)

        # ============ 推理特定选项 ============
        # 运行阶段：固定为 'test'
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # 数据集加载模式
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')

        # 输入文件所在目录
        parser.add_argument('--input_dir', type=str, help='the folder of the input files')
        # 关键点文件所在目录
        parser.add_argument('--keypoint_dir', type=str, help='the folder of the keypoint files')
        # 输出目录：保存提取的系数
        parser.add_argument('--output_dir', type=str, default='mp4', help='the output dir to save the extracted coefficients')
        # 是否保存拆分后的文件
        parser.add_argument('--save_split_files', action='store_true', help='save split files or not')
        # 推理批处理大小
        parser.add_argument('--inference_batch_size', type=int, default=8)
        
        # Dropout 和 BatchNorm 在训练和测试时行为不同
        # 测试模式下关闭 Dropout，BatchNorm 使用运行统计量
        self.isTrain = False
        return parser
