"""模型类模板

该模块提供了一个模型模板，供用户实现自定义模型。
你可以通过指定 '--model template' 来使用此模型。

类名应与文件名和模型选项保持一致：
- 文件名格式：<model>_model.py
- 类名格式：<Model>Model.py

该模板实现了一个简单的基于回归损失的图像到图像翻译基线模型。
给定输入-输出对 (data_A, data_B)，它学习一个网络 netG 来最小化以下 L1 损失：
    min_<netG> ||netG(data_A) - data_B||_1

你需要实现以下函数：
    <modify_commandline_options>: 添加模型特定的选项并重写已有选项的默认值
    <__init__>: 初始化模型类
    <set_input>: 解包输入数据并执行数据预处理
    <forward>: 执行前向传播。由 <optimize_parameters> 和 <test> 调用
    <optimize_parameters>: 更新网络权重；在每个训练迭代中调用
"""
import numpy as np                  # NumPy 数值计算库
import torch                        # PyTorch 深度学习框架
from .base_model import BaseModel   # 导入模型基类
from . import networks              # 导入网络定义模块


class TemplateModel(BaseModel):
    """模板模型类，展示如何实现自定义模型。

    该类实现了一个简单的图像翻译模型：
    使用编码器-解码器网络将输入图像转换为输出图像，
    使用 L1 回归损失进行训练。
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的模型特定选项，并重写已有选项的默认值。

        参数:
            parser -- 命令行参数解析器
            is_train -- 是否为训练阶段。可据此添加训练/测试特定的选项

        返回:
            修改后的解析器
        """
        # 设置默认数据集模式为 'aligned'（配对数据集）
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            # 训练时添加回归损失权重参数
            parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')

        return parser

    def __init__(self, opt):
        """初始化模型类。

        参数:
            opt -- 训练/测试选项

        在此方法中可以执行以下操作：
        - （必须）调用 BaseModel 的初始化函数
        - 定义损失函数、可视化图像、模型名称和优化器
        """
        BaseModel.__init__(self, opt)  # 调用基类初始化

        # 指定要打印的训练损失名称
        # 程序会调用 base_model.get_current_losses 将损失输出到控制台并保存到磁盘
        self.loss_names = ['loss_G']

        # 指定要保存和显示的图像名称
        # 程序会调用 base_model.get_current_visuals 来保存和显示这些图像
        self.visual_names = ['data_A', 'data_B', 'output']

        # 指定要保存到磁盘的模型
        # 程序会调用 base_model.save_networks 和 base_model.load_networks 来保存和加载网络
        # 使用 opt.isTrain 可以指定训练和测试时的不同行为
        self.model_names = ['G']

        # 定义生成器网络
        # opt.input_nc: 输入通道数, opt.output_nc: 输出通道数
        # opt.ngf: 生成器特征图数, opt.netG: 网络架构类型
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)

        if self.isTrain:  # 仅在训练时定义损失和优化器
            # 定义损失函数：L1回归损失
            # 也可以使用 torch.nn 提供的其他损失，如 torch.nn.L1Loss
            self.criterionLoss = torch.nn.L1Loss()

            # 定义优化器：使用 Adam 优化器
            # 也可以为每个网络定义独立的优化器
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        # 程序会自动调用 <model.setup> 来定义调度器、加载网络和打印网络信息

    def set_input(self, input):
        """从数据加载器中解包输入数据并执行必要的预处理步骤。

        参数:
            input: 包含数据本身及其元数据信息的字典
        """
        # 使用 direction 参数决定数据映射方向（A->B 或 B->A）
        AtoB = self.opt.direction == 'AtoB'
        # 获取图像数据 A 和 B，并移到目标设备
        self.data_A = input['A' if AtoB else 'B'].to(self.device)
        self.data_B = input['B' if AtoB else 'A'].to(self.device)
        # 获取图像路径
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """执行前向传播。由 <optimize_parameters> 和 <test> 两个函数调用。"""
        # 使用生成器 G 生成输出图像
        self.output = self.netG(self.data_A)

    def backward(self):
        """计算损失和梯度；在每个训练迭代中调用。"""
        # 计算 L1 回归损失，并乘以权重系数
        self.loss_G = self.criterionLoss(self.output, self.data_B) * self.opt.lambda_regression
        # 反向传播计算网络 G 关于 loss_G 的梯度
        self.loss_G.backward()

    def optimize_parameters(self):
        """更新网络权重；在每个训练迭代中调用。"""
        self.forward()               # 首先执行前向传播，计算中间结果
        self.optimizer.zero_grad()   # 清除网络 G 的现有梯度
        self.backward()              # 计算网络 G 的梯度
        self.optimizer.step()        # 更新网络 G 的梯度
