"""3D人脸重建基础模型

该脚本定义了 Deep3DFaceRecon_pytorch 的基础网络模型类 BaseModel。
所有具体模型都应继承此抽象基类。
"""
import os                           # 操作系统接口模块
import numpy as np                  # NumPy 数值计算库
import torch                        # PyTorch 深度学习框架
from collections import OrderedDict  # 有序字典，用于保持损失和可视化图像的顺序
from abc import ABC, abstractmethod  # 抽象基类支持
from . import networks              # 从当前包导入网络定义模块


class BaseModel(ABC):
    """模型抽象基类 (ABC)。

    所有自定义模型都应继承此类，并实现以下五个函数：
        -- <__init__>:                      初始化类；首先调用 BaseModel.__init__(self, opt)
        -- <set_input>:                     从数据集解包数据并进行预处理
        -- <forward>:                       产生中间结果
        -- <optimize_parameters>:           计算损失、梯度并更新网络权重
        -- <modify_commandline_options>:    （可选）添加模型特定的选项并设置默认选项
    """

    def __init__(self, opt):
        """初始化 BaseModel 类。

        参数:
            opt (Option类) -- 存储所有实验标志位；需要是 BaseOptions 的子类

        创建自定义类时，你需要：
        1. 首先调用 <BaseModel.__init__(self, opt)>
        2. 定义以下四个列表：
            -- self.loss_names (str list):          指定要绘制和保存的训练损失
            -- self.model_names (str list):         指定要显示和保存的图像
            -- self.visual_names (str list):        定义训练中使用的网络
            -- self.optimizers (optimizer list):    定义并初始化优化器
        """
        self.opt = opt              # 保存实验选项
        self.isTrain = False        # 是否为训练模式标志
        self.device = torch.device('cpu')  # 默认使用 CPU 设备
        self.save_dir = " "         # 模型保存目录（os.path.join(opt.checkpoints_dir, opt.name)）
        self.loss_names = []        # 损失名称列表
        self.model_names = []       # 模型名称列表
        self.visual_names = []      # 可视化名称列表
        self.parallel_names = []    # 需要并行化的网络名称列表
        self.optimizers = []        # 优化器列表
        self.image_paths = []       # 图像路径列表
        self.metric = 0             # 用于学习率策略 'plateau' 的指标值

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        """创建梯度钩子工厂，用于捕获和存储网络的梯度信息。

        参数:
            add_func: 对梯度进行额外处理的函数，默认为恒等函数

        返回:
            hook_gen (callable): 生成钩子的工厂函数
            saved_dict (dict): 存储梯度值的字典
        """
        saved_dict = dict()         # 存储捕获到的梯度值

        def hook_gen(name):
            """为指定名称的参数生成梯度钩子。"""
            def grad_hook(grad):
                # 对梯度应用自定义函数并保存
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的模型特定选项，并重写已有选项的默认值。

        参数:
            parser          -- 原始的命令行参数解析器
            is_train (bool) -- 是否为训练阶段。可据此添加训练/测试特定的选项

        返回:
            修改后的解析器
        """
        return parser  # 基类默认不修改任何选项

    @abstractmethod
    def set_input(self, input):
        """从数据加载器中解包输入数据并执行必要的预处理步骤。

        参数:
            input (dict): 包含数据本身及其元数据信息
        """
        pass

    @abstractmethod
    def forward(self):
        """执行前向传播；由 <optimize_parameters> 和 <test> 两个函数调用。"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重；在每个训练迭代中调用。"""
        pass

    def setup(self, opt):
        """加载和打印网络信息；创建学习率调度器。

        参数:
            opt (Option类) -- 存储所有实验标志位；需要是 BaseOptions 的子类
        """
        if self.isTrain:
            # 为每个优化器创建学习率调度器
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        
        # 非训练模式或继续训练时，加载预训练权重
        if not self.isTrain or opt.continue_train:
            load_suffix = opt.epoch  # 加载哪个轮次的模型
            self.load_networks(load_suffix)
 
            
        # self.print_networks(opt.verbose)

    def parallelize(self, convert_sync_batchnorm=True):
        """将网络模型移至GPU并进行并行化处理。

        根据是否使用分布式数据并行(DDP)：
        - 非DDP模式：直接将模型移到GPU设备
        - DDP模式：将模型包装为 DistributedDataParallel

        参数:
            convert_sync_batchnorm (bool): 是否将 BatchNorm 转换为 SyncBatchNorm，默认 True
        """
        if not self.opt.use_ddp:
            # 非分布式模式：直接将网络移到目标设备
            for name in self.parallel_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))
        else:
            # 分布式数据并行模式
            for name in self.model_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    if convert_sync_batchnorm:
                        # 将 BatchNorm 转换为跨GPU同步的 BatchNorm
                        module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
                    # 包装为 DistributedDataParallel
                    setattr(self, name, torch.nn.parallel.DistributedDataParallel(module.to(self.device),
                        device_ids=[self.device.index], 
                        find_unused_parameters=True,  # 查找未使用的参数
                        broadcast_buffers=True))  # 广播缓冲区
            
            # 对于没有需要梯度的参数的网络，不需要 DDP 包装
            for name in self.parallel_names:
                if isinstance(name, str) and name not in self.model_names:
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))
            
        # 将优化器的状态字典移到GPU设备
        if self.opt.phase != 'test':
            if self.opt.continue_train:
                for optim in self.optimizers:
                    for state in optim.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)

    def data_dependent_initialize(self, data):
        """数据依赖的初始化（子类可覆盖）。"""
        pass

    def train(self):
        """将所有模型设置为训练模式。"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()  # 启用 Dropout 和 BatchNorm 的训练行为

    def eval(self):
        """将所有模型设置为评估模式。"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()  # 禁用 Dropout，BatchNorm 使用运行统计量

    def test(self):
        """测试时使用的前向传播函数。

        该函数在 torch.no_grad() 上下文中包装 <forward> 函数，
        不保存中间步骤的梯度信息（节省内存）。
        同时调用 <compute_visuals> 生成额外的可视化结果。
        """
        with torch.no_grad():  # 禁用梯度计算，节省显存
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """计算用于 visdom 和 HTML 可视化的额外输出图像。"""
        pass

    def get_image_paths(self, name='A'):
        """返回用于加载当前数据的图像路径。

        参数:
            name (str): 数据域标识，'A' 或 'B'

        返回:
            图像路径列表
        """
        return self.image_paths if name == 'A' else self.image_paths_B

    def update_learning_rate(self):
        """更新所有网络的学习率；在每个训练轮次结束时调用。

        根据学习率策略更新调度器：
        - plateau 策略：基于指标值（如验证损失）调整
        - 其他策略：直接按预定规则调整
        """
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)  # 基于指标值更新
            else:
                scheduler.step()             # 按步数更新

        # 打印当前学习率
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """返回可视化图像。train.py 将使用 visdom 显示这些图像，并保存到 HTML。"""
        visual_ret = OrderedDict()  # 使用有序字典保持顺序
        for name in self.visual_names:
            if isinstance(name, str):
                # 获取可视化张量，取前3个通道（RGB）
                visual_ret[name] = getattr(self, name)[:, :3, ...]
        return visual_ret

    def get_current_losses(self):
        """返回训练损失/误差。train.py 将在控制台打印这些错误并保存到文件。"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # 将损失张量转换为浮点数
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        """将所有网络保存到磁盘。

        参数:
            epoch (int) -- 当前轮次；用于文件名格式 '%s_net_%s.pth' % (epoch, name)

        保存内容包括：
        - 所有模型的 state_dict
        - 所有优化器的 state_dict
        - 所有学习率调度器的 state_dict
        """
        # 如果保存目录不存在，创建它
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        # 构建保存文件名和路径
        save_filename = 'epoch_%s.pth' % (epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        
        save_dict = {}
        # 保存每个模型的 state_dict
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                # 如果模型被 DataParallel 或 DDP 包装，需要获取原始模型
                if isinstance(net, torch.nn.DataParallel) or isinstance(net,
                        torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                save_dict[name] = net.state_dict()
                

        # 保存优化器状态
        for i, optim in enumerate(self.optimizers):
            save_dict['opt_%02d' % i] = optim.state_dict()

        # 保存学习率调度器状态
        for i, sched in enumerate(self.schedulers):
            save_dict['sched_%02d' % i] = sched.state_dict()
        
        # 使用 torch.save 保存整个字典
        torch.save(save_dict, save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """修复 InstanceNorm 检查点不兼容问题（0.4版本之前）。

        在 PyTorch 0.4 之前，InstanceNorm 层的 running_mean、running_var
        和 num_batches_tracked 参数可能不存在，需要从 state_dict 中移除。
        """
        key = keys[i]
        if i + 1 == len(keys):  # 到达最后一层，指向参数/缓冲区
            # 移除不存在的 InstanceNorm 运行统计量
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            # 移除 num_batches_tracked
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            # 递归处理嵌套模块
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """从磁盘加载所有网络。

        参数:
            epoch (int) -- 当前轮次；用于文件名格式 '%s_net_%s.pth' % (epoch, name)

        加载逻辑：
        - 如果指定了预训练模型名称，从对应目录加载
        - 否则从默认保存目录加载
        - 继续训练时还会加载优化器和调度器状态
        """
        # 确定加载目录
        if self.opt.isTrain and self.opt.pretrained_name is not None:
            load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
        else:
            load_dir = self.save_dir    
        load_filename = 'epoch_%s.pth' % (epoch)
        load_path = os.path.join(load_dir, load_filename)
        # 加载检查点文件
        state_dict = torch.load(load_path, map_location=self.device)
        print('loading the model from %s' % load_path)

        # 加载每个模型的权重
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name])
        
        # 继续训练时，加载优化器和调度器状态
        if self.opt.phase != 'test':
            if self.opt.continue_train:
                print('loading the optim from %s' % load_path)
                for i, optim in enumerate(self.optimizers):
                    optim.load_state_dict(state_dict['opt_%02d' % i])

                try:
                    print('loading the sched from %s' % load_path)
                    for i, sched in enumerate(self.schedulers):
                        sched.load_state_dict(state_dict['sched_%02d' % i])
                except:
                    # 如果调度器加载失败，根据轮次计数手动设置
                    print('Failed to load schedulers, set schedulers according to epoch count manually')
                    for i, sched in enumerate(self.schedulers):
                        sched.last_epoch = self.opt.epoch_count - 1
                    

    def print_networks(self, verbose):
        """打印网络的参数总数和（如果verbose为True）网络架构。

        参数:
            verbose (bool) -- 是否打印详细的网络架构信息
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                # 统计参数总数
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)  # 打印详细架构
                # 打印参数量（以百万为单位）
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """设置网络是否需要梯度计算，避免不必要的计算开销。

        参数:
            nets (network list)   -- 网络列表
            requires_grad (bool)  -- 是否需要梯度，默认为 False
        """
        if not isinstance(nets, list):
            nets = [nets]  # 将单个网络转换为列表
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        """为评估生成可视化结果（子类可覆盖）。

        参数:
            data: 输入数据
            mode: 模式标识

        返回:
            空字典（默认实现）
        """
        return {}
