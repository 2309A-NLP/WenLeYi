# -*- coding: utf-8 -*-
# config.py - 配置管理模块
# 本文件提供了配置类（Config）和辅助工具函数，
# 用于从YAML文件加载超参数配置，支持递归更新和属性字典访问。

import collections
import functools
import os
import re

import yaml

class AttrDict(dict):
    """属性字典类
    继承自Python内置dict，允许通过点号（.）语法访问字典的键值对。
    例如：d.key 等价于 d['key']。
    支持嵌套的AttrDict转换，即将内部的dict也转换为AttrDict。
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        # 将自身字典作为对象的__dict__，实现属性访问
        self.__dict__ = self
        for key, value in self.__dict__.items():
            # 递归将内部dict转换为AttrDict
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, (list, tuple)):
                # 如果列表/元组的第一个元素是dict，将所有元素转为AttrDict
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """将AttrDict对象转换为普通字典，用于YAML序列化
        递归地将所有嵌套的AttrDict转换为普通dict，
        以便可以正确地序列化为YAML格式。
        """
        yaml_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """格式化打印所有变量
        递归地将所有键值对格式化为可读的字符串表示，
        嵌套的AttrDict会缩进显示。
        """
        ret_str = []
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in value:
                        # 与上面的AttrDict处理方式相同
                        child_ret_str = item.__repr__().split('\n')
                        for item in child_ret_str:
                            ret_str.append('    ' + item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)


class Config(AttrDict):
    """配置类
    继承自AttrDict，用于管理训练和推理的所有超参数配置。
    包含默认参数设置，并支持从YAML文件加载和更新配置。
    配置项包括：日志设置、网络结构、优化器参数、数据配置、训练器配置等。
    
    参数:
        filename: YAML配置文件路径
        args: 额外参数（目前未使用）
        verbose: 是否打印配置信息
        is_train: 是否为训练模式（决定phase字段）
    """

    def __init__(self, filename=None, args=None, verbose=False, is_train=True):
        super(Config, self).__init__()
        # ===== 设置默认参数 =====
        # 用于设置较大默认值的常量（表示"无限大"或"不触发"）
        large_number = 1000000000

        # --- 日志相关参数 ---
        self.snapshot_save_iter = large_number        # 模型保存的迭代间隔
        self.snapshot_save_epoch = large_number       # 模型保存的epoch间隔
        self.snapshot_save_start_iter = 0             # 开始保存模型的迭代数
        self.snapshot_save_start_epoch = 0            # 开始保存模型的epoch数
        self.image_save_iter = large_number           # 图像保存的迭代间隔
        self.eval_epoch = large_number                # 评估的epoch间隔
        self.start_eval_epoch = large_number          # 开始评估的epoch数
        self.eval_epoch = large_number                # 评估的epoch间隔（重复设置）
        self.max_epoch = large_number                 # 最大训练epoch数
        self.max_iter = large_number                  # 最大训练迭代数
        self.logging_iter = 100                       # 日志打印的迭代间隔
        self.image_to_tensorboard=False               # 是否将图像写入TensorBoard
        self.which_iter = 0 # args.which_iter         # 指定加载哪个迭代的模型
        self.resume = False                           # 是否从断点恢复训练

        self.checkpoints_dir = '/Users/shadowcun/Downloads/'  # 模型检查点保存目录
        self.name = 'face'                            # 实验名称
        self.phase = 'train' if is_train else 'test'  # 当前阶段：训练或测试

        # --- 网络结构配置 ---
        self.gen = AttrDict(type='generators.dummy')  # 生成器配置
        self.dis = AttrDict(type='discriminators.dummy')  # 判别器配置

        # --- 优化器配置 ---
        # 生成器优化器（Adam优化器）
        self.gen_optimizer = AttrDict(type='adam',
                                    lr=0.0001,                    # 学习率
                                    adam_beta1=0.0,               # Adam一阶矩衰减率
                                    adam_beta2=0.999,             # Adam二阶矩衰减率
                                    eps=1e-8,                     # 数值稳定性小量
                                    lr_policy=AttrDict(iteration_mode=False,  # 学习率调度策略
                                                    type='step',
                                                    step_size=large_number,
                                                    gamma=1))
        # 判别器优化器（Adam优化器）
        self.dis_optimizer = AttrDict(type='adam',
                                lr=0.0001,
                                adam_beta1=0.0,
                                adam_beta2=0.999,
                                eps=1e-8,
                                lr_policy=AttrDict(iteration_mode=False,
                                                   type='step',
                                                   step_size=large_number,
                                                   gamma=1))
        # --- 数据配置 ---
        # 训练数据配置
        self.data = AttrDict(name='dummy',
                             type='datasets.images',
                             num_workers=0)                  # 数据加载器工作进程数
        # 测试数据配置
        self.test_data = AttrDict(name='dummy',
                                  type='datasets.images',
                                  num_workers=0,
                                  test=AttrDict(is_lmdb=False,    # 是否使用LMDB数据格式
                                                roots='',          # 数据路径
                                                batch_size=1))     # 测试batch大小
        # --- 训练器配置 ---
        self.trainer = AttrDict(
            model_average=False,                           # 是否使用模型平均
            model_average_beta=0.9999,                     # 模型平均的衰减率
            model_average_start_iteration=1000,            # 开始模型平均的迭代数
            model_average_batch_norm_estimation_iteration=30,  # 模型平均中BN统计量估计的迭代数
            model_average_remove_sn=True,                  # 模型平均时是否移除谱归一化
            image_to_tensorboard=False,                    # 是否将图像写入TensorBoard
            hparam_to_tensorboard=False,                   # 是否将超参数写入TensorBoard
            distributed_data_parallel='pytorch',           # 分布式训练框架
            delay_allreduce=True,                          # 是否延迟梯度归约
            gan_relativistic=False,                        # 是否使用相对GAN
            gen_step=1,                                    # 生成器训练步数
            dis_step=1)                                    # 判别器训练步数

        # --- CuDNN配置 ---
        self.cudnn = AttrDict(deterministic=False,         # 是否使用确定性算法
                              benchmark=True)               # 是否启用CuDNN benchmark模式

        # --- 其他配置 ---
        self.pretrained_weight = ''                        # 预训练权重路径
        self.inference_args = AttrDict()                   # 推理参数

        # ===== 从YAML文件加载配置并更新 =====
        # 确保配置文件存在
        assert os.path.exists(filename), 'File {} not exist.'.format(filename)
        # 配置YAML加载器，添加自定义浮点数正则解析器
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        try:
            with open(filename, 'r') as f:
                cfg_dict = yaml.load(f, Loader=loader)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        # 递归更新默认配置
        recursive_update(self, cfg_dict)

        # 将common配置同时应用到生成器和判别器
        if 'common' in cfg_dict:
            self.common = AttrDict(**cfg_dict['common'])
            self.gen.common = self.common
            self.dis.common = self.common


        if verbose:
            print(' config '.center(80, '-'))
            print(self.__repr__())
            print(''.center(80, '-'))


def rsetattr(obj, attr, val):
    """递归设置对象属性
    支持点号分隔的嵌套属性路径，例如 'a.b.c' 会递归访问 obj.a.b 并设置 c 的值。
    
    参数:
        obj: 目标对象
        attr: 点号分隔的属性路径字符串
        val: 要设置的值
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """递归获取对象属性
    支持点号分隔的嵌套属性路径，例如 'a.b.c' 会递归访问 obj.a.b.c。
    
    参数:
        obj: 目标对象
        attr: 点号分隔的属性路径字符串
        *args: 默认值（可选），当属性不存在时返回
    返回:
        属性值
    """

    def _getattr(obj, attr):
        """获取单个属性的辅助函数"""
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def recursive_update(d, u):
    """递归更新AttrDict
    将字典u中的值递归地更新到字典d中。
    对于嵌套的字典（Mapping），递归更新；
    对于列表/元组中的字典元素，转换为AttrDict；
    其他类型直接赋值。
    
    参数:
        d: 被更新的AttrDict
        u: 用于更新的字典
    返回:
        更新后的AttrDict
    """
    for key, value in u.items():
        if isinstance(value, collections.abc.Mapping):
            # 递归更新嵌套字典
            d.__dict__[key] = recursive_update(d.get(key, AttrDict({})), value)
        elif isinstance(value, (list, tuple)):
            if isinstance(value[0], dict):
                # 将列表中的字典转换为AttrDict
                d.__dict__[key] = [AttrDict(item) for item in value]
            else:
                d.__dict__[key] = value
        else:
            d.__dict__[key] = value
    return d
