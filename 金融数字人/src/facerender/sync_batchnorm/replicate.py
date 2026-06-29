# -*- coding: utf-8 -*-
# File   : replicate.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

# 同步BatchNorm的模型复制模块
# 本文件实现了支持回调机制的数据并行类，
# 使得模型在复制到多个GPU时能够执行自定义的初始化回调，
# 用于建立主-从设备间的同步通信连接。

import functools

from torch.nn.parallel.data_parallel import DataParallel

__all__ = [
    'CallbackContext',
    'execute_replication_callbacks',
    'DataParallelWithCallback',
    'patch_replication_callback'
]


class CallbackContext(object):
    """回调上下文对象
    在模型复制过程中，用于在不同设备上的相同子模块之间共享信息。
    每个子模块的所有副本共享同一个CallbackContext实例。
    """
    pass


def execute_replication_callbacks(modules):
    """执行模型复制回调
    在原始DataParallel复制模型后，对每个模块执行
    __data_parallel_replicate__ 回调函数。
    
    该回调函数允许模块在复制到不同设备后进行初始化，
    例如建立同步BatchNorm的主-从通信连接。
    
    重要保证：主设备（第一个副本）的回调会先于任何从设备的回调执行。
    
    参数:
        modules: 通过DataParallel复制得到的模块列表（每个元素是一个完整模型副本）
    """
    master_copy = modules[0]
    # 统计主副本中的子模块数量
    nr_modules = len(list(master_copy.modules()))
    # 为每个子模块创建一个共享的上下文对象
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    # 遍历所有副本和子模块，执行回调
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                # 调用模块的复制回调，传入共享上下文和当前副本ID
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """带复制回调的数据并行类
    扩展PyTorch的DataParallel，在模型复制后执行
    __data_parallel_replicate__ 回调函数。
    
    这使得同步BatchNorm等需要在复制时初始化通信的模块
    能够正确地建立主-从设备间的连接。
    
    使用示例:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ 将被自动调用
    """

    def replicate(self, module, device_ids):
        """重写复制方法
        在调用父类的replicate后，额外执行回调函数。
        
        参数:
            module: 要复制的模块
            device_ids: 目标设备ID列表
        返回:
            modules: 复制后的模块列表（已执行回调）
        """
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def patch_replication_callback(data_parallel):
    """猴子补丁：为现有DataParallel对象添加复制回调
    通过替换replicate方法，为已有的DataParallel对象添加回调功能。
    适用于自定义DataParallel实现的场景。
    
    使用示例:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # 这等价于:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    
    参数:
        data_parallel: 已有的DataParallel对象
    """
    assert isinstance(data_parallel, DataParallel)

    # 保存原始的replicate方法
    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        """新的replicate方法，在原始方法基础上添加回调执行"""
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    # 替换replicate方法
    data_parallel.replicate = new_replicate
