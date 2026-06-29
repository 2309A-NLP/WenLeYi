# -*- coding: utf-8 -*-
# File   : batchnorm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

# 同步批量归一化核心实现
# 本文件实现了跨GPU同步的BatchNorm层，包括1D、2D和3D版本。
# 核心思想：在多GPU训练时，通过主-从（Master-Slave）通信机制，
# 聚合所有GPU上的统计量（均值和方差），实现全局同步归一化。

import collections

import torch
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

from .comm import SyncMaster

# 导出的公开接口
__all__ = ['SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d']


def _sum_ft(tensor):
    """沿第一维（batch维）和最后一维求和
    用于计算跨GPU的统计量聚合。
    参数:
        tensor: 输入张量
    返回:
        求和后的张量
    """
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """在第一维和最后一维添加新的维度
    用于广播操作，将统计量扩展到正确的维度以进行归一化计算。
    参数:
        tensor: 输入张量
    返回:
        扩展维度后的张量
    """
    return tensor.unsqueeze(0).unsqueeze(-1)


# 定义主-从通信的消息格式
# _ChildMessage: 从设备发送给主设备的统计量消息
_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
# _MasterMessage: 主设备返回给从设备的归一化参数消息
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(_BatchNorm):
    """同步批量归一化基类
    继承自PyTorch的_BatchNorm，扩展了跨GPU同步功能。
    
    工作原理：
    1. 非并行模式或评估模式时，使用PyTorch内置BatchNorm
    2. 并行训练模式时：
       a. 每个设备计算本地的sum和square-sum
       b. 主设备聚合所有设备的统计量
       c. 主设备计算全局均值和标准差
       d. 将结果广播回所有设备
    
    参数:
        num_features: 特征通道数
        eps: 数值稳定性小量（默认1e-5）
        momentum: 运动平均的动量系数（默认0.1）
        affine: 是否使用可学习的仿射参数（默认True）
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        # 创建同步主控对象，用于跨设备通信
        self._sync_master = SyncMaster(self._data_parallel_master)

        # 并行状态标志
        self._is_parallel = False       # 是否处于并行模式
        self._parallel_id = None        # 当前设备的并行ID
        self._slave_pipe = None         # 从设备通信管道

    def forward(self, input):
        # 如果不是并行计算或处于评估模式，直接使用PyTorch内置实现
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # 将输入reshape为 (B, C, -1) 格式
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # 计算本地的求和与平方和
        sum_size = input.size(0) * input.size(2)  # 总样本数 = batch_size * 空间大小
        input_sum = _sum_ft(input)                  # 各通道的求和
        input_ssum = _sum_ft(input ** 2)            # 各通道的平方和

        # 跨设备聚合统计量并广播
        if self._parallel_id == 0:
            # 主设备：收集所有设备的统计量，计算全局均值和标准差
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            # 从设备：发送本地统计量到主设备，接收全局均值和标准差
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # 使用全局统计量计算归一化输出
        if self.affine:
            # 使用可学习参数的仿射变换（融合乘法以提高速度）
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            # 无可学习参数的归一化
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # 恢复原始输入形状
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        """数据并行复制回调
        当使用DataParallel复制模型到多个GPU时，此方法被调用。
        主设备（copy_id==0）创建SyncMaster，从设备注册到主设备。
        
        参数:
            ctx: 共享上下文对象
            copy_id: 当前副本的设备ID（0为主设备）
        """
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 表示主设备
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            # 从设备向主设备注册，获得通信管道
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """主设备的数据并行处理函数
        聚合所有设备的求和与平方和统计量，计算全局均值和标准差，
        然后将结果广播回所有设备。
        
        参数:
            intermediates: 所有设备发送的中间结果列表
        返回:
            outputs: 发送回各设备的处理结果
        """
        # 按设备ID排序，使ReduceAdd操作更快
        # 感谢: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        # 提取需要归约的数据（sum和ssum）
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # 展平列表
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        # 计算总样本数
        sum_size = sum([i[1].sum_size for i in intermediates])
        # 使用PyTorch内置的归约操作，将所有设备的sum和ssum聚合到主设备
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        # 计算全局均值和标准差
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        # 将均值和标准差广播到所有设备
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """使用求和与平方和计算均值和标准差
        同时更新运动平均（running_mean和running_var）。
        
        参数:
            sum_: 各通道的求和值
            ssum: 各通道的平方和值
            size: 总样本数
        返回:
            mean: 全局均值
            inv_std: 全局标准差的倒数
        """
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size  # 计算均值
        sumvar = ssum - sum_ * mean  # 计算方差的分子
        unbias_var = sumvar / (size - 1)  # 无偏方差（用于更新running_var）
        bias_var = sumvar / size  # 有偏方差（用于归一化计算）

        # 使用动量更新运动平均
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        # 返回均值和标准差的倒数（clamp确保数值稳定性）
        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    """同步批量归一化1D版本
    对2D或3D输入（视为mini-batch）进行跨设备同步的批量归一化。
    
    数学公式:
        y = (x - mean[x]) / sqrt(Var[x] + epsilon) * gamma + beta
    
    与PyTorch内置BatchNorm1d的区别：
    - 内置版本：每个GPU独立计算统计量（可能导致不准确）
    - 同步版本：跨所有GPU计算全局统计量（更准确）
    
    当仅使用单GPU或CPU时，行为与PyTorch内置实现完全相同。
    
    输入形状: (N, C) 或 (N, C, L)
    输出形状: (N, C) 或 (N, C, L)（与输入相同）
    """

    def _check_input_dim(self, input):
        """检查输入维度是否为2D或3D"""
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    """同步批量归一化2D版本
    对4D输入（视为3D输入的mini-batch）进行跨设备同步的批量归一化。
    
    数学公式:
        y = (x - mean[x]) / sqrt(Var[x] + epsilon) * gamma + beta
    
    与PyTorch内置BatchNorm2d的区别：
    - 内置版本：每个GPU独立计算统计量
    - 同步版本：跨所有GPU计算全局统计量
    
    输入形状: (N, C, H, W)
    输出形状: (N, C, H, W)（与输入相同）
    """

    def _check_input_dim(self, input):
        """检查输入维度是否为4D"""
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    """同步批量归一化3D版本
    对5D输入（视为4D输入的mini-batch）进行跨设备同步的批量归一化。
    适用于3D卷积网络（如视频处理、3D医学图像等）。
    
    数学公式:
        y = (x - mean[x]) / sqrt(Var[x] + epsilon) * gamma + beta
    
    也称为体积批量归一化（Volumetric BatchNorm）或
    时空批量归一化（Spatio-temporal BatchNorm）。
    
    输入形状: (N, C, D, H, W)
    输出形状: (N, C, D, H, W)（与输入相同）
    """

    def _check_input_dim(self, input):
        """检查输入维度是否为5D"""
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)
