# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

# 同步批量归一化（Synchronized BatchNorm）模块
# 本模块实现了跨多个GPU的同步批量归一化层。
# 在多GPU训练时，PyTorch默认的BatchNorm仅在每个GPU上独立计算统计量，
# 这可能导致不准确，尤其是当每个GPU上的batch size较小时。
# 本模块通过跨GPU同步来计算全局的均值和方差，提高归一化精度。

# 导出同步BatchNorm的1D、2D、3D版本
from .batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
# 导出带有回调的数据并行类和补丁函数
from .replicate import DataParallelWithCallback, patch_replication_callback
