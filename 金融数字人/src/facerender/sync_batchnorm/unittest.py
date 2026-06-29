# -*- coding: utf-8 -*-
# File   : unittest.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

# 同步BatchNorm的单元测试辅助模块
# 本文件提供了用于测试同步BatchNorm层的辅助工具函数和测试用例基类。

import unittest

import numpy as np
from torch.autograd import Variable


def as_numpy(v):
    """将PyTorch张量或Variable转换为NumPy数组
    处理Variable和普通张量两种情况，
    自动将数据移到CPU并转换为NumPy格式。
    
    参数:
        v: PyTorch张量或Variable对象
    返回:
        对应的NumPy数组
    """
    if isinstance(v, Variable):
        v = v.data
    return v.cpu().numpy()


class TorchTestCase(unittest.TestCase):
    """PyTorch测试用例基类
    扩展Python的unittest.TestCase，添加了张量比较方法。
    用于同步BatchNorm等功能的正确性验证。
    """
    def assertTensorClose(self, a, b, atol=1e-3, rtol=1e-3):
        """断言两个张量在数值上接近
        使用NumPy的allclose函数进行比较，
        支持绝对容差（atol）和相对容差（rtol）。
        
        参数:
            a: 第一个张量
            b: 第二个张量
            atol: 绝对容差（默认1e-3）
            rtol: 相对容差（默认1e-3）
        """
        npa, npb = as_numpy(a), as_numpy(b)
        self.assertTrue(
                np.allclose(npa, npb, atol=atol),
                'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())
        )
