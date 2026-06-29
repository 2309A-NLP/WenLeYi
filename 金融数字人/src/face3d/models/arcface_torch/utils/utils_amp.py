"""
混合精度训练（AMP）辅助工具

该模块提供了混合精度训练（Automatic Mixed Precision）的辅助功能，
主要包括：
1. 多设备张量复制器（_MultiDeviceReplicator）
2. 带最大缩放限制的梯度缩放器（MaxClipGradScaler）

混合精度训练通过使用 FP16 半精度浮点数加速训练并减少显存占用，
同时使用梯度缩放器（GradScaler）避免梯度下溢问题。
"""

from typing import Dict, List

import torch

# 兼容不同 PyTorch 版本的 Iterable 类型
if torch.__version__ < '1.9':
    Iterable = torch._six.container_abcs.Iterable
else:
    import collections
    Iterable = collections.abc.Iterable
from torch.cuda.amp import GradScaler


class _MultiDeviceReplicator(object):
    """
    多设备张量复制器
    
    惰性地将主张量的副本分发到指定的设备（GPU）。
    每个设备的副本会被缓存，避免重复复制。
    
    这是 MaxClipGradScaler 的内部辅助类，用于在多 GPU 训练时
    将缩放因子分发到各设备。
    
    属性:
        master (torch.Tensor): 主张量（位于 GPU 上）
        _per_device_tensors (Dict): 各设备的张量缓存字典
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        """
        初始化复制器
        
        参数:
            master_tensor: 主张量，必须在 CUDA 设备上
        """
        assert master_tensor.is_cuda
        self.master = master_tensor
        self._per_device_tensors: Dict[torch.device, torch.Tensor] = {}

    def get(self, device) -> torch.Tensor:
        """
        获取指定设备上的张量副本
        
        如果该设备尚未缓存副本，则创建并缓存。
        
        参数:
            device: 目标设备
            
        返回:
            该设备上的张量副本
        """
        retval = self._per_device_tensors.get(device, None)
        if retval is None:
            retval = self.master.to(device=device, non_blocking=True, copy=True)
            self._per_device_tensors[device] = retval
        return retval


class MaxClipGradScaler(GradScaler):
    """
    带最大缩放限制的梯度缩放器
    
    继承自 PyTorch 的 GradScaler，增加了最大缩放因子限制功能。
    当缩放因子达到 max_scale 时，停止增长；超过时，强制截断到 max_scale。
    
    这可以防止在训练后期缩放因子过大导致的数值不稳定问题。
    
    参数:
        init_scale (float): 初始缩放因子
        max_scale (float): 最大缩放因子限制
        growth_interval (int): 缩放因子增长间隔（步数），默认 100
    """
    def __init__(self, init_scale, max_scale: float, growth_interval=100):
        GradScaler.__init__(self, init_scale=init_scale, growth_interval=growth_interval)
        self.max_scale = max_scale

    def scale_clip(self):
        """
        控制缩放因子的增长行为
        
        - 如果缩放因子等于 max_scale：停止增长（growth_factor=1）
        - 如果缩放因子小于 max_scale：允许增长（growth_factor=2）
        - 如果缩放因子大于 max_scale：截断到 max_scale（growth_factor=1）
        """
        if self.get_scale() == self.max_scale:
            self.set_growth_factor(1)
        elif self.get_scale() < self.max_scale:
            self.set_growth_factor(2)
        elif self.get_scale() > self.max_scale:
            self._scale.fill_(self.max_scale)
            self.set_growth_factor(1)

    def scale(self, outputs):
        """
        对输出张量进行缩放
        
        将梯度乘以缩放因子，以防止 FP16 训练中的梯度下溢。
        支持单张量和多设备张量的情况。
        
        参数:
            outputs: 待缩放的张量或张量可迭代对象
            
        返回:
            缩放后的输出
        """
        if not self._enabled:
            return outputs
        self.scale_clip()
        # 单张量的快速路径
        if isinstance(outputs, torch.Tensor):
            assert outputs.is_cuda
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            assert self._scale is not None
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # 多张量的复杂处理路径
        stash: List[_MultiDeviceReplicator] = []  # holds a reference that can be overwritten by apply_scale

        def apply_scale(val):
            if isinstance(val, torch.Tensor):
                assert val.is_cuda
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    assert self._scale is not None
                    stash.append(_MultiDeviceReplicator(self._scale))
                return val * stash[0].get(val.device)
            elif isinstance(val, Iterable):
                iterable = map(apply_scale, val)
                if isinstance(val, list) or isinstance(val, tuple):
                    return type(val)(iterable)
                else:
                    return iterable
            else:
                raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)
