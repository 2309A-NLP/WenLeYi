# -*- coding: utf-8 -*-
# File   : comm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

# 同步BatchNorm的通信模块
# 本文件实现了跨设备通信的基础设施，包括：
# 1. FutureResult: 线程安全的异步结果容器
# 2. SlavePipe: 主-从设备间的通信管道
# 3. SyncMaster: 同步主控对象，协调所有设备的通信

import queue
import collections
import threading

__all__ = ['FutureResult', 'SlavePipe', 'SyncMaster']


class FutureResult(object):
    """线程安全的异步结果容器
    实现了类似Java Future的模式，用于线程间的一对一通信。
    一个线程可以put结果，另一个线程可以get获取结果，
    如果结果还未就绪，get会阻塞等待。
    """

    def __init__(self):
        self._result = None          # 存储结果值
        self._lock = threading.Lock()  # 互斥锁
        self._cond = threading.Condition(self._lock)  # 条件变量

    def put(self, result):
        """放入结果并通知等待的线程
        参数:
            result: 要存储的结果值
        """
        with self._lock:
            assert self._result is None, 'Previous result hasn\'t been fetched.'
            self._result = result
            self._cond.notify()  # 唤醒等待的线程

    def get(self):
        """获取结果，如果尚未就绪则阻塞等待
        返回:
            存储的结果值
        """
        with self._lock:
            if self._result is None:
                self._cond.wait()  # 阻塞等待结果

            res = self._result
            self._result = None  # 获取后清空，支持一次性使用
            return res


# 主设备注册表：存储FutureResult对象
_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
# 从设备管道基础元组：包含标识符、队列和结果容器
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """主-从设备间的通信管道
    从设备通过此管道向主设备发送统计量，并接收主设备返回的全局参数。
    
    通信流程：
    1. 将本地统计量放入队列
    2. 等待主设备处理并返回结果
    3. 确认接收完成
    """

    def run_slave(self, msg):
        """从设备执行同步操作
        参数:
            msg: 要发送给主设备的消息（如本地统计量）
        返回:
            主设备处理后返回的结果
        """
        # 将消息放入队列发送给主设备
        self.queue.put((self.identifier, msg))
        # 阻塞等待主设备的处理结果
        ret = self.result.get()
        # 发送确认信号，表示已接收结果
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """同步主控对象
    协调多个设备之间的同步BatchNorm通信。
    
    工作流程：
    1. 模型复制阶段：各从设备调用register()获取SlavePipe
    2. 前向传播阶段：主设备调用run_master()，收集所有设备的统计量
    3. 结果分发阶段：主设备通过callback计算全局参数，分发回各设备
    """

    def __init__(self, master_callback):
        """
        参数:
            master_callback: 主设备的回调函数，
                在收集到所有从设备的消息后被调用，
                用于计算全局均值和标准差并分发回各设备。
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()           # 用于收集从设备消息的队列
        self._registry = collections.OrderedDict()  # 从设备注册表（有序）
        self._activated = False                # 是否已激活

    def __getstate__(self):
        """序列化状态（用于pickle）
        仅保存master_callback，其他运行时状态在反序列化后重新初始化。
        """
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        """反序列化状态
        从保存的状态中恢复master_callback并重新初始化。
        """
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """注册从设备
        在模型复制时，每个从设备调用此方法向主设备注册，
        获取一个SlavePipe用于后续通信。
        
        参数:
            identifier: 设备标识符（通常是设备ID）
        返回:
            SlavePipe: 用于与主设备通信的管道对象
        """
        if self._activated:
            # 如果已激活（新一轮训练开始），清空注册表和队列
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        # 创建FutureResult用于接收主设备返回的结果
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """主设备的主入口
        在每次前向传播时被主设备调用。
        收集所有设备（包括主设备自身）的消息，
        调用callback函数计算全局参数，然后分发回各设备。
        
        参数:
            master_msg: 主设备自身要发送的消息（统计量），
                将作为第一个消息传递给master_callback。
        返回:
            发送回主设备的处理结果
        """
        self._activated = True

        # 收集所有设备的消息（主设备的消息放在第一个）
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        # 调用回调函数处理所有设备的统计量
        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        # 将处理结果分发回各从设备
        for i, res in results:
            if i == 0:
                continue  # 跳过主设备
            self._registry[i].result.put(res)

        # 等待所有从设备确认接收
        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        # 返回主设备自己的结果
        return results[0][1]

    @property
    def nr_slaves(self):
        """获取已注册的从设备数量"""
        return len(self._registry)
