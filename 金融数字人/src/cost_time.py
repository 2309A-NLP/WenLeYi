"""
计时工具模块
提供装饰器函数，用于测量和打印函数运行时间，方便性能分析与调试
"""

import time

# 定义装饰器
def calculate_time(func):
    """
    计算函数运行时间的装饰器。
    
    使用方法：
        @calculate_time
        def my_function():
            ...
    
    该装饰器会在函数执行前后分别记录时间戳，
    并在函数执行完毕后打印出该函数的总运行时间（单位：秒）。
    
    参数：
        func: 被装饰的目标函数
    
    返回：
        wrapper: 包装后的函数，功能与原函数一致，但会额外打印运行时间
    """
    def wrapper(*args, **kwargs):
        # 记录函数开始执行时的时间戳（Unix时间，单位：秒）
        start_time = time.time()
        
        # 调用并执行原始函数，传入所有参数，获取返回值
        result = func(*args, **kwargs)
        
        # 记录函数执行完毕时的时间戳
        end_time = time.time()
        
        # 计算并打印函数的运行耗时（结束时间 - 开始时间）
        print(f"函数 {func.__name__} 运行时间： {end_time - start_time} 秒")
        
        # 返回原始函数的返回值，确保装饰器不影响原函数逻辑
        return result
    return wrapper
