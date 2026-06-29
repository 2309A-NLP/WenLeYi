"""
配置加载工具模块

该模块提供了配置文件的动态加载功能。
支持从 configs 目录下的 Python 配置文件中加载训练参数，
并自动合并基础配置（configs/base.py）和任务特定配置。

使用方法：
    cfg = get_config('configs/ms1mv3_r50')
    
    这会先加载 configs/base.py 作为基础配置，
    然后用 configs/ms1mv3_r50.py 的配置覆盖/合并。
"""

import importlib
import os.path as osp


def get_config(config_file):
    """
    动态加载并合并配置文件
    
    配置加载流程：
    1. 验证配置文件路径必须以 'configs/' 开头
    2. 加载基础配置（configs/base.py）
    3. 加载任务特定配置文件
    4. 用任务配置更新基础配置（覆盖相同键的值）
    5. 如果输出目录未指定，自动生成默认路径
    
    参数:
        config_file (str): 配置文件路径，如 'configs/ms1mv3_r50'
        
    返回:
        合并后的配置对象（EasyDict 格式）
    """
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    
    # 提取配置文件名（不含路径和扩展名）
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    
    # 加载基础配置
    config = importlib.import_module("configs.base")
    cfg = config.config
    
    # 加载并合并任务特定配置
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    
    # 如果输出目录未指定，使用默认路径：work_dirs/<配置名>
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name)
    return cfg
