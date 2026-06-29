"""
safetensors辅助工具模块
本模块提供从safetensors检查点中提取特定模型权重的辅助函数。

safetensors格式的模型权重通常带有模块前缀（如'generator.conv1.weight'），
此函数可以提取并去除指定前缀，方便加载到对应的模型中。
"""


def load_x_from_safetensor(checkpoint, key):
    """
    从safetensors检查点中提取指定模块的权重
    
    该函数遍历检查点中所有的键值对，筛选出包含指定key的权重，
    并去除key前缀后返回新的权重字典，可直接用于model.load_state_dict()
    
    参数:
        checkpoint (dict): safetensors检查点字典，键为权重名称，值为张量
        key (str): 要提取的模块名称前缀（如'generator'、'kp_detector'等）
    返回:
        x_generator (dict): 提取并去除前缀后的权重字典
    示例:
        >>> checkpoint = {'generator.conv1.weight': tensor1, 'generator.bn1.weight': tensor2}
        >>> load_x_from_safetensor(checkpoint, 'generator')
        {'conv1.weight': tensor1, 'bn1.weight': tensor2}
    """
    x_generator = {}
    for k,v in checkpoint.items():
        if key in k:
            x_generator[k.replace(key+'.', '')] = v  # 去除模块前缀
    return x_generator
