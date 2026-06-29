"""
ArcFace模型骨干网络初始化模块
本模块是backbones包的入口文件，提供模型工厂函数get_model()，
根据传入的模型名称返回对应的骨干网络实例。

支持的骨干网络类型：
- iresnet系列（r18, r34, r50, r100, r200, r2060）：改进的ResNet网络
- mbf：MobileFaceNet轻量级人脸网络
"""
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf


def get_model(name, **kwargs):
    """
    模型工厂函数，根据名称返回对应的骨干网络模型
    
    参数:
        name (str): 模型名称，支持：
            - "r18": IResNet-18（18层改进ResNet）
            - "r34": IResNet-34（34层改进ResNet）
            - "r50": IResNet-50（50层改进ResNet）
            - "r100": IResNet-100（100层改进ResNet）
            - "r200": IresNet-200（200层改进ResNet）
            - "r2060": IResNet-2060（2060层超深网络，含梯度检查点优化）
            - "mbf": MobileFaceNet（轻量级移动端人脸网络）
        **kwargs: 传递给模型的额外参数
    返回:
        对应的骨干网络模型实例
    """
    # ResNet系列模型
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)
    # MobileFaceNet轻量级网络
    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)         # 是否使用半精度浮点数
        num_features = kwargs.get("num_features", 512)  # 输出特征维度
        return get_mbf(fp16=fp16, num_features=num_features)
    else:
        raise ValueError()
