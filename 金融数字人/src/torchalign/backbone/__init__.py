# backbone 子模块的初始化文件
# 导入所有可用的骨干网络（特征提取器）
# hrnet: 高分辨率网络（High-Resolution Network），保持高分辨率特征表示
# mobilenet: 轻量级MobileNetV2网络，适合移动端和实时应用
from .hrnet import *
from .mobilenet import *
