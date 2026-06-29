# torchalign 的配置文件
# 使用 yacs 库定义模型的所有超参数配置
# 包含输入配置、骨干网络配置和热图头配置三个部分

from yacs.config import CfgNode as CN

# 创建全局配置节点
_C = CN()

### 输入配置（INPUT）###
# 定义图像输入相关的参数
_C.INPUT = CN()
_C.INPUT.SIZE = [256, 256]  # 模型输入图像尺寸，宽度x高度
_C.INPUT.SCALE = 1.25  # 边界框裁剪缩放因子，>1 表示裁剪区域比人脸框稍大
_C.INPUT.DATASET = 'WFLW'  # 训练使用的数据集名称（WFLW: 98点人脸关键点数据集）
_C.INPUT.BBOX = 'P1'  # 人脸检测器类型
_C.INPUT.FLIP = True  # 是否在推理时使用水平翻转增强（TTA，测试时增强）
# 关键点翻转索引映射表：当图像水平翻转时，关键点索引需要重新排列
# 这个映射将翻转后的关键点顺序恢复到与原始图像一致的对应关系
_C.INPUT.FLIP_ORDER = [
    32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
    14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0, 46, 45, 44,
    43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52, 53,
    54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61,
    60, 67, 66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91,
    90, 89, 88, 95, 94, 93, 97, 96
]

### 骨干网络配置（BACKBONE）###
# 定义特征提取网络的类型
_C.BACKBONE = CN()
_C.BACKBONE.ARCH = 'hrnet18'  # 骨干网络架构名称，可选 hrnet18s/hrnet18/hrnet32/mobilenetv2

### 热图头配置（HEATMAP）###
# 定义关键点检测头的参数
_C.HEATMAP = CN()
_C.HEATMAP.ARCH = 'HeatmapHead'  # 热图头类名
_C.HEATMAP.IN_CHANNEL = 270  # 输入通道数（来自骨干网络的特征维度）
_C.HEATMAP.PROJ_CHANNEL = 270  # 投影层通道数（中间层维度）
_C.HEATMAP.OUT_CHANNEL = 98  # 输出通道数，即关键点数量（WFLW数据集有98个关键点）
_C.HEATMAP.STRIDE = 4.0  # 热图到坐标的步长因子（热图分辨率与输入分辨率的比值）
_C.HEATMAP.ENCODER = 'Coordinate2BinaryHeatmap'  # 编码器：坐标到二值热图的转换方法
_C.HEATMAP.DECODER = 'BinaryHeatmap2Coordinate'  # 解码器：二值热图到坐标的转换方法
_C.HEATMAP.BLOCK = 'BinaryHeadBlock'  # 热图头的网络块类型
_C.HEATMAP.TOPK = 9  # 解码时选取top-k个最大值点进行加权平均

# 导出全局配置对象
cfg = _C
