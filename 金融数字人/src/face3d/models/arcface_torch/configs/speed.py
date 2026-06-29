"""
速度测试配置文件
用于测试模型训练速度的最小化配置。
使用合成数据集（synthetic）和 ResNet-50 网络，仅包含 100 万类别。
此配置不进行验证，专注于测量训练吞吐量。
"""

# 导入 EasyDict 用于创建字典风格的配置对象
from easydict import EasyDict as edict

# 用于测试训练速度的配置（speed test configuration）

# 创建配置对象
config = edict()
config.loss = "arcface"                     # 损失函数类型：ArcFace（角度间隔边际损失）
config.network = "r50"                      # 骨干网络架构：ResNet-50
config.resume = False                       # 是否从检查点恢复训练
config.output = None                        # 输出目录路径
config.embedding_size = 512                 # 人脸嵌入特征向量的维度
config.sample_rate = 1.0                    # 数据采样率（100%，使用全部数据）
config.fp16 = True                          # 启用混合精度训练
config.momentum = 0.9                       # SGD 动量参数
config.weight_decay = 5e-4                  # 权重衰减系数
config.batch_size = 128                     # 每 GPU 批量大小
config.lr = 0.1                             # 初始学习率（batch size 为 512 时的参考值）

config.rec = "synthetic"                    # 训练数据集路径（synthetic = 合成数据集，用于速度测试）
config.num_classes = 100 * 10000            # 人脸身份类别数：100万（100 × 10000）
config.num_epoch = 30                       # 总训练轮数
config.warmup_epoch = -1                    # 不使用学习率预热
config.decay_epoch = [10, 16, 22]           # 学习率衰减轮数列表
config.val_targets = []                     # 验证目标列表（空 = 不验证，纯粹测速度）
