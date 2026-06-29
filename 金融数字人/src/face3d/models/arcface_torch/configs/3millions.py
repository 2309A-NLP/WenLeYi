"""
3millions 数据集配置文件
用于在300万级别合成数据集上训练 ArcFace 人脸识别模型的超参数配置。
该配置使用 ResNet-50 作为骨干网络，ArcFace 损失函数。
"""

# 导入 EasyDict 用于创建字典风格的配置对象
from easydict import EasyDict as edict

# 用于测试训练速度的配置（speed test configuration）

# 创建配置对象
config = edict()
config.loss = "arcface"                     # 损失函数类型：ArcFace（角度间隔边际损失）
config.network = "r50"                      # 骨干网络架构：ResNet-50
config.resume = False                       # 是否从检查点恢复训练（False 表示从头开始训练）
config.output = None                        # 输出目录路径（None 表示使用默认路径）
config.embedding_size = 512                 # 人脸嵌入特征向量的维度大小
config.sample_rate = 1.0                    # 数据采样率（1.0 表示使用全部数据）
config.fp16 = True                          # 是否启用混合精度训练（FP16 半精度浮点，加速训练并节省显存）
config.momentum = 0.9                       # SGD 优化器的动量参数
config.weight_decay = 5e-4                  # 权重衰减系数（L2 正则化，防止过拟合）
config.batch_size = 128                     # 每个 GPU 的批量大小
config.lr = 0.1                             # 初始学习率（当 batch size 为 512 时的参考值）

config.rec = "synthetic"                    # 训练数据集路径（synthetic 表示合成数据集）
config.num_classes = 300 * 10000            # 人脸身份类别数：300万（300 × 10000）
config.num_epoch = 30                       # 总训练轮数
config.warmup_epoch = -1                    # 学习率预热轮数（-1 表示不使用预热策略）
config.decay_epoch = [10, 16, 22]           # 学习率衰减的轮数列表（在第10、16、22轮衰减）
config.val_targets = []                     # 验证目标数据集列表（空表示不进行验证）
