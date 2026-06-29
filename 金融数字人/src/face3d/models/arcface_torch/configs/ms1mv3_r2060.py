"""
MS1MV3 数据集 + R2060 网络配置文件
用于在 MS1MV3 人脸数据集上训练 R2060 骨干网络的配置。
R2060 是一种定制的深度残差网络架构（2060 可能表示其特殊结构编号）。
该配置使用较小的批量大小（64），适合显存较小的 GPU 环境。
"""

# 优化提示：使用 tmpfs 加速训练
# 本项目服务器内存为 256G
# 可通过挂载 tmpfs 将 140G 内存作为临时存储加速 I/O：
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

# 导入 EasyDict 用于创建字典风格的配置对象
from easydict import EasyDict as edict

# 创建配置对象
config = edict()
config.loss = "arcface"                     # 损失函数类型：ArcFace（角度间隔边际损失）
config.network = "r2060"                    # 骨干网络架构：R2060（定制深度残差网络）
config.resume = False                       # 是否从检查点恢复训练
config.output = None                        # 输出目录路径
config.embedding_size = 512                 # 人脸嵌入特征向量的维度
config.sample_rate = 1.0                    # 数据采样率（100%，使用全部数据）
config.fp16 = True                          # 启用混合精度训练
config.momentum = 0.9                       # SGD 动量参数
config.weight_decay = 5e-4                  # 权重衰减系数
config.batch_size = 64                      # 每 GPU 批量大小（较小，适合显存有限的 GPU）
config.lr = 0.1                             # 初始学习率（batch size 为 512 时的参考值）

config.rec = "/train_tmp/ms1m-retinaface-t1"  # 训练数据集路径（MS1MV3）
config.num_classes = 93431                  # 人脸身份类别数：93,431
config.num_image = 5179510                  # 训练图片总数：约 518 万张
config.num_epoch = 25                       # 总训练轮数
config.warmup_epoch = -1                    # 不使用学习率预热
config.decay_epoch = [10, 16, 22]           # 学习率衰减轮数列表
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]  # 验证数据集列表
