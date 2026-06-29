"""
Glint360K 数据集 + ResNet-34 网络配置文件
用于在 Glint360K 人脸数据集上训练 ResNet-34 骨干网络的配置。
ResNet-34 是 34 层的中等规模残差网络，在速度和精度之间取得平衡。
"""

# 优化提示：使用 tmpfs 加速训练
# 本项目服务器内存为 256G
# 可通过挂载 tmpfs 将 140G 内存作为临时存储加速 I/O：
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

# 导入 EasyDict 用于创建字典风格的配置对象
from easydict import EasyDict as edict

# 创建配置对象
config = edict()
config.loss = "cosface"                     # 损失函数类型：CosFace（余弦间隔边际损失）
config.network = "r34"                      # 骨干网络架构：ResNet-34（34层残差网络）
config.resume = False                       # 是否从检查点恢复训练
config.output = None                        # 输出目录路径
config.embedding_size = 512                 # 人脸嵌入特征向量的维度
config.sample_rate = 1.0                    # 数据采样率（100%，使用全部数据）
config.fp16 = True                          # 启用混合精度训练
config.momentum = 0.9                       # SGD 动量参数
config.weight_decay = 5e-4                  # 权重衰减系数
config.batch_size = 128                     # 每 GPU 批量大小
config.lr = 0.1                             # 初始学习率（batch size 为 512 时的参考值）

config.rec = "/train_tmp/glint360k"         # 训练数据集路径
config.num_classes = 360232                 # 人脸身份类别数：360,232
config.num_image = 17091657                 # 训练图片总数：约 1709 万张
config.num_epoch = 20                       # 总训练轮数
config.warmup_epoch = -1                    # 不使用学习率预热
config.decay_epoch = [8, 12, 15, 18]        # 学习率衰减轮数列表
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]  # 验证数据集列表
