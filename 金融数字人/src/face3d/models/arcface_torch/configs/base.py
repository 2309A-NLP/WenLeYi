"""
ArcFace训练基础配置模块
本模块定义了人脸识别模型训练的默认配置参数，使用EasyDict方便通过属性名访问配置项。

配置内容包括：
1. 损失函数类型（arcface/cosface）
2. 骨干网络类型（r50等）
3. 数据集相关参数（数据路径、类别数、图片数等）
4. 训练超参数（批量大小、学习率、训练轮数等）
5. 不同数据集的专用配置（emore、ms1m-retinaface-t1、glint360k、webface）

提示：使用tmpfs挂载可以加速训练数据读取
RAM=256G时可以使用：mount -t tmpfs -o size=140G tmpfs /train_tmp
"""
from easydict import EasyDict as edict

# 使用tmpfs内存文件系统可以加速训练
# 我们的服务器RAM为256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

# ========== 基础配置 ==========
config = edict()
config.loss = "arcface"          # 损失函数类型（arcface: 角间隔损失, cosface: 余弦间隔损失）
config.network = "r50"           # 骨干网络类型（r50: IResNet-50）
config.resume = False            # 是否从检查点恢复训练
config.output = "ms1mv3_arcface_r50"  # 输出目录名称

# ========== 数据集配置 ==========
config.dataset = "ms1m-retinaface-t1"  # 使用的数据集
config.embedding_size = 512      # 人脸特征嵌入维度
config.sample_rate = 1           # PartialFC采样率（1.0表示使用全部类别中心）
config.fp16 = False              # 是否使用半精度训练

# ========== 优化器参数 ==========
config.momentum = 0.9            # SGD优化器动量
config.weight_decay = 5e-4       # 权重衰减（L2正则化）
config.batch_size = 128          # 批量大小
config.lr = 0.1                  # 初始学习率（基准batch size为512）

# ========== 各数据集的专用配置 ==========

# MS-Celeb-1M (emore) 数据集配置
if config.dataset == "emore":
    config.rec = "/train_tmp/faces_emore"  # 数据集路径
    config.num_classes = 85742             # 人脸身份类别数
    config.num_image = 5822653            # 训练图片总数
    config.num_epoch = 16                 # 训练总轮数
    config.warmup_epoch = -1              # 预热轮数（-1表示不使用预热）
    config.decay_epoch = [8, 14, ]        # 学习率衰减的轮数
    config.val_targets = ["lfw", ]        # 验证集列表

# MS1MV3 (ms1m-retinaface-t1) 数据集配置（推荐使用）
elif config.dataset == "ms1m-retinaface-t1":
    config.rec = "/train_tmp/ms1m-retinaface-t1"  # 数据集路径
    config.num_classes = 93431            # 人脸身份类别数
    config.num_image = 5179510            # 训练图片总数
    config.num_epoch = 25                 # 训练总轮数
    config.warmup_epoch = -1              # 预热轮数
    config.decay_epoch = [11, 17, 22]     # 学习率衰减的轮数（分3次衰减）
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]  # 验证集列表（包含年龄变化）

# Glint360K大规模数据集配置
elif config.dataset == "glint360k":
    config.rec = "/train_tmp/glint360k"  # 数据集路径
    config.num_classes = 360232           # 人脸身份类别数（36万+类别）
    config.num_image = 17091657           # 训练图片总数（1700万+图片）
    config.num_epoch = 20                 # 训练总轮数
    config.warmup_epoch = -1              # 预热轮数
    config.decay_epoch = [8, 12, 15, 18]  # 学习率衰减的轮数（分4次衰减）
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]  # 验证集列表

# WebFace数据集配置
elif config.dataset == "webface":
    config.rec = "/train_tmp/faces_webface_112x112"  # 数据集路径
    config.num_classes = 10572            # 人脸身份类别数
    config.num_image = "forget"           # 图片数未知
    config.num_epoch = 34                 # 训练总轮数
    config.warmup_epoch = -1              # 预热轮数
    config.decay_epoch = [20, 28, 32]     # 学习率衰减的轮数
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]  # 验证集列表
