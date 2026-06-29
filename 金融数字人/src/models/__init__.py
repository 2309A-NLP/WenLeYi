# 模型包初始化文件
# 从 wav2lip 模块导入 Wav2Lip 生成器模型和 Wav2Lip_disc_qual 判别器模型
# Wav2Lip: 基于音频驱动生成嘴型的核心网络
# Wav2Lip_disc_qual: 用于质量判别的判别器网络
from .wav2lip import Wav2Lip, Wav2Lip_disc_qual
# 从 syncnet 模块导入 SyncNet_color 音视频同步性判断网络
# SyncNet_color: 用于判断音频和视频帧之间同步性的网络
from .syncnet import SyncNet_color
