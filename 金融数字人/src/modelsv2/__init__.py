# V2版本模型包初始化文件
# 从 wav2lip_v2 模块导入 V2版本的 Wav2Lip 生成器模型和 Wav2Lip_disc_qual 判别器模型
# V2版本相比V1版本在网络结构上有所改进（如增加了更多编码器层和a_alpha音频强度控制参数）
from .wav2lip_v2 import Wav2Lip, Wav2Lip_disc_qual
# 从 syncnet 模块导入 V2版本的 SyncNet_color 音视频同步性判断网络
# V2版本的SyncNet结构与V1类似，增加了调试相关的import和部分中文注释
from .syncnet import SyncNet_color
