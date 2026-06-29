# audio.py - 音频处理工具模块
# 本模块提供音频的加载、保存、频谱图计算等核心功能
# 主要用于语音合成（TTS）流程中的音频预处理和后处理

import librosa
import librosa.filters
import numpy as np
# import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from src.utils.hparams import hparams as hp  # 导入超参数配置（采样率、帧移、梅尔滤波器参数等）


def load_wav(path, sr):
    """加载音频文件并重采样到指定采样率
    
    Args:
        path: 音频文件路径
        sr: 目标采样率
    
    Returns:
        numpy数组，包含音频波形数据
    """
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    """将音频波形数据保存为WAV文件
    
    先将浮点波形归一化到int16范围（-32768~32767），再写入文件。
    
    Args:
        wav: 音频波形numpy数组
        path: 输出文件路径
        sr: 采样率
    """
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))  # 归一化到int16范围
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))  # 写入16位WAV文件


def save_wavenet_wav(wav, path, sr):
    """使用librosa保存WAV文件（兼容WaveNet格式）"""
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    """对音频进行预加重处理
    
    预加重可以提升高频分量，增强语音信号的高频部分，
    常用于语音识别和语音合成的预处理。
    
    Args:
        wav: 输入音频波形
        k: 预加重系数（通常取0.97左右）
        preemphasize: 是否执行预加重
    
    Returns:
        预加重后的音频波形
    """
    if preemphasize:
        # 使用scipy的线性滤波器实现预加重：y[n] = x[n] - k*x[n-1]
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    """对音频进行反预加重处理（预加重的逆操作）
    
    用于从预加重后的信号恢复原始信号。
    
    Args:
        wav: 预加重后的音频波形
        k: 预加重系数
        inv_preemphasize: 是否执行反预加重
    
    Returns:
        反预加重后的音频波形
    """
    if inv_preemphasize:
        # 使用scipy的线性滤波器实现反预加重：y[n] = x[n] + k*y[n-1]
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():
    """获取帧移大小（hop_size）
    
    优先使用hop_size参数，如果未设置则根据帧移毫秒数和采样率计算。
    
    Returns:
        帧移大小（样本数）
    """
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None  # 确保帧移毫秒数已设置
        # 帧移(样本数) = 帧移(毫秒) / 1000 * 采样率
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size


def linearspectrogram(wav):
    """计算线性频谱图
    
    流程：预加重 -> STFT -> 幅度转dB -> 减去参考电平 -> 归一化（可选）
    
    Args:
        wav: 输入音频波形
    
    Returns:
        线性频谱图（dB域）
    """
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))  # 预加重后做短时傅里叶变换
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db  # 幅度转dB并减去参考电平
    
    if hp.signal_normalization:
        return _normalize(S)  # 信号归一化
    return S


def melspectrogram(wav):
    """计算梅尔频谱图
    
    梅尔频谱图在人耳听觉感知上更均匀，是语音合成模型常用的输入特征。
    
    流程：预加重 -> STFT -> 幅度转梅尔尺度 -> 幅度转dB -> 减去参考电平 -> 归一化（可选）
    
    Args:
        wav: 输入音频波形
    
    Returns:
        梅尔频谱图（dB域）
    """
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))  # 预加重后做STFT
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db  # 线性频谱转梅尔频谱再转dB
    
    if hp.signal_normalization:
        return _normalize(S)  # 信号归一化
    return S


def _lws_processor():
    """创建LWS（Learned Windowed STFT）处理器
    
    LWS是一种改进的STFT方法，适用于语音信号处理。
    """
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")


def _stft(y):
    """执行短时傅里叶变换（STFT）
    
    根据配置选择使用LWS或标准librosa的STFT。
    
    Args:
        y: 输入音频信号
    
    Returns:
        复数STFT结果
    """
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T  # 使用LWS模式
    else:
        # 使用标准librosa STFT
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)


##########################################################
# 以下函数仅在使用LWS时正确（曾长期影响WaveNet音质！）
def num_frames(length, fsize, fshift):
    """计算频谱图的时间帧数
    
    Args:
        length: 音频信号长度
        fsize: 帧大小（FFT窗口大小）
        fshift: 帧移大小
    
    Returns:
        时间帧数
    """
    pad = (fsize - fshift)  # 计算填充量
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """计算左右填充量
    
    用于确保音频长度恰好能被整除为整数帧。
    
    Args:
        x: 输入音频
        fsize: 帧大小
        fshift: 帧移大小
    
    Returns:
        (左填充量, 右填充量)
    """
    M = num_frames(len(x), fsize, fshift)  # 计算总帧数
    pad = (fsize - fshift)  # 基础填充量
    T = len(x) + 2 * pad  # 填充后的总长度
    r = (M - 1) * fshift + fsize - T  # 额外需要的右填充
    return pad, pad + r
##########################################################


def librosa_pad_lr(x, fsize, fshift):
    """使用librosa标准方式计算左右填充量
    
    与pad_lr不同，这是librosa的正确填充方式。
    
    Args:
        x: 输入音频
        fsize: 帧大小
        fshift: 帧移大小
    
    Returns:
        (左填充量, 右填充量)
    """
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# 频谱转换相关函数
_mel_basis = None  # 全局缓存的梅尔滤波器组矩阵


def _linear_to_mel(spectogram):
    """将线性频谱图转换为梅尔频谱图
    
    使用预构建的梅尔滤波器组矩阵进行矩阵乘法转换。
    
    Args:
        spectogram: 线性频谱图（幅度）
    
    Returns:
        梅尔频谱图
    """
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()  # 首次调用时构建梅尔滤波器组
    return np.dot(_mel_basis, spectogram)  # 矩阵乘法转换到梅尔尺度


def _build_mel_basis():
    """构建梅尔滤波器组矩阵
    
    梅尔滤波器组是一组三角形带通滤波器，模拟人耳对频率的非线性感知。
    
    Returns:
        梅尔滤波器组矩阵 (n_mels, n_fft/2+1)
    """
    assert hp.fmax <= hp.sample_rate // 2  # 确保最大频率不超过奈奎斯特频率
    return librosa.filters.mel(
        sr=hp.sample_rate,    # 采样率
        n_fft=hp.n_fft,       # FFT窗口大小
        n_mels=hp.num_mels,   # 梅尔滤波器数量
        fmin=hp.fmin,         # 最小频率
        fmax=hp.fmax          # 最大频率
    )


def _amp_to_db(x):
    """将幅度谱转换为分贝（dB）谱
    
    转换公式：dB = 20 * log10(amplitude)
    
    Args:
        x: 幅度值
    
    Returns:
        分贝值
    """
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))  # 计算最小电平阈值
    return 20 * np.log10(np.maximum(min_level, x))  # 取对数并限制最小值


def _db_to_amp(x):
    """将分贝（dB）谱转换回幅度谱
    
    转换公式：amplitude = 10^(dB/20)
    
    Args:
        x: 分贝值
    
    Returns:
        幅度值
    """
    return np.power(10.0, (x) * 0.05)


def _normalize(S):
    """对频谱图进行归一化处理
    
    将频谱值映射到[-max_abs_value, max_abs_value]或[0, max_abs_value]范围。
    支持对称梅尔和非对称梅尔两种模式。
    
    Args:
        S: 频谱图（dB域）
    
    Returns:
        归一化后的频谱图
    """
    if hp.allow_clipping_in_normalization:
        # 允许裁剪的归一化模式
        if hp.symmetric_mels:
            # 对称模式：归一化到 [-max_abs_value, max_abs_value]
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            # 非对称模式：归一化到 [0, max_abs_value]
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
    
    # 不允许裁剪时，先检查值范围是否合法
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))


def _denormalize(D):
    """对归一化后的频谱图进行反归一化
    
    将频谱值从归一化范围恢复到原始的dB范围。
    
    Args:
        D: 归一化后的频谱图
    
    Returns:
        反归一化后的频谱图（dB域）
    """
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
    
    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
