"""
超参数管理模块 v2
本模块是hparams.py的简化版本，定义了HParams类和默认的超参数配置。
用于管理和访问音频处理和模型训练的超参数。
"""


class HParams:
    """超参数类，支持通过属性名直接访问参数值"""

    def __init__(self, **kwargs):
        """初始化超参数，将传入的关键字参数存储到data字典中"""
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        """属性访问方法，通过属性名获取对应的参数值"""
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        """设置或更新超参数"""
        self.data[key] = value


# 默认超参数配置
hparams = HParams(
    num_mels=80,  # 梅尔频谱图通道数和局部条件维度
    #  网络参数
    rescale=True,  # 是否在预处理前重新缩放音频
    rescaling_max=0.9,  # 重新缩放的最大值

    # 使用LWS进行STFT和相位重建
    # 推荐设置为True以配合https://github.com/r9y9/wavenet_vocoder使用
    # 注意：n_fft必须是hop_size的倍数！
    use_lws=False,

    n_fft=800,  # FFT窗口大小，多余的窗口用零填充以匹配此参数
    hop_size=200,  # 对于16000Hz采样率，200对应12.5毫秒(0.0125 * 采样率)
    win_size=800,  # 对于16000Hz采样率，800对应50毫秒(如果为None，则win_size = n_fft)
    sample_rate=16000,  # 采样率16000Hz（与librispeech数据集对应）

    frame_shift_ms=None,  # 可替代hop_size参数（推荐值：12.5）

    # 梅尔频谱图和线性频谱图的归一化/缩放和裁剪
    signal_normalization=True,
    # 是否将梅尔频谱图归一化到预定义范围（遵循以下参数）
    allow_clipping_in_normalization=True,  # 仅在signal_normalization=True时有效
    symmetric_mels=True,
    # 是否将数据缩放到以0为中心的对称范围（同时将输出范围乘以2，
    # 有利于更快更干净的收敛）
    max_abs_value=4.,
    # 数据的最大绝对值。如果对称，数据范围为[-max, max]，否则为[0, max]
    # 注意：不能太大以避免梯度爆炸，也不能太小以保证快速收敛

    # 预加重滤波（Lfilter：降低频谱噪声，有助于模型确定性，
    # 同时有助于更好的格拉姆-拉格朗日相位重建）
    preemphasize=True,  # 是否应用预加重滤波
    preemphasis=0.97,  # 滤波器系数

    # 频率范围限制
    min_level_db=-100,  # 最小分贝级别
    ref_level_db=20,   # 参考分贝级别
    fmin=55,           # 最小频率（男性设为55，女性设为95有助于去噪）
    fmax=7600,         # 最大频率（根据数据集调整）

)


def hparams_debug_string():
    """生成超参数的调试字符串，按字母顺序排列所有参数（排除sentences参数）"""
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)
