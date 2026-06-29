"""
超参数配置模块
定义语音合成模型所需的各类超参数，包括Mel频谱图参数、音频处理参数、信号归一化参数等。
提供HParams类用于超参数的存储和访问，并预定义了一组默认超参数值。
"""


class HParams:
    """
    超参数管理类。
    
    该类提供了一种灵活的方式来存储和访问超参数。
    通过关键字参数初始化，支持通过属性方式访问参数值，
    也支持动态修改参数。
    
    使用示例：
        hp = HParams(lr=0.001, batch_size=32)
        print(hp.lr)  # 输出: 0.001
        hp.set_hparam('lr', 0.01)  # 修改学习率
    """
    def __init__(self, **kwargs):
        """
        初始化超参数。
        
        参数：
            **kwargs: 任意关键字参数，每个键值对代表一个超参数
        """
        # 使用字典存储所有超参数
        self.data = {}

        # 遍历所有传入的关键字参数，存入data字典
        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        """
        通过属性方式访问超参数值。
        
        例如：hparams.num_mels 会返回对应的值。
        
        参数：
            key (str): 超参数名称
        
        返回：
            对应超参数的值
        
        异常：
            AttributeError: 当请求的超参数不存在时抛出
        """
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        """
        设置或修改指定的超参数值。
        
        参数：
            key (str): 超参数名称
            value: 超参数的新值
        """
        self.data[key] = value


# 默认超参数配置
# 以下是语音合成/TTS模型常用的超参数默认值
hparams = HParams(
    # ===== Mel频谱图相关参数 =====
    num_mels=80,  # Mel频谱图的通道数（Mel滤波器组的数量），同时也是局部条件维度
    #  网络参数
    rescale=True,  # 是否在预处理之前对音频进行重缩放（归一化）
    rescaling_max=0.9,  # 重缩放的最大值，音频信号会归一化到[-0.9, 0.9]范围

    # ===== STFT与相位重建参数 =====
    # 使用LWS（https://github.com/Jonathan-LeRoux/lws）进行STFT和相位重建
    # 推荐设置为True以配合 https://github.com/r9y9/wavenet_vocoder 使用
    # 注意：当n_fft不是hop_size的整数倍时不工作！
    use_lws=False,

    # ===== FFT与窗口参数 =====
    n_fft=800,  # FFT窗口大小，多余的窗口位置用0填充（零填充）以匹配此参数
    hop_size=200,  # 帧移大小，对于16000Hz采样率，200帧=12.5毫秒（0.0125 * 采样率）
    win_size=800,  # 窗口大小，对于16000Hz采样率，800帧=50毫秒（若为None则等于n_fft）
    sample_rate=16000,  # 音频采样率：16000Hz（与librispeech数据集一致）

    frame_shift_ms=None,  # 帧移（毫秒），可替代hop_size参数（推荐值：12.5毫秒）

    # ===== Mel频谱图归一化与缩放参数 =====
    signal_normalization=True,  # 是否将Mel频谱图归一化到预定义的范围
    allow_clipping_in_normalization=True,  # 是否允许在归一化过程中裁剪（仅当signal_normalization=True时有效）
    symmetric_mels=True,
    # 是否将数据对称缩放到0附近范围（同时将输出范围乘以2）
    # 对称处理可以使模型收敛更快、更稳定
    max_abs_value=4.,
    # 数据的最大绝对值。若对称模式，数据范围为[-max, max]；否则为[0, max]
    # 注意：不宜过大以避免梯度爆炸，也不宜过小以保证快速收敛
    # 贡献者：@begeekmyfriend

    # ===== 预加重滤波器参数 =====
    # 预加重（Pre-Emphasis）：降低频谱噪声，提升模型确定性，
    # 同时有助于更好的Griffin-Lim相位重建
    preemphasize=True,  # 是否应用预加重滤波器
    preemphasis=0.97,  # 预加重滤波器系数

    # ===== 频率范围限制 =====
    min_level_db=-100,  # 最小分贝级别
    ref_level_db=20,    # 参考分贝级别
    fmin=55,  # 最小频率。若说话人为男性设为55，女性设为95可更好地去噪
    # （根据数据集测试确定。音高范围：男性约[65, 260]，女性约[100, 525]）
    fmax=7600,  # 最大频率，根据数据特点可适当增减

)


def hparams_debug_string():
    """
    生成超参数的调试打印字符串。
    
    将所有超参数按名称排序后格式化输出，
    方便在训练开始前检查超参数配置是否正确。
    
    返回：
        str: 格式化的超参数字符串，每行一个参数
    """
    # 获取所有超参数的字典
    values = hparams.values()
    # 将每个超参数格式化为 "  参数名: 参数值" 的字符串列表
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    # 返回完整的调试字符串
    return "Hyperparameters:\n" + "\n".join(hp)
