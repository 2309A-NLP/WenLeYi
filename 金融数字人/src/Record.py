"""
录音模块
提供基于PyAudio的麦克风录音功能，支持静音自动检测与停止录音。
录音结果以WAV格式保存到本地文件。
"""

import pyaudio
import wave
import audioop

class Record:
    """
    录音类，封装了麦克风录音的完整流程。
    
    功能特点：
        - 通过麦克风实时录制音频
        - 自动检测静音并停止录音（静音检测基于RMS能量计算）
        - 录音结果保存为标准WAV文件
    
    使用示例：
        recorder = Record("output.wav", silence_threshold=500)
        recorder.start_recording()
    """
    def __init__(self, output_file, silence_threshold=1000):
        """
        初始化录音器。
        
        参数：
            output_file (str): 录音保存的WAV文件路径
            silence_threshold (int): 静音检测阈值，RMS能量低于此值视为静音。
                值越小越灵敏（更容易判定为静音），默认1000
        """
        # 保存输出文件路径
        self.output_file = output_file
        # 保存静音阈值
        self.silence_threshold = silence_threshold

        # 设置录音参数
        # 音频格式：16位有符号整数（每个采样点2字节）
        self.format = pyaudio.paInt16
        # 声道数：1表示单声道（mono）
        self.channels = 1
        # 采样率：44100Hz，即每秒采集44100个样本点
        self.sample_rate = 44100
        # 每次从音频流读取的帧数（缓冲区大小），1024帧
        self.chunk = 1024
        # 初始化PyAudio实例，用于管理音频设备
        self.audio = pyaudio.PyAudio()

    def start_recording(self):
        """
        开始录音。
        
        录音流程：
            1. 打开音频输入流
            2. 循环读取音频数据
            3. 计算每段音频的RMS能量
            4. 当连续静音超过1秒时自动停止录音
            5. 将录制的音频帧保存为WAV文件
        """
        # 打开音频输入流，配置格式、声道、采样率等参数
        stream = self.audio.open(format=self.format,
                                channels=self.channels,
                                rate=self.sample_rate,
                                input=True,               # 表示这是一个输入（录音）流
                                frames_per_buffer=self.chunk)  # 每次读取的帧数

        print("开始录音...")

        frames = []          # 存储所有录制的音频帧数据
        silence_counter = 0  # 静音帧计数器，用于判断是否持续静音

        # 录制音频的主循环
        while True:
            # 从音频流中读取一块数据（chunk帧）
            data = stream.read(self.chunk)
            frames.append(data)

            # 计算音频能量（RMS，均方根值），2表示样本宽度为2字节（16位）
            # RMS值越大表示音量越大，越小表示越安静
            rms = audioop.rms(data, 2)  # 2表示样本的宽度为2字节（16位）

            if rms < self.silence_threshold:
                # 如果当前帧的RMS低于阈值，视为静音，静音计数器+1
                silence_counter += 1
            else:
                # 如果检测到声音（非静音），重置静音计数器
                silence_counter = 0

            # 如果连续100个chunk都检测到静音（约1秒），则认为录音结束
            # 假设每个chunk为10毫秒，连续1秒的静音
            if silence_counter >= 100:  # 假设每个chunk为10毫秒，连续1秒的静音
                break

        print("录音结束.")

        # 关闭音频流，释放资源
        stream.stop_stream()
        stream.close()
        self.audio.terminate()

        # 将录制的音频帧数据保存为WAV格式文件
        with wave.open(self.output_file, 'wb') as wf:
            # 设置声道数（与录音参数一致）
            wf.setnchannels(self.channels)
            # 设置采样宽度（每个采样点的字节数，16位=2字节）
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            # 设置采样率
            wf.setframerate(self.sample_rate)
            # 将所有帧数据合并并写入文件
            wf.writeframes(b''.join(frames))


def test():
    """
    录音功能测试函数。
    
    创建一个Record对象，配置输出文件和静音阈值，
    然后执行录音操作。录音会在检测到约1秒静音后自动停止。
    """
    # 创建Record对象并开始录音
    output_file = "recording.wav"  # 输出文件名
    silence_threshold = 500  # 静音阈值

    recorder = Record(output_file, silence_threshold)
    recorder.start_recording()

# test()
