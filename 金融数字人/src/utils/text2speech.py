# text2speech.py - 文本转语音模块
# 本模块封装了Coqui TTS库的文本转语音功能
# 用于将输入文本转换为语音音频文件，作为数字人说话的音源

import os
import tempfile  # 用于创建临时文件
from TTS.api import TTS  # Coqui TTS库，支持多种TTS模型


class TTSTalker():
    """文本转语音说话人类
    
    封装了Coqui TTS库的功能，提供文本到语音的转换接口。
    使用预训练的TTS模型将文本合成为语音音频。
    """

    def __init__(self) -> None:
        """初始化TTS说话人
        
        自动选择并加载第一个可用的TTS模型。
        """
        model_name = TTS.list_models()[0]  # 获取第一个可用的预训练模型名称
        self.tts = TTS(model_name)  # 加载TTS模型

    def test(self, text, language='en'):
        """将文本转换为语音并保存为WAV文件
        
        Args:
            text: 需要转换的文本内容
            language: 语言代码（默认'en'英语）
        
        Returns:
            生成的WAV音频文件的路径
        """
        # 创建临时WAV文件（不自动删除，以便后续使用）
        tempf  = tempfile.NamedTemporaryFile(
                delete = False,  # 不自动删除临时文件
                suffix = ('.'+'wav'),  # 文件后缀为.wav
            )

        # 执行TTS合成，将文本转换为语音并保存到临时文件
        # speaker: 选择说话人音色（使用模型的第一个说话人）
        # language: 指定语言
        # file_path: 输出音频文件路径
        self.tts.tts_to_file(text, speaker=self.tts.speakers[0], language=language, file_path=tempf.name)

        return tempf.name  # 返回生成的音频文件路径
