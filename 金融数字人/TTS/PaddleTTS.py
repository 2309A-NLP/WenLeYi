import os
# 导入PaddleSpeech的TTS执行器，用于调用语音合成模型
from paddlespeech.cli.tts.infer import TTSExecutor

"""
PaddleSpeech

声码器说明：这里预制了三种声码器【PWGan】【WaveRnn】【HifiGan】, 三种声码器效果和生成时间有比较大的差距，请跟进自己的需要进行选择。不过只选择了前两种，因为WaveRNN太慢了

| 声码器 | 音频质量 | 生成速度 |
| :----: | :----: | :----: |
| PWGan | 中等 | 中等 |
| WaveRnn | 高 | 非常慢（耐心等待） |
| HifiGan | 低 | 快 |

这些PaddleSpeech中的样例主要按数据集分类，我们主要使用的TTS数据集有：

CSMCS (普通话单发音人)
AISHELL3 (普通话多发音人)
LJSpeech (英文单发音人)
VCTK (英文多发音人)

PaddleSpeech 的 TTS 模型具有以下映射关系：

tts0 - Tacotron2
tts1 - TransformerTTS
tts2 - SpeedySpeech
tts3 - FastSpeech2
voc0 - WaveFlow
voc1 - Parallel WaveGAN
voc2 - MelGAN
voc3 - MultiBand MelGAN
voc4 - Style MelGAN
voc5 - HiFiGAN
vc0 - Tacotron2 Voice Clone with GE2E
vc1 - FastSpeech2 Voice Clone with GE2E

以下是 PaddleSpeech 提供的可以被命令行和 python API 使用的预训练模型列表：

- 声学模型
  | 模型 | 语言 |
  | :--- | :---: |
  |      speedyspeech_csmsc      |    zh    |
  |      fastspeech2_csmsc       |    zh    |
  |     fastspeech2_ljspeech     |    en    |
  |     fastspeech2_aishell3     |    zh    |
  |       fastspeech2_vctk       |    en    |
  | fastspeech2_cnndecoder_csmsc |    zh    |
  |       fastspeech2_mix        |   mix    |
  |       tacotron2_csmsc        |    zh    |
  |      tacotron2_ljspeech      |    en    |
  |       fastspeech2_male       |    zh    |
  |       fastspeech2_male       |    en    |
  |       fastspeech2_male       |   mix    |
  |       fastspeech2_canton     |  canton  |

- 声码器
  | 模型 | 语言 |
  | :--- | :---: |
  |         pwgan_csmsc          |    zh    |
  |        pwgan_ljspeech        |    en    |
  |        pwgan_aishell3        |    zh    |
  |          pwgan_vctk          |    en    |
  |       mb_melgan_csmsc        |    zh    |
  |      style_melgan_csmsc      |    zh    |
  |        hifigan_csmsc         |    zh    |
  |       hifigan_ljspeech       |    en    |
  |       hifigan_aishell3       |    zh    |
  |         hifigan_vctk         |    en    |
  |        wavernn_csmsc         |    zh    |
  |         pwgan_male           |    zh    |
  |        hifigan_male          |    zh    |
"""


# PaddleTTS语音合成类，封装了PaddleSpeech的TTS功能
class PaddleTTS:
    # 初始化方法（暂无需初始化操作）
    def __init__(self) -> None:
        pass
        
    # 核心方法：将文字合成为语音音频文件
    # 参数：
    #   text - 要合成的文字内容
    #   am - 声学模型名称（如'FastSpeech2'、'Tacotron2'）
    #   voc - 声码器名称（如'PWGan'、'HifiGan'）
    #   spk_id - 发音人ID（默认174，对应中文女声）
    #   lang - 语言（'zh'中文、'en'英文、'mix'混合、'canton'粤语）
    #   male - 是否使用男声（默认False）
    #   save_path - 输出音频文件路径（默认'output.wav'）
    def predict(self, text, am, voc, spk_id = 174, lang = 'zh', male=False, save_path = 'output.wav'):
        # 创建PaddleSpeech的TTS执行器实例，负责实际的语音合成
        self.tts = TTSExecutor()
        
        # 使用ONNX格式加速推理（比原生PyTorch更快）
        use_onnx = True
        # 统一转为小写，方便后续字符串匹配
        voc = voc.lower()
        am = am.lower()
        
        # 如果选择了男声，使用专门的男声模型
        if male:
            # 男声只支持pwgan和hifigan两种声码器,因为PaddleSpeech的男声预训练模型只在这两种声码器上训练过。
            assert voc in ["pwgan", "hifigan"], "male voc must be 'pwgan' or 'hifigan'"
            # 调用TTS引擎进行语音合成
            wav_file = self.tts(
            text = text,                    # 要合成的文字
            output = save_path,             # 输出文件路径
            am='fastspeech2_male',          # 男声专用声学模型
            voc= voc + '_male',             # 男声专用声码器
            lang=lang,                      # 语言
            use_onnx=use_onnx               # 使用ONNX加速
            )
            # 返回生成的音频文件路径
            return wav_file
    
        # 验证声学模型是否为支持的类型（Tacotron2或FastSpeech2）
        assert am in ['tacotron2', 'fastspeech2'], "am must be 'tacotron2' or 'fastspeech2'"
        
          # 根据语言类型，自动拼接对应的模型后缀名
        # 混合中文英文语音合成
        if lang == 'mix':
            # mix只有fastspeech2支持
            am = 'fastspeech2_mix'
            voc += '_csmsc'
        # 英文语音合成
        elif lang == 'en':
            # 使用LJSpeech英文数据集训练的模型
            am += '_ljspeech'
            voc += '_ljspeech'
        # 中文语音合成（默认）
        elif lang == 'zh':
            # 验证声码器是否在中文支持列表中
            assert voc in ['wavernn', 'pwgan', 'hifigan', 'style_melgan', 'mb_melgan'], "voc must be 'wavernn' or 'pwgan' or 'hifigan' or 'style_melgan' or 'mb_melgan'"
            # 使用CSMSC中文数据集训练的模型
            am += '_csmsc'
            voc += '_csmsc'
        # 粤语语音合成
        elif lang == 'canton':
            # 粤语使用专门的模型和声码器
            am = 'fastspeech2_canton'
            voc = 'pwgan_aishell3'
            spk_id = 10  # 粤语发音人ID
        # 打印当前使用的模型配置，方便调试
        print("am:", am, "voc:", voc, "lang:", lang, "male:", male, "spk_id:", spk_id)
        try:
            # 优先尝试用命令行方式调用paddlespeech进行语音合成（速度更快）
            # 第一步这里–am参数指定用FastSpeech2声学模型，系统拿到文字后分析每个字的发音、语速、语调，生成梅尔频谱图。
            cmd = f'paddlespeech tts --am {am} --voc {voc} --input "{text}" --output {save_path} --lang {lang} --spk_id {spk_id} --use_onnx {use_onnx}'
            # 第二步这条命令执行时，PWGAN声码器拿到梅尔频谱图，把它变成真正的音频波形。
            os.system(cmd)
            # 第三步，合成完成后，音频文件保存到你指定的路径，音频文件已保存到save_path
            wav_file = save_path
        except:
            # 如果命令行方式失败，回退到Python API方式调用（兼容性更好）
            # 语音合成
            wav_file = self.tts(
                text = text,            # 要合成的文字
                output = save_path,     # 输出文件路径
                am = am,                # 声学模型
                voc = voc,              # 声码器
                lang = lang,            # 语言
                spk_id = spk_id,        # 发音人ID
                use_onnx=use_onnx       # 使用ONNX加速
                )
        # 返回生成的音频文件路径
        return wav_file 
        

# 测试代码：直接运行此文件时执行
if __name__ == "__main__":
    # 创建PaddleTTS实例
    tts = PaddleTTS()
    # 测试合成：用FastSpeech2+PWGan合成英文"Hello world"
    tts.predict("Hello world", 'FastSpeech2', 'PWGan', spk_id=174, lang='en', male=False, save_path='output.wav')
