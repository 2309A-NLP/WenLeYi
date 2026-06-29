#整个文件就做了一件事：把音频文件变成文字。调用方式就是asr.transcribe(“音频路径”)，返回"识别出的文字"。
'''
说明这个模块来自OpenAI的Whisper项目
https://github.com/openai/whisper
pip install -U openai-whisper
'''
#导入Whisper库（Whisper是OpenAI开源的语音识别模型）
#它在68万小时的多语言音频数据上训练过，能识别99种语言的语音并转成文字。
import whisper
#把当前目录加到Python搜索路径，这样能找到src目录下的模块
import sys
sys.path.append('./')
# 导入一个计时装饰器，用来统计函数运行耗时
from src.cost_time import calculate_time 

#定义一个语音识别类
class WhisperASR:
    #两件事
    def __init__(self, model_path):
        #第一件：self.LANGUAGES = {一个超大的字典} — 定义了99种语言的映射表“en"对应"english”，"zh"对应"chinese"
        self.LANGUAGES = {
            "en": "english",
            "zh": "chinese",
            "de": "german",
            "es": "spanish",
            "ru": "russian",
            "ko": "korean",
            "fr": "french",
            "ja": "japanese",
            "pt": "portuguese",
            "tr": "turkish",
            "pl": "polish",
            "ca": "catalan",
            "nl": "dutch",
            "ar": "arabic",
            "sv": "swedish",
            "it": "italian",
            "id": "indonesian",
            "hi": "hindi",
            "fi": "finnish",
            "vi": "vietnamese",
            "he": "hebrew",
            "uk": "ukrainian",
            "el": "greek",
            "ms": "malay",
            "cs": "czech",
            "ro": "romanian",
            "da": "danish",
            "hu": "hungarian",
            "ta": "tamil",
            "no": "norwegian",
            "th": "thai",
            "ur": "urdu",
            "hr": "croatian",
            "bg": "bulgarian",
            "lt": "lithuanian",
            "la": "latin",
            "mi": "maori",
            "ml": "malayalam",
            "cy": "welsh",
            "sk": "slovak",
            "te": "telugu",
            "fa": "persian",
            "lv": "latvian",
            "bn": "bengali",
            "sr": "serbian",
            "az": "azerbaijani",
            "sl": "slovenian",
            "kn": "kannada",
            "et": "estonian",
            "mk": "macedonian",
            "br": "breton",
            "eu": "basque",
            "is": "icelandic",
            "hy": "armenian",
            "ne": "nepali",
            "mn": "mongolian",
            "bs": "bosnian",
            "kk": "kazakh",
            "sq": "albanian",
            "sw": "swahili",
            "gl": "galician",
            "mr": "marathi",
            "pa": "punjabi",
            "si": "sinhala",
            "km": "khmer",
            "sn": "shona",
            "yo": "yoruba",
            "so": "somali",
            "af": "afrikaans",
            "oc": "occitan",
            "ka": "georgian",
            "be": "belarusian",
            "tg": "tajik",
            "sd": "sindhi",
            "gu": "gujarati",
            "am": "amharic",
            "yi": "yiddish",
            "lo": "lao",
            "uz": "uzbek",
            "fo": "faroese",
            "ht": "haitian creole",
            "ps": "pashto",
            "tk": "turkmen",
            "nn": "nynorsk",
            "mt": "maltese",
            "sa": "sanskrit",
            "lb": "luxembourgish",
            "my": "myanmar",
            "bo": "tibetan",
            "tl": "tagalog",
            "mg": "malagasy",
            "as": "assamese",
            "tt": "tatar",
            "haw": "hawaiian",
            "ln": "lingala",
            "ha": "hausa",
            "ba": "bashkir",
            "jw": "javanese",
            "su": "sundanese",
        }
        #第二件： 加载Whisper模型。
        self.model = whisper.load_model(model_path)

    @calculate_time
    #接收一个音频文件路径
    def transcribe(self, audio_file):
        #调用Whisper模型把音频转成文字
        result = self.model.transcribe(audio_file)
        #返回识别出的文本（result是个字典，里面还有语言、时间戳等信息，我们只要text）
        return result["text"]

# 测试代码（if __name__ == "__main__"）：只有直接运行这个文件时才会执行下面的代码块
if __name__ == "__main__":
    import os  # 导入 os 模块，用于文件和路径操作
    # 创建 ASR（自动语音识别）对象并进行语音识别
    model_path = "./Whisper/tiny.pt"  # 指定 Whisper 模型的本地路径（tiny 是轻量级模型）
    audio_file = "output.wav"         # 指定待识别的音频文件名

    # 检查 audio_file（output.wav）这个音频文件是否存在
    if not os.path.exists(audio_file):
        # 如果文件不存在，则使用 edge-tts 命令行工具生成一段测试音频
        # edge-tts 是微软 Edge 浏览器的文本转语音工具，这里将 "hello" 转为语音并保存为 output.wav
        os.system('edge-tts --text "hello" --write-media output.wav')
    # 实例化 WhisperASR(自动语音识别)类，传入模型路径，加载语音识别模型
    asr = WhisperASR(model_path)
    # 调用 transcribe 方法对音频文件进行语音识别，并打印识别出的文本内容
    print(asr.transcribe(audio_file))