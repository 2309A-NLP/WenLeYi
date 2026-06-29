# 异步编程支持，Edge-TTS内部使用异步HTTP请求
import asyncio
import requests
import json
import os
# 从edge_tts库导入Communicate（语音合成）和SubMaker（字幕生成）
from edge_tts import Communicate, SubMaker
from io import TextIOWrapper
from typing import Any, TextIO, Union
import sys
sys.path.append('../Linly-Talker')
from src import cost_time
from src.cost_time import calculate_time
os.environ["GRADIO_TEMP_DIR"]= './temp'

"""
Constants for the Edge TTS project. 
https://github.com/rany2/edge-tts/blob/master/src/edge_tts/constants.py
"""

# 微软Edge TTS服务的API地址
BASE_URL = "speech.platform.bing.com/consumer/speech/synthesize/readaloud"
# 微软提供的可信客户端令牌（用于认证）
TRUSTED_CLIENT_TOKEN = "6A5AA1D4EAFF4E9FB37E23D68491D6F4"

# WebSocket连接地址（用于流式语音合成）
WSS_URL = f"wss://{BASE_URL}/edge/v1?TrustedClientToken={TRUSTED_CLIENT_TOKEN}"
# 获取所有可用音色的列表地址
VOICE_LIST = f"https://{BASE_URL}/voices/list?trustedclienttoken={TRUSTED_CLIENT_TOKEN}"

# 默认音色：英文Emma多语言神经网络语音
DEFAULT_VOICE = "en-US-EmmaMultilingualNeural"

def list_voices_fn(proxy=None):
    """
    获取所有可用的语音音色列表
    从微软Edge TTS服务获取所有支持的音色信息
    返回：包含所有音色属性的字典列表
    """
    # 模拟浏览器请求头，避免被微软服务器拒绝
    headers = {
        "Authority": "speech.platform.bing.com",
        "Sec-CH-UA": '" Not;A Brand";v="99", "Microsoft Edge";v="91", "Chromium";v="91"',
        "Sec-CH-UA-Mobile": "?0",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36 Edg/91.0.864.41",
        "Accept": "*/*",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
    }
    # 向微软服务器发送请求，获取音色列表（3秒超时）
    response = requests.get(VOICE_LIST, headers=headers, timeout=3)
    # 解析JSON响应
    data = json.loads(response.text)
    return data



# EdgeTTS语音合成类，封装了微软Edge TTS的在线语音合成功能
class EdgeTTS:
    # 初始化：获取所有可用音色列表
    # list_voices: 是否打印音色列表
    # proxy: 代理服务器地址
    def __init__(self, list_voices = False, proxy = None) -> None:
        try:
            # 尝试从微软服务器获取音色列表
            voices = list_voices_fn(proxy=proxy)
            # 提取所有音色的简称（如zh-CN-XiaoxiaoNeural）
            self.SUPPORTED_VOICE = [item['ShortName'] for item in voices]
            # 按字母倒序排列
            self.SUPPORTED_VOICE.sort(reverse=True)
            if list_voices:
                print(", ".join(self.SUPPORTED_VOICE))
            # 标记网络可用
            self.network = True
        except:
            # 网络不可用时，使用预置的音色列表（离线备用）
            self.network = False
            # 预置了300多种音色，覆盖全球主要语言
            self.SUPPORTED_VOICE = ['zu-ZA-ThembaNeural', 'zu-ZA-ThandoNeural',  'zh-TW-YunJheNeural', 'zh-TW-HsiaoYuNeural', 'zh-TW-HsiaoChenNeural', 'zh-HK-WanLungNeural', 
                                    'zh-HK-HiuMaanNeural', 'zh-HK-HiuGaaiNeural', 'zh-CN-shaanxi-XiaoniNeural', 'zh-CN-liaoning-XiaobeiNeural', 
                                    'zh-CN-YunyangNeural', 'zh-CN-YunxiaNeural', 'zh-CN-YunxiNeural', 'zh-CN-YunjianNeural', 
                                    'zh-CN-XiaoyiNeural', 'zh-CN-XiaoxiaoNeural', 'vi-VN-NamMinhNeural', 'vi-VN-HoaiMyNeural', 
                                    'uz-UZ-SardorNeural', 'uz-UZ-MadinaNeural', 'ur-PK-UzmaNeural', 'ur-PK-AsadNeural', 
                                    'ur-IN-SalmanNeural', 'ur-IN-GulNeural', 'uk-UA-PolinaNeural', 'uk-UA-OstapNeural', 
                                    'tr-TR-EmelNeural', 'tr-TR-AhmetNeural', 'th-TH-PremwadeeNeural', 'th-TH-NiwatNeural', 
                                    'te-IN-ShrutiNeural', 'te-IN-MohanNeural', 'ta-SG-VenbaNeural', 'ta-SG-AnbuNeural', 
                                    'ta-MY-SuryaNeural', 'ta-MY-KaniNeural', 'ta-LK-SaranyaNeural', 'ta-LK-KumarNeural', 
                                    'ta-IN-ValluvarNeural', 'ta-IN-PallaviNeural', 'sw-TZ-RehemaNeural', 'sw-TZ-DaudiNeural', 
                                    'sw-KE-ZuriNeural', 'sw-KE-RafikiNeural', 'sv-SE-SofieNeural', 'sv-SE-MattiasNeural', 
                                    'su-ID-TutiNeural', 'su-ID-JajangNeural', 'sr-RS-SophieNeural', 'sr-RS-NicholasNeural', 'sq-AL-IlirNeural', 'sq-AL-AnilaNeural', 
                                    'so-SO-UbaxNeural', 'so-SO-MuuseNeural', 'sl-SI-RokNeural', 'sl-SI-PetraNeural', 
                                    'sk-SK-...ural', '***', 'si-LK-ThiliniNeural', 'si-LK-SameeraNeural', 
                                    'ru-RU-SvetlanaNeural', 'ru-RU-DmitryNeural', 'ro-RO-EmilNeural', 'ro-RO-AlinaNeural', 'pt-PT-RaquelNeural', 'pt-PT-DuarteNeural', 'pt-BR-ThalitaNeural', 'pt-BR-FranciscaNeural', 
                                    'pt-BR-AntonioNeural', 'ps-AF-LatifaNeural', 'ps-AF-GulNawazNeural', 'pl-PL-ZofiaNeural', 
                                    'pl-PL-MarekNeural', 'nl-NL-MaartenNeural', 'nl-NL-FennaNeural', 'nl-NL-ColetteNeural', 
                                    'nl-BE-DenaNeural', 'nl-BE-ArnaudNeural', 'ne-NP-SagarNeural', 'ne-NP-HemkalaNeural', 
                                    'nb-NO-PernilleNeural', 'nb-NO-FinnNeural', 'my-MM-ThihaNeural', 'my-MM-NilarNeural', 
                                    'mt-MT-JosephNeural', 'mt-MT-GraceNeural', 'ms-MY-YasminNeural', 'ms-MY-OsmanNeural', 
                                    'mr-IN-ManoharNeural', 'mr-IN-AarohiNeural', 'mn-MN-YesuiNeural', 'mn-MN-BataaNeural', 
                                    'ml-IN-SobhanaNeural', 'ml-IN-MidhunNeural', 'mk-MK-MarijaNeural', 'mk-MK-AleksandarNeural', 
                                    'lv-LV-NilsNeural', 'lv-LV-EveritaNeural', 'lt-LT-OnaNeural', 'lt-LT-LeonasNeural', 'lo-LA-KeomanyNeural', 'lo-LA-ChanthavongNeural', 
                                    'ko-KR-SunHiNeural', 'ko-KR-InJoonNeural', 'ko-KR-HyunsuNeural', 'kn-IN-SapnaNeural', 
                                    'kn-IN-GaganNeural', 'km-KH-SreymomNeural', 'km-KH-PisethNeural', 'kk-KZ-DauletNeural', 
                                    'kk-KZ-AigulNeural', 'ka-GE-GiorgiNeural', 'ka-GE-EkaNeural', 'jv-ID-SitiNeural', 'jv-ID-DimasNeural', 
                                    'ja-JP-NanamiNeural', 'ja-JP-KeitaNeural', 'it-IT-IsabellaNeural', 'it-IT-GiuseppeNeural', 'it-IT-ElsaNeural', 
                                    'it-IT-DiegoNeural', 'is-IS-GunnarNeural', 'is-IS-GudrunNeural', 'id-ID-GadisNeural', 'id-ID-ArdiNeural', 
                                    'hu-HU-TamasNeural', 'hu-HU-NoemiNeural', 'hr-HR-SreckoNeural', 'hr-HR-GabrijelaNeural', 'hi-IN-SwaraNeural', 
                                    'hi-IN-MadhurNeural', 'he-IL-HilaNeural', 'he-IL-AvriNeural', 'gu-IN-NiranjanNeural', 'gu-IN-DhwaniNeural', 
                                    'gl-ES-SabelaNeural', 'gl-ES-RoiNeural', 'ga-IE-OrlaNeural', 'ga-IE-ColmNeural', 'fr-FR-VivienneMultilingualNeural', 
                                    'fr-FR-RemyMultilingualNeural', 'fr-FR-HenriNeural', 'fr-FR-EloiseNeural', 'fr-FR-DeniseNeural', 'fr-CH-FabriceNeural', 
                                    'fr-CH-ArianeNeural', 'fr-CA-ThierryNeural', 'fr-CA-SylvieNeural', 'fr-CA-JeanNeural', 'fr-CA-AntoineNeural', 
                                    'fr-BE-GerardNeural', 'fr-BE-CharlineNeural', 'fil-PH-BlessicaNeural', 'fil-PH-AngeloNeural', 'fi-FI-NooraNeural', 
                                    'fi-FI-HarriNeural', 'fa-IR-FaridNeural', 'fa-IR-DilaraNeural', 'et-EE-KertNeural', 'et-EE-AnuNeural', 
                                    'es-VE-SebastianNeural', 'es-VE-PaolaNeural', 'es-UY-ValentinaNeural', 'es-UY-MateoNeural', 'es-US-PalomaNeural', 
                                    'es-US-AlonsoNeural', 'es-SV-RodrigoNeural', 'es-SV-LorenaNeural', 'es-PY-TaniaNeural', 'es-PY-MarioNeural', 
                                    'es-PR-VictorNeural', 'es-PR-KarinaNeural', 'es-PE-CamilaNeural', 'es-PE-AlexNeural', 'es-PA-RobertoNeural', 
                                    'es-PA-MargaritaNeural', 'es-NI-YolandaNeural', 'es-NI-FedericoNeural', 'es-MX-JorgeNeural', 'es-MX-DaliaNeural', 
                                    'es-HN-KarlaNeural', 'es-HN-CarlosNeural', 'es-GT-MartaNeural', 'es-GT-AndresNeural', 'es-GQ-TeresaNeural', 
                                    'es-GQ-JavierNeural', 'es-ES-XimenaNeural', 'es-ES-ElviraNeural', 'es-ES-AlvaroNeural', 'es-EC-LuisNeural', 
                                    'es-EC-AndreaNeural', 'es-DO-RamonaNeural', 'es-DO-EmilioNeural', 'es-CU-ManuelNeural', 'es-CU-BelkysNeural', 
                                    'es-CR-MariaNeural', 'es-CR-JuanNeural', 'es-CO-SalomeNeural', 'es-CO-GonzaloNeural', 'es-CL-LorenzoNeural', 
                                    'es-CL-CatalinaNeural', 'es-BO-SofiaNeural', 'es-BO-MarceloNeural', 'es-AR-TomasNeural', 'es-AR-ElenaNeural', 
                                    'en-ZA-LukeNeural', 'en-ZA-LeahNeural', 'en-US-SteffanNeural', 'en-US-RogerNeural', 'en-US-MichelleNeural', 
                                    'en-US-JennyNeural', 'en-US-GuyNeural', 'en-US-EricNeural', 'en-US-EmmaNeural', 'en-US-ChristopherNeural', 
                                    'en-US-BrianNeural', 'en-US-AvaNeural', 'en-US-AriaNeural', 'en-US-AndrewNeural', 'en-US-AnaNeural', 
                                    'en-TZ-ImaniNeural', 'en-TZ-ElimuNeural', 'en-SG-WayneNeural', 'en-SG-LunaNeural', 'en-PH-RosaNeural', 
                                    'en-PH-JamesNeural', 'en-NZ-MollyNeural', 'en-NZ-MitchellNeural', 'en-NG-EzinneNeural', 
                                    'en-NG-AbeoNeural', 'en-KE-ChilembaNeural', 'en-KE-AsiliaNeural', 'en-IN-PrabhatNeural', 
                                    'en-IN-NeerjaNeural', 'en-IN-NeerjaExpressiveNeural', 'en-IE-EmilyNeural', 'en-IE-ConnorNeural', 
                                    'en-HK-YanNeural', 'en-HK-SamNeural', 'en-GB-ThomasNeural', 'en-GB-SoniaNeural', 'en-GB-RyanNeural', 
                                    'en-GB-MaisieNeural', 'en-GB-LibbyNeural', 'en-CA-LiamNeural', 'en-CA-ClaraNeural', 'en-AU-WilliamNeural', 
                                    'en-AU-NatashaNeural', 'el-GR-NestorasNeural', 'el-GR-AthinaNeural', 'de-DE-SeraphinaMultilingualNeural', 
                                    'de-DE-KillianNeural', 'de-DE-KatjaNeural', 'de-DE-FlorianMultilingualNeural', 'de-DE-ConradNeural', 
                                    'de-DE-AmalaNeural', 'de-CH-LeniNeural', 'de-CH-JanNeural', 'de-AT-JonasNeural', 'de-AT-IngridNeural', 
                                    'da-DK-JeppeNeural', 'da-DK-ChristelNeural', 'cy-GB-NiaNeural', 'cy-GB-AledNeural', 'cs-CZ-VlastaNeural',
                                    'cs-CZ-AntoninNeural', 'ca-ES-JoanaNeural', 'ca-ES-EnricNeural', 'bs-BA-VesnaNeural', 'bs-BA-GoranNeural', 
                                    'bn-IN-TanishaaNeural', 'bn-IN-BashkarNeural', 'bn-BD-PradeepNeural', 'bn-BD-NabanitaNeural', 'bg-BG-KalinaNeural', 
                                    'bg-BG-BorislavNeural', 'az-AZ-BanuNeural', 'az-AZ-BabekNeural', 'ar-YE-SalehNeural', 'ar-YE-MaryamNeural', 
                                    'ar-TN-ReemNeural', 'ar-TN-HediNeural', 'ar-SY-LaithNeural', 'ar-SY-AmanyNeural', 'ar-SA-ZariyahNeural', 
                                    'ar-SA-HamedNeural', 'ar-QA-MoazNeural', 'ar-QA-AmalNeural', 'ar-OM-AyshaNeural', 'ar-OM-AbdullahNeural', 
                                    'ar-MA-MounaNeural', 'ar-MA-JamalNeural', 'ar-LY-OmarNeural', 'ar-LY-ImanNeural', 'ar-LB-RamiNeural', 'ar-LB-LaylaNeural', 
                                    'ar-KW-NouraNeural', 'ar-KW-FahedNeural', 'ar-JO-TaimNeural', 'ar-JO-SanaNeural', 'ar-IQ-RanaNeural', 'ar-IQ-BasselNeural', 
                                    'ar-EG-ShakirNeural', 'ar-EG-SalmaNeural', 'ar-DZ-IsmaelNeural', 'ar-DZ-AminaNeural', 'ar-BH-LailaNeural', 'ar-BH-AliNeural', 
                                    'ar-AE-HamdanNeural', 'ar-AE-FatimaNeural', 'am-ET-MekdesNeural', 'am-ET-AmehaNeural', 'af-ZA-WillemNeural', 'af-ZA-AdriNeural']

    # 预处理语速、音量、音调参数，转为微软API需要的格式
    def preprocess(self, rate, volume, pitch):
        # 语速：正数加+号，负数保持原样，单位%
        if rate >= 0:
            rate = f'+{rate}%'
        else:
            rate = f'{rate}%'
        # 音调：正数加+号，单位Hz
        if pitch >= 0:
            pitch = f'+{pitch}Hz'
        else:
            pitch = f'{pitch}Hz'
        # 音量：微软API用负数表示降低，100减去用户设置的值
        volume = 100 - volume
        volume = f'-{volume}%'
        return rate, volume, pitch

    # 异步版本的语音合成（带字幕生成），实际项目中未使用
    @calculate_time
    def apredict(self,TEXT, VOICE, RATE, VOLUME, PITCH, OUTPUT_FILE='result.wav', OUTPUT_SUBS='result.vtt', words_in_cue = 8):
        async def amain() -> None:
            """异步主函数：流式合成音频并生成字幕"""
            rate, volume, pitch = self.preprocess(rate = RATE, volume = VOLUME, pitch = PITCH)
            # 创建Communicate对象，设置文字、音色、语速、音量、音调
            communicate = Communicate(TEXT, VOICE, rate = rate, volume = volume, pitch = pitch)
            submaker = SubMaker()  # 字幕生成器
            with open(OUTPUT_FILE, "wb") as file:
                # 流式获取音频数据块
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        file.write(chunk["data"])  # 写入音频数据
                    elif chunk["type"] == "WordBoundary":
                        submaker.feed(chunk)  # 记录字幕时间戳

            # 保存字幕文件
            with open(OUTPUT_SUBS, "w", encoding="utf-8") as file:
                file.write(submaker.get_srt())

        asyncio.run(amain())
        # 读取字幕文件，去掉中文文字中的空格
        with open(OUTPUT_SUBS, 'r', encoding='utf-8') as file:
            vtt_lines = file.readlines()

        # 去掉每一行文字中的空格
        vtt_lines_without_spaces = [line.replace(" ", "") if "-->" not in line else line for line in vtt_lines]
        with open(OUTPUT_SUBS, 'w', encoding='utf-8') as output_file:
            output_file.writelines(vtt_lines_without_spaces)
        return OUTPUT_FILE, OUTPUT_SUBS

    # 【核心方法】同步版本的语音合成（实际使用的方法）
    @calculate_time
    def predict(self,TEXT, VOICE, RATE, VOLUME, PITCH, OUTPUT_FILE='result.wav', OUTPUT_SUBS='result.vtt', words_in_cue = 8):
        """
        将文字合成为语音音频
        参数：
            TEXT - 要合成的文字
            VOICE - 音色名称（如'zh-CN-XiaoxiaoNeural'）
            RATE - 语速调整（-50到+50）
            VOLUME - 音量调整（0到100）
            PITCH - 音调调整（-50到+50）
            OUTPUT_FILE - 输出音频路径
        """
        # 预处理参数：转为微软API需要的格式
        rate, volume, pitch = self.preprocess(rate = RATE, volume = VOLUME, pitch = PITCH)
        # 创建Communicate对象：文字+音色+语速+音量+音调
        communicate = Communicate(TEXT, VOICE, rate = rate, volume = volume, pitch = pitch)
        # 同步保存音频文件（一行代码搞定语音合成）
        communicate.save_sync(OUTPUT_FILE)

        
        '''
        由于EdgeTTS的一些问题，并且最近发现其生成的字幕还是有些奇怪，所以会重新写一个字幕生成，暂不使用其字母生成
        '''
        # communicate = Communicate(TEXT, VOICE)
        # submaker = SubMaker()
        # with open(OUTPUT_FILE, "wb") as file:
        #     for chunk in communicate.stream_sync():
        #         if chunk["type"] == "audio":
        #             file.write(chunk["data"])
        #         elif chunk["type"] == "WordBoundary":
        #             submaker.feed(chunk)

        # with open(OUTPUT_SUBS, "w", encoding="utf-8") as file:
        #     file.write(submaker.get_srt())

        # 返回音频文件路径，字幕暂时返回None
        return OUTPUT_FILE, None

# 测试函数：验证Edge-TTS是否正常工作
def test():
    # 创建EdgeTTS实例（不打印音色列表）
    tts = EdgeTTS(list_voices=False)
    # 测试文本：一段中文金融新闻
    TEXT = '''近日，苹果公司起诉高通公司，状告其未按照相关合约进行合作，高通方面尚未回应。这句话中"其"指的是谁？'''
    # 使用中文晓晓音色
    VOICE = "zh-CN-XiaoxiaoNeural"
    OUTPUT_FILE, OUTPUT_SUBS = "tts.wav", "tts.vtt"
    # 调用语音合成
    audio_file, sub_file = tts.predict(TEXT, VOICE, 0, 0, 0, OUTPUT_FILE, OUTPUT_SUBS)
    print("Audio file written to", audio_file, "and subtitle file written to", sub_file)

# 直接运行此文件时执行测试
if __name__ == "__main__":
    test()
