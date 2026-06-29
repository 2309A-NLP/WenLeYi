# 导入语言分段模块，用于自动识别文本中的中英文
import LangSegment
import numpy as np
import librosa  # 音频处理库，用于加载和处理音频文件
import torch  # PyTorch深度学习框架
import re, os
from transformers import AutoModelForMaskedLM, AutoTokenizer  # HuggingFace预训练模型，用于BERT文本特征提取
import sys
# 把GPT_SoVITS目录加入Python搜索路径，这样才能找到子模块
sys.path.append('GPT_SoVITS/')
from text import cleaned_text_to_sequence  # 文本转音素序列
from text.cleaner import clean_text  # 文本清洗和分词
from feature_extractor import cnhubert  # 音频特征提取模型（HuBERT）
from my_utils import load_audio  # 音频加载工具
from module.mel_processing import spectrogram_torch  # 梅尔频谱图计算
from module.models import SynthesizerTrn  # SoVITS合成器模型
from AR.models.t2s_lightning_module import Text2SemanticLightningModule  # GPT文本转语义模型
from scipy.io.wavfile import write  # 写入wav音频文件
from time import time as ttime  # 计时器，用于统计各阶段耗时

# ==================== 设备检测 ====================
# 自动检测可用的计算设备：优先GPU，其次Apple MPS，最后CPU
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"   # Apple Silicon GPU
else:
    device = "cpu"   # 纯CPU

# 是否使用半精度（float16）加速推理，老显卡不支持
is_half = True
# 标点符号集合，用于文本分句
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
# 检测显卡型号，老显卡（GTX16系列、P40、P10、1060/1070/1080）不支持半精度
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if (
            ("16" in gpu_name and "V100" not in gpu_name.upper())
            or "P40" in gpu_name.upper()
            or "P10" in gpu_name.upper()
            or "1060" in gpu_name
            or "1070" in gpu_name
            or "1080" in gpu_name
    ):
        is_half=False

# CPU模式下也关闭半精度
if device=="cpu":
    is_half=False

# 根据is_half决定数据类型：float16更快，float32更稳定
dtype=torch.float16 if is_half == True else torch.float32

# ==================== 模型路径配置 ====================
# BERT模型路径：用于提取文本的语义特征（中文RoBERTa）
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
# HuBERT模型路径：用于提取参考音频的音色特征
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
# 设置HuBERT模型路径
cnhubert.cnhubert_base_path = cnhubert_base_path

# ==================== 加载预训练模型 ====================
# 加载BERT分词器和模型（用于理解文本含义）
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

# 根据精度设置将模型移到GPU并设置精度
if is_half == True:
    bert_model = bert_model.half().to(device)  # 半精度，节省显存
else:
    bert_model = bert_model.to(device)  # 全精度

# 加载HuBERT音频特征提取模型（用于理解参考音频的音色）
ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def get_spepc(hps, filename):
    """
    从参考音频文件中提取梅尔频谱图
    梅尔频谱图是声音的"视觉表示"，包含了音频的频率和能量信息
    用于后续SoVITS模型解码时的参考
    """
    # 加载音频文件，采样率与模型配置一致
    audio = load_audio(filename, int(hps.data.sampling_rate))
    # 转为PyTorch张量
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    # 增加batch维度（模型需要4D输入）
    audio_norm = audio_norm.unsqueeze(0)
    # 计算梅尔频谱图
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,     # FFT长度
        hps.data.sampling_rate,     # 采样率
        hps.data.hop_length,        # 帧移
        hps.data.win_length,        # 窗口长度
        center=False,
    )
    return spec


def get_bert_feature(text, word2ph):
    """
    提取文本的BERT语义特征
    BERT能理解文本的含义，生成的特征向量可以帮助模型更好地理解要合成的文本
    """
    with torch.no_grad():  # 不计算梯度，节省显存
        # 用BERT分词器处理文本
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入移到GPU
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 用BERT模型提取隐藏层特征（取倒数第3层）
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    # 验证字数与音素数对应关系
    assert len(word2ph) == len(text)
    phone_level_feature = []
    # 每个字重复word2ph次（一个字可能对应多个音素）
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    # 拼接所有特征
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # 转置为模型需要的格式
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    """递归字典转对象：把嵌套字典转为可以点号访问的对象"""
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def clean_text_inf(text, language):
    """文本清洗：将文字转为音素序列（如"你好"→["n","i","h","ao"]）"""
    phones, word2ph, norm_text = clean_text(text, language.replace("all_",""))
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text


def get_bert_inf(phones, word2ph, norm_text, language):
    """根据语言获取BERT特征：中文用BERT提取，其他语言用零向量占位"""
    language=language.replace("all_","")
    if language == "zh":
        # 中文：用BERT提取文本语义特征
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        # 非中文：用全零向量占位（这些语言不使用BERT特征）
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


def splite_en_inf(sentence, language):
    """中英混合文本分割：把中英文混合的句子按语言拆开"""
    pattern = re.compile(r'[a-zA-Z ]+')  # 匹配英文单词
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])  # 英文前的中文部分
            langlist.append(language)
        textlist.append(sentence[start:end])  # 英文部分
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])  # 英文后的中文部分
        langlist.append(language)
    # 将标点符号合并到前一个词
    for i in range(len(textlist)-1, 0, -1):
        if re.match(r'^[\W_]+$', textlist[i]):
            textlist[i-1] += textlist[i]
            del textlist[i]
            del langlist[i]
    # 合并连续相同语言的片段
    i = 0
    while i < len(langlist) - 1:
        if langlist[i] == langlist[i+1]:
            textlist[i] += textlist[i+1]
            del textlist[i+1]
            del langlist[i+1]
        else:
            i += 1

    return textlist, langlist


def nonen_clean_text_inf(text, language):
    """非纯英文文本清洗：处理中英混合、纯中文、纯日文等"""
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        # 自动模式：用LangSegment自动识别每段文字的语言
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    # 对每段文字分别清洗
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "zh":
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    # 合并所有段的音素和特征
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text


def nonen_get_bert_inf(text, language):
    """非纯英文文本的BERT特征提取"""
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    bert_list = []
    for i in range(len(textlist)):
        text = textlist[i]
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(text, lang)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        bert_list.append(bert)
    # 拼接所有BERT特征
    bert = torch.cat(bert_list, dim=1)

    return bert


def get_first(text):
    """获取文本中第一个标点符号之前的内容"""
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_cleaned_text_fianl(text,language):
    """最终文本清洗入口：根据语言类型选择清洗方式"""
    if language in {"en","all_zh","all_ja"}:
        # 纯英文/纯中文/纯日文
        phones, word2ph, norm_text = clean_text_inf(text, language)
    elif language in {"zh", "ja","auto"}:
        # 中英混合/日英混合/自动识别
        phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
    return phones, word2ph, norm_text


def get_bert_final(phones, word2ph, norm_text, text_language, device, text):
    """最终BERT特征提取入口：根据语言类型选择提取方式"""
    if text_language == "en":
        bert = get_bert_inf(phones, word2ph, norm_text, text_language)
    elif text_language in {"zh", "ja","auto"}:
        bert = nonen_get_bert_inf(text, text_language)
    elif text_language == "all_zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros((1024, len(phones))).to(device)
    return bert


def split(todo_text):
    """按标点符号分割文本为多个句子"""
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    """切句策略1：每4句合并为一段"""
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    """切句策略2：每50个字符合并为一段"""
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # 最后一段太短就合并到前一段
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    """切句策略3：按中文句号切分"""
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])


def cut4(inp):
    """切句策略4：按英文句号切分"""
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])


def cut5(inp):
    """切句策略5：按所有标点符号切分"""
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：]'
    items = re.split(f'({punds})', inp)
    items = ["".join(group) for group in zip(items[::2], items[1::2])]
    opt = "\n".join(items)
    return opt


# ==================== GPT-SoVITS核心类 ====================
class GPT_SoVITS:
    def __init__(self):
        """初始化：模型暂时为空，需要调用load_model加载"""
        self.model = None

    def load_model(self, gpt_path, sovits_path):
        """
        【第一步：加载模型（load_model方法）】
        加载GPT-SoVITS两个核心模型：
        1. GPT模型（t2s_model）：负责理解文字含义，把文字转成语义token
        2. SoVITS模型（vq_model）：负责生成音频，把语义token变成声音
        """
        # GPT部分：加载文本到语义的转换模型
        self.hz = 50  # 语义帧率
        dict_s1 = torch.load(gpt_path, map_location="cpu")  # 加载GPT模型权重
        self.config = dict_s1["config"]
        self.max_sec = self.config["data"]["max_sec"]
        # 创建Text2Semantic模型（GPT负责把文字变成语义token）
        t2s_model = Text2SemanticLightningModule(self.config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if is_half == True:
            t2s_model = t2s_model.half()
        self.t2s_model = t2s_model.to(device)
        self.t2s_model.eval()  # 设为评估模式（关闭dropout等训练行为）
        total = sum([param.nelement() for param in t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        
        # SoVITS部分：加载音频合成模型
        dict_s2 = torch.load(sovits_path, map_location="cpu")  # 加载SoVITS模型权重
        self.hps = dict_s2["config"]
        self.hps = DictToAttrRecursive(self.hps)  # 字典转对象，方便点号访问
        self.hps.model.semantic_frame_rate = "25hz"
        # 创建SynthesizerTrn模型（SoVITS负责把语义token变成音频）
        vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        # 如果不是预训练权重，删除编码器（推理不需要）
        if ("pretrained" not in sovits_path):
            del vq_model.enc_q
        if is_half == True:
            self.vq_model = vq_model.half().to(device)
        else:
            self.vq_model = vq_model.to(device)
        self.vq_model.eval()
        print(self.vq_model.load_state_dict(dict_s2["weight"], strict=False))

    def predict(self, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="不切", save_path = 'vits_res.wav'):
        """
        语音克隆的入口函数
        参数：
            ref_wav_path - 参考音频路径（3-10秒，你想克隆的声音）
            prompt_text - 参考音频对应的文本（参考音频说了什么）
            prompt_language - 参考音频的语言
            text - 要合成的目标文本（想让克隆声音说什么）
            text_language - 目标文本的语言
            how_to_cut - 切句策略
            save_path - 输出音频路径
        """
        print(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut)
        return self.get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, save_path)

    def get_tts_wav(self, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="不切", save_path = 'vits_res.wav'):
        """
        【核心函数】语音克隆的实际执行逻辑
        整个流程分四个阶段：
        阶段1：从参考音频中提取音色特征（"记住这个人的声音"）
        阶段2：把目标文字转成音素和BERT特征（"理解要说什么话"）
        阶段3：GPT模型根据音色+文字预测语义token（"决定怎么说"）
        阶段4：SoVITS模型根据语义token生成音频波形（"把声音造出来"）
        
        参数说明：
            ref_wav_path - 参考音频文件路径（3-10秒，你想克隆的声音样本）
            prompt_text - 参考音频对应的文本内容（这段音频说了什么话）
            prompt_language - 参考音频的语言（中文/英文/日文等）
            text - 要合成的目标文本（你想让克隆声音说什么话）
            text_language - 目标文本的语言
            how_to_cut - 长文本切句策略（不切/凑四句一切/按标点切等）
            save_path - 输出音频文件的保存路径
        返回：保存的wav文件路径
        """
        t0 = ttime()  # 记录函数开始时间，用于统计各阶段耗时
        # 文本预处理：确保文本末尾有标点符号
        # 原因：模型训练时数据末尾都有标点，没有标点会导致合成效果变差
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        text = text.strip("\n")
        if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
        print("实际输入的参考文本:", prompt_text)  # 打印参考文本，方便调试
        print("实际输入的目标文本:", text)  # 打印目标文本，方便调试
        # 创建0.3秒的静音音频片段
        # 用途：在拼接多段音频时，段与段之间加0.3秒静音，听起来更自然
        # 采样率从hps配置读取（通常是44100Hz）
        # 数据类型根据is_half决定：半精度用float16节省显存，全精度用float32更稳定
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if is_half == True else np.float32,
        )
        with torch.no_grad():  # 关闭梯度计算，节省显存，推理时不需要反向传播
            # 【阶段1-步骤1】加载参考音频
            # librosa.load把音频文件加载为numpy数组
            # sr=16000表示重采样到16kHz（16000个采样点/秒）
            # HuBERT模型要求输入必须是16kHz的音频
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            # 验证音频长度在3-10秒之间
            # 16kHz采样率下：3秒=48000个采样点，10秒=160000个采样点
            # 太短（<3秒）音色信息不够，太长（>10秒）显存不够
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError("参考音频在3~10秒范围外，请更换！")
            # 把numpy数组转为PyTorch张量（tensor），才能送入GPU计算
            wav16k = torch.from_numpy(wav16k)  # 参考音频转张量
            zero_wav_torch = torch.from_numpy(zero_wav)  # 静音片段转张量
            # 将数据移到GPU（cuda）并设置精度
            # half()把float32转为float16，显存占用减半，推理速度更快
            if is_half == True:
                wav16k = wav16k.half().to(device)  # 参考音频移到GPU，半精度
                zero_wav_torch = zero_wav_torch.half().to(device)  # 静音移到GPU
            else:
                wav16k = wav16k.to(device)  # 参考音频移到GPU，全精度
                zero_wav_torch = zero_wav_torch.to(device)  # 静音移到GPU
            # 在参考音频末尾拼接0.3秒静音
            # 原因：防止音频突然截断，加一段静音让结尾更自然
            wav16k = torch.cat([wav16k, zero_wav_torch])
            # 【阶段1-步骤2】用HuBERT模型提取音频的语义特征
            # HuBERT是一个自监督语音模型，能从音频中提取高层语义特征
            # unsqueeze(0)增加batch维度：(N,) → (1, N)，模型要求4D输入
            # model()返回一个字典，取"last_hidden_state"（最后一层隐藏状态）
            # transpose(1,2)转置维度，变成模型需要的格式
            # 输出ssl_content的形状：(batch, 特征维度, 时间步数)
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(1, 2)
            # 【阶段1-步骤3】用SoVITS的VQ编码器提取潜在语义编码
            # extract_latent把HuBERT的特征进一步压缩成离散的语义token
            # 这些token就是参考音频的"音色指纹"（prompt_semantic）
            # 后续GPT模型会参考这个指纹来生成新音频
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]  # 取第一个batch的第一个通道，作为参考音色编码
        t1 = ttime()  # 记录阶段1完成时间（音色提取耗时）
        
        # 语言映射表：把用户界面选择的语言名转为模型内部使用的代码
        # 例如用户选"中文"，模型内部用"all_zh"表示全部按中文处理
        dict_language = {
            "中文": "all_zh",      # 全部按中文识别
            "英文": "en",          # 全部按英文识别
            "日文": "all_ja",      # 全部按日文识别
            "中英混合": "zh",       # 按中英混合识别
            "日英混合": "ja",       # 按日英混合识别
            "多语种混合": "auto",   # 多语种自动切分识别语种
        }
        prompt_language = dict_language[prompt_language]  # 参考文本语言转换
        text_language = dict_language[text_language]  # 目标文本语言转换

        # 【阶段2-步骤1】对参考文本进行文本清洗
        # clean_text把文字转为音素序列（如"你好"→["n","i","h","ao"]）
        # phones: 音素ID列表，模型实际处理的是音素而不是文字
        # word2ph: 每个字对应的音素数量（一个字可能对应多个音素）
        # norm_text: 归一化后的文本（繁转简、全角转半角等）
        phones1, word2ph1, norm_text1=get_cleaned_text_fianl(prompt_text, prompt_language)

        # 【阶段2-步骤2】对目标文本进行切分
        # 长文本需要切分成短句，因为模型对单次输入长度有限制
        # 不同的切句策略适用于不同场景：
        # "不切"：整段一起合成，适合短文本
        # "凑四句一切"：每4个句子合并为一段
        # "凑50字一切"：每50个字符合并为一段
        # "按中文句号。切"：按中文句号切分
        # "按标点符号切"：按所有标点符号切分
        if (how_to_cut == "凑四句一切"):
            text = cut1(text)
        elif (how_to_cut == "凑50字一切"):
            text = cut2(text)
        elif (how_to_cut == "按中文句号。切"):
            text = cut3(text)
        elif (how_to_cut == "按英文句号.切"):
            text = cut4(text)
        elif (how_to_cut == "按标点符号切"):
            text = cut5(text)
        # 清理多余空行：把连续的换行符合并为单个换行符
        text = text.replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n")
        print("实际输入的目标文本(切句后):", text)  # 打印切分后的文本，方便调试
        texts = text.split("\n")  # 按换行符分割为多个短句
        audio_opt = []  # 列表：存储每段生成的音频片段，最后拼接成完整音频
        # 【阶段2-步骤3】提取参考文本的BERT语义特征
        # BERT能理解文本的深层含义，生成的特征向量帮助模型理解"这句话的意思"
        # bert1是参考文本的特征，在循环外计算一次即可（所有句子共用）
        bert1=get_bert_final(phones1, word2ph1, norm_text1, prompt_language, device, text).to(dtype)

        # 【阶段3+4】逐句生成音频
        # 为什么要逐句？因为GPT和SoVITS对单次输入长度有限制
        # 长文本切分成短句后，每句单独合成，最后拼接成完整音频
        for text in texts:
            # 跳过空行（切句后可能产生空行）
            if (len(text.strip()) == 0):
                continue
            # 确保句子末尾有标点符号（模型训练数据的惯例）
            if (text[-1] not in splits): text += "。" if text_language != "en" else "."
            print("实际输入的目标文本(每句):", text)  # 打印当前处理的句子
            # 【阶段2-步骤4】对当前句子进行文本清洗，转为音素序列
            phones2, word2ph2, norm_text2 = get_cleaned_text_fianl(text, text_language)
            # 提取当前句子的BERT语义特征
            bert2 = get_bert_final(phones2, word2ph2, norm_text2, text_language, device, text).to(dtype)

            # 【关键】拼接参考文本和目标文本的BERT特征
            # 原因：GPT模型需要同时看到"参考文本的语义"和"目标文本的语义"
            # 这样它才能学会"用参考音频的风格来说目标文本"
            bert = torch.cat([bert1, bert2], 1)

            # 【阶段3-步骤1】准备GPT模型的输入数据
            # 把参考文本和目标文本的音素ID拼接成一个完整序列
            # 例如：参考音素[1,2,3] + 目标音素[4,5,6] = [1,2,3,4,5,6]
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
            bert = bert.to(device).unsqueeze(0)  # BERT特征移到GPU
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)  # 音素总长度
            prompt = prompt_semantic.unsqueeze(0).to(device)  # 参考音色编码移到GPU
            t2 = ttime()  # 记录阶段2完成时间，开始阶段3（GPT推理）
            with torch.no_grad():  # 关闭梯度计算，节省显存
                # 【阶段3-步骤2】GPT模型推理：预测语义token
                # 这是语音克隆的核心步骤之一
                # GPT模型接收4个输入：
                #   1. all_phoneme_ids - 参考+目标的音素序列（"要读什么字"）
                #   2. all_phoneme_len - 音素序列的总长度
                #   3. prompt - 参考音频的音色编码（"用谁的声音"）
                #   4. bert - BERT语义特征（"这句话什么意思"）
                # 输出：pred_semantic（语义token序列）和idx（有效长度）
                # 语义token包含了"这句话应该用什么语调、节奏、情感来读"的信息
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,       # 音素序列
                    all_phoneme_len,       # 音素长度
                    prompt,                # 参考音频的音色编码
                    bert,                  # BERT语义特征
                    top_k=self.config["inference"]["top_k"],  # 采样参数，控制生成多样性
                    early_stop_num=self.hz * self.max_sec,     # 最大生成长度，防止无限生成
                )
            t3 = ttime()  # 记录GPT推理完成时间
            # 截取有效部分：GPT输出的token包含prompt部分和生成部分
            # 我们只需要生成部分（后idx个token），去掉前面的prompt部分
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            # 【阶段4-步骤1】提取参考音频的梅尔频谱图
            # 梅尔频谱图是声音的"视觉表示"，包含音频的频率和能量信息
            # SoVITS解码时需要参考音频的频谱图作为辅助信息
            refer = get_spepc(self.hps, ref_wav_path)
            if is_half == True:
                refer = refer.half().to(device)  # 半精度移到GPU
            else:
                refer = refer.to(device)  # 全精度移到GPU
            # 【阶段4-步骤2】SoVITS模型解码：生成音频波形
            # 这是语音克隆的最后一个核心步骤
            # SoVITS.decode接收3个输入：
            #   1. pred_semantic - GPT生成的语义token（"怎么说"）
            #   2. phones2 - 目标文本的音素序列（"读什么字"）
            #   3. refer - 参考音频的梅尔频谱图（"声音的蓝图"）
            # 输出：音频波形数据（numpy数组，每个元素是-1到1之间的浮点数）
            # .detach()断开计算图，.cpu()移到CPU，.numpy()转为numpy数组
            audio = (
                self.vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
                )
                    .detach()
                    .cpu()
                    .numpy()[0, 0]
            )
            # 防止爆音：音频振幅超过1会导致播放时失真（破音）
            # 归一化：把所有值除以最大值，使振幅范围回到-1到1之间
            max_audio=np.abs(audio).max()
            if max_audio>1:audio/=max_audio
            audio_opt.append(audio)      # 保存这段音频片段
            audio_opt.append(zero_wav)   # 在段尾加0.3秒静音间隔
            t4 = ttime()  # 记录当前句子合成完成时间
        # 打印各阶段耗时统计（单位：秒）
        # 格式：音色提取耗时  文本处理耗时  GPT推理耗时  SoVITS合成耗时
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        # 【最终输出】将所有音频片段拼接成完整音频并保存
        # np.concatenate把所有片段首尾相连
        # * 32768 把-1~1的浮点数转为-32768~32767的16位整数（wav标准格式）
        # .astype(np.int16)转为16位整数
        # write写入wav文件：指定采样率和音频数据
        write(save_path, self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16))
        return save_path  # 返回生成的音频文件路径


# 测试代码：直接运行此文件时执行
if __name__ == "__main__":
    # 创建GPT-SoVITS实例
    GPT_SoVITS_inference = GPT_SoVITS()
    # 模型路径
    gpt_path = "GPT_SoVITS/pretrained_models/Gnews-e15.ckpt"
    sovits_path = "GPT_SoVITS/pretrained_models/Gnews_e8_s96.pth"
    # 加载模型
    GPT_SoVITS_inference.load_model(gpt_path, sovits_path)
    # 参考音频
    ref_wav_path = "GPT_SoVITS/reference_wav/Gnews/Gnews.mp3_0000270720_0000424960.wav"
    # 用ASR自动识别参考音频的文本
    from ASR import WhisperASR, FunASR
    asr = FunASR()
    prompt_text = ""
    prompt_text = asr.transcribe(ref_wav_path)
    prompt_language = "中文"
    # 要合成的目标文本
    text = "大家好，这是我语音克隆的声音。"
    text_language = "中英混合"
    how_to_cut = "不切"
    print("参考音频文本：", prompt_text)
