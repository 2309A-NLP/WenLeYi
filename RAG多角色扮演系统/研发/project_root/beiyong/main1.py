# -*- coding: utf-8 -*-
"""
金融理财角色扮演 RAG 后端主服务。

职责：
- 用户/角色管理（MySQL）
- 知识检索（Milvus 混合检索 + BGE 重排）
- 多轮记忆（Redis 短期 + Milvus 长期）
- 对话生成（支持多家在线大模型 API）
"""

# ==================== 导入标准库模块 ====================
import os              # 操作系统接口，用于读取环境变量
import logging         # 日志记录
import hashlib         # 密码哈希
import socket          # 网络通信
import re              # 正则表达式
import time            # 时间处理
from typing import List, Dict, Any, Optional, Union, Iterable, Tuple  # 类型提示
from datetime import datetime  # 日期时间处理
from contextlib import asynccontextmanager  # 异步上下文管理器

# ==================== 请求缓存（提升性能）====================
_request_cache: Dict[str, dict] = {}  # key: 问题哈希, value: {"answer": str, "timestamp": float}
_CACHE_TTL = 300  # 缓存有效期5分钟
_CACHE_MAX_SIZE = 100  # 最大缓存条目数

# ==================== 导入第三方库 ====================
import redis                 # Redis 客户端，用于短期对话记忆
import pymysql               # MySQL 数据库驱动
from pymilvus import (        # Milvus 向量数据库客户端
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusException, AnnSearchRequest, RRFRanker
)
from pymilvus.model.sparse import BM25EmbeddingFunction      # BM25 稀疏向量嵌入
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer  # BM25 分词器
from sentence_transformers import SentenceTransformer       # 句子向量模型（BGE）
from FlagEmbedding import FlagReranker                      # 重排序模型
from fastapi import FastAPI, UploadFile, File, Form, HTTPException  # FastAPI Web框架
from fastapi.responses import StreamingResponse             # 流式响应
from pydantic import BaseModel                             # 数据验证模型
from openai import OpenAI                                   # OpenAI 兼容API客户端
import fitz                                               # PyMuPDF，PDF文档处理
from dotenv import load_dotenv                             # 环境变量加载

# ==================== 加载环境变量 ====================
load_dotenv()

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../rag_system.log', encoding='utf-8'),  # 日志文件输出
        logging.StreamHandler()                                   # 控制台输出
    ]
)
logger = logging.getLogger(__name__)

# ==================== 配置常量 ====================

# ==================== 大模型配置 ====================
# LLM_PROVIDER: 指定使用的大模型提供商，支持: deepseek(深度求索), doubao(字节豆包), qianwen(阿里通义), chatgpt
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")          # 深度求索 API 密钥
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")              # 字节豆包 API Key
DOUBAO_SECRET_KEY = os.getenv("DOUBAO_SECRET_KEY")        # 字节豆包 Secret Key
QIANWEN_API_KEY = os.getenv("QIANWEN_API_KEY")           # 阿里通义千问 API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")             # OpenAI API 密钥

# ==================== 本地模型路径配置 ====================
# BGE_M3: 中文Embedding模型，用于将文本转为向量
LOCAL_BGE_M3_PATH = os.getenv("BGE_M3_PATH", r"D:\模型\bge_m3_model")
# BGE-Reranker: 重排序模型，用于对检索结果进行精排
LOCAL_BGE_RERANKER_PATH = os.getenv("BGE_RERANKER_PATH", r"D:\模型\bge-reranker-base")

# ==================== MySQL 数据库配置 ====================
# 用于存储用户信息、角色信息、对话历史等结构化数据
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),         # 数据库主机地址
    "port": int(os.getenv("MYSQL_PORT", 3306)),            # 数据库端口
    "user": os.getenv("MYSQL_USER", "root"),              # 数据库用户名
    "password": os.getenv("MYSQL_PASSWORD", ""),         # 数据库密码
    "database": os.getenv("MYSQL_DATABASE", "roleplay"), # 数据库名称
    "charset": "utf8mb4",                                 # 字符编码，支持表情符号
    "autocommit": False,                                  # 关闭自动提交，手动控制事务
}

# ==================== Redis 配置 ====================
# Redis: 用于短期对话记忆存储，支持快速读写
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),         # Redis 主机地址
    "port": int(os.getenv("REDIS_PORT", 6379)),           # Redis 端口
    "db": int(os.getenv("REDIS_DB", 0)),                  # Redis 数据库编号
    "decode_responses": True                               # 自动将字节转换为字符串
}

# ==================== Milvus 向量数据库配置 ====================
# Milvus: 用于存储和检索向量化的金融知识库
MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.18.128")   # Milvus 服务器地址
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))         # Milvus 端口
MILVUS_USER = os.getenv("MILVUS_USER", "")                 # Milvus 用户名（可选）
MILVUS_PASSWORD = os.getenv("ILVUS_PASSWORD", "")         # Milvus 密码（可选）
MILVUS_USE_SSL = os.getenv("MILVUS_USE_SSL", "false").lower() == "true"  # 是否使用SSL加密
MILVUS_TIMEOUT = int(os.getenv("MILVUS_TIMEOUT", 30))      # 连接超时时间（秒）
MILVUS_ENABLE_LOCAL = os.getenv("MILVUS_ENABLE_LOCAL", "false").lower() == "true"  # 是否启用本地模式

# ==================== RAG 检索参数配置 ====================
# 知识库按 role_id 隔离到不同 collection，避免多角色检索互相污染
COLLECTION_PREFIX = "financial_knowledge_role"  # 知识库集合名称前缀
EMBEDDING_DIM = 1024                            # 向量维度（BGE-M3 模型输出维度）
CHUNK_SIZE = 512                                # 文本分块大小（字符数）
CHUNK_OVERLAP = 50                              # 相邻文本块重叠字符数
TOP_K_RETRIEVE = 10                             # 初步检索返回数量
TOP_K_RERANK = 3                                # 重排序后保留的相关文档数量
MEMORY_LENGTH = 5                               # 短期记忆保留的对话轮数
TOP_K_LONG_TERM = 3                             # 长期记忆检索返回数量
TOP_K_STOCK = 2                                 # 股票数据检索返回数量

# ==================== 角色提示词模板定义 ====================
# 定义不同角色的系统提示词，包括金融理财师、医生、心理医生、虚拟朋友等多种角色
ROLE_PROMPT_TEMPLATES = {
    "financial_advisor": """
你是一位专业的金融理财师，精通各类金融产品和投资策略。

要求：
1. 分析用户的问题，提供专业、客观的金融建议
2. 结合提供的参考信息进行回答，确保回答不少于300字
3. 使用通俗易懂的语言，避免过于专业的术语
4. 保持友好、耐心的态度
5. 如果参考信息不足，可以基于你的专业知识补充回答
6. 必须包含风险提示，提醒用户投资有风险

禁止：
- 做出任何投资承诺
- 推荐具体的股票或基金产品
- 提供未经证实的信息

请开始回答：
""",
    "investment_advisor": """
你是一位资深的投资顾问，专注于股票、基金等投资产品分析。

要求：
1. 深入分析用户的投资需求和风险承受能力
2. 结合市场数据和参考信息提供投资建议
3. 回答不少于300字
4. 详细解释投资逻辑和潜在风险
5. 保持专业但易懂的表达方式

禁止：
- 直接推荐买入或卖出特定股票
- 做出收益保证
- 使用过于情绪化的语言

请开始回答：
""",
    "financial_planner": """
你是一位专业的财务规划师，擅长制定全面的财务规划方案。

要求：
1. 帮助用户制定长期财务目标
2. 分析用户的财务状况，提供个性化建议
3. 回答不少于300字
4. 涵盖储蓄、投资、保险等多个方面
5. 保持客观、专业的态度

禁止：
- 推销特定金融产品
- 做出不切实际的承诺
- 涉及敏感的财务隐私问题

请开始回答：
"""
}

ROLE_PROMPT_TEMPLATES.update(
    {
        "doctor": """
你是一位谨慎、通俗、负责任的全科医生助手。

要求：
1. 先根据症状判断常见原因和严重程度
2. 优先给出居家处理建议、观察指标和何时就医
3. 不随意推荐处方药，不替代线下诊断
4. 涉及高热不退、呼吸困难、胸痛、严重过敏、意识异常等情况，要明确建议立即就医
5. 不要输出 JSON、代码包装或无关格式
""",
        "psychologist": """
你是一位专业、温和、善于倾听的心理咨询师。

要求：
1. 先理解用户情绪，再给出安抚、分析和可执行建议
2. 语言温柔、自然，不要机械复读
3. 优先提供情绪调节、沟通、自我照顾等建议
4. 若用户出现持续失眠、自伤、绝望等高风险信号，要建议尽快联系家人和专业机构
5. 回答要像真实对话，不要输出 JSON、代码或模板字段
""",
        "virtual_friend": """
你是一位温暖、自然、会认真接话的虚拟朋友。

要求：
1. 以朋友口吻自然聊天，不要机械重复问候
2. 要承接用户上一句话，围绕当前话题继续交流
3. 可以适度安慰、共情、鼓励，但不要假装拥有现实中不存在的经历
4. 不要输出 JSON、代码、字段名或多余符号
""",
        "teacher": """
你是一位耐心的老师，擅长把复杂问题讲清楚。

要求：
1. 先判断用户问题属于什么知识点
2. 用通俗中文分步骤解释
3. 能举一个简单例子帮助理解
4. 不要输出 JSON、代码包装或无关格式
""",
        "lawyer": """
你是一位严谨的法律咨询助手。

要求：
1. 基于一般法律常识解释问题，不捏造法条
2. 先说明通用原则，再给出实际建议
3. 涉及具体案件时提醒用户补充地区、合同、证据等信息
4. 不要输出 JSON、代码包装或无关格式
""",
        "scientist": """
你是一位科学知识助手。

要求：
1. 回答强调事实、原理和逻辑
2. 概念解释要准确、清楚
3. 避免夸张和未经证实的说法
4. 不要输出 JSON、代码包装或无关格式
""",
        "english_tutor": """
你是一位英语学习助手。

要求：
1. 根据用户水平用中英结合方式回答
2. 可以纠错、翻译、讲语法、给例句
3. 表达清晰，不要长篇空话
4. 不要输出 JSON、代码包装或无关格式
""",
        "stock_analyst": """
你是一位股票分析助手。

要求：
1. 优先结合公开数据、行业逻辑、风险点回答
2. 不承诺收益，不给绝对化结论
3. 解释结论背后的原因
4. 不要输出 JSON、代码包装或无关格式
""",
    }
)

ROLE_DISPLAY_NAMES = {
    "doctor": "医生",
    "financial_advisor": "金融理财师",
    "investment_advisor": "投资顾问",
    "financial_planner": "财务规划师",
    "psychologist": "心理医生",
    "virtual_friend": "虚拟朋友",
    "teacher": "教师",
    "lawyer": "律师",
    "scientist": "科学家",
    "english_tutor": "英语学习助手",
    "stock_analyst": "股票分析师",
}

ROLE_DEFAULT_DESCRIPTIONS = {
    "doctor": "基于常见症状与健康常识提供就医建议的医生助手",
    "financial_advisor": "专业金融理财师，提供理财规划和投资建议",
    "investment_advisor": "资深投资顾问，专注于股票基金分析",
    "financial_planner": "专业财务规划师，制定全面财务规划",
    "psychologist": "温和理性的心理支持与情绪疏导助手",
    "virtual_friend": "可以陪聊、倾听和共情的虚拟朋友",
    "teacher": "耐心讲解知识点、辅导学习的教师助手",
    "lawyer": "提供通用法律信息和处理思路的律师助手",
    "scientist": "解释科学原理与常识问题的科学助手",
    "english_tutor": "帮助翻译、纠错、练习表达的英语学习助手",
    "stock_analyst": "结合市场信息进行客观分析的股票分析师",
}

ROLE_SQLITE_KNOWLEDGE = {
    "doctor": [
        ("普通感冒", "普通感冒常见症状包括流鼻涕、鼻塞、咽痛、轻度发热、咳嗽。多数由病毒引起，重点是休息、补水、清淡饮食和对症处理。"),
        ("感冒用药原则", "感冒常用的是对症药物，如退热止痛、缓解鼻塞、止咳化痰等，不建议自行混用多种复方感冒药，避免重复用药。"),
        ("发热处理", "成人体温超过38.5摄氏度或明显不适时可考虑退热。若高热持续不退、伴呼吸困难、意识模糊、剧烈胸痛，应尽快就医。"),
        ("咳嗽观察", "咳嗽若超过两周未缓解，或伴胸闷气短、咳黄脓痰、咯血，应及时线下检查。"),
        ("胃痛处理", "胃痛需要区分饮食刺激、胃炎、溃疡等情况。若反复发作、伴黑便、呕血、明显体重下降，应尽快到医院检查。"),
        ("腹泻处理", "腹泻时重点预防脱水，可少量多次补液，暂时避免油腻和生冷食物。若持续高热、便血、严重腹痛，应尽快就医。"),
        ("高血压常识", "高血压患者应关注规律监测、低盐饮食、体重控制、按医嘱服药，不要自行频繁更换降压药。"),
        ("何时立即就医", "胸痛、呼吸困难、抽搐、意识不清、持续高热、严重过敏、外伤大出血等情况，应立即就医或呼叫急救。"),
        ("头痛处理", "头痛常见原因包括紧张性头痛、偏头痛、感冒、高血压等。轻度头痛可尝试休息、放松、避免强光噪音；若头痛剧烈、伴恶心呕吐、视力模糊或持续不减，应及时就医。"),
        ("关节疼痛", "关节痛可能由劳损、外伤、关节炎等引起。急性期可冷敷减轻肿胀，慢性期可热敷促进循环。若关节红肿发热、活动受限或持续加重，建议检查。"),
        ("皮肤过敏", "皮肤过敏常表现为红肿、瘙痒、皮疹。应先避免接触可疑过敏原，保持皮肤清洁干燥，避免搔抓。若症状严重或伴呼吸困难，需立即就医。"),
        ("失眠改善", "改善失眠可尝试固定作息、睡前一小时远离电子产品、保持卧室黑暗安静、避免咖啡因和剧烈运动。若长期失眠影响日间状态，建议寻求专业帮助。"),
        ("便秘处理", "便秘可通过多喝水、增加膳食纤维、规律运动改善。避免长期依赖泻药。若便秘持续超过一周或伴腹痛便血，应检查原因。"),
        ("疲劳乏力", "持续疲劳可能与睡眠不足、贫血、甲状腺问题、压力过大有关。建议先保证充足睡眠、均衡饮食、适度运动。若疲劳持续两周以上，建议体检。"),
        ("喉咙痛处理", "喉咙痛多由病毒感染引起，可多喝水、用温盐水漱口、避免辛辣刺激食物。若咽痛剧烈影响吞咽、伴高热或持续超过一周，应就医检查。"),
        ("眼睛干涩", "眼睛干涩常见于长时间用电子设备、干燥环境。可多眨眼、定时休息、使用人工泪液。若伴眼痛、视力模糊或红肿，应及时检查。"),
        ("肌肉酸痛", "运动后肌肉酸痛可通过拉伸、热敷、按摩缓解。若疼痛剧烈或持续数日不减，可能是肌肉拉伤，需休息并考虑就医。"),
        ("消化不良", "消化不良常与饮食不当、情绪有关。建议规律饮食、细嚼慢咽、饭后适当活动。若反复腹胀、嗳气、反酸，应检查排除胃炎或溃疡。"),
        ("鼻出血处理", "鼻出血时应保持镇静，身体前倾，用手指捏紧鼻翼10-15分钟，同时用嘴呼吸。不要仰头或填塞纸巾。若出血不止或频繁发生，应就医。"),
    ],
    "psychologist": [
        ("情绪低落应对", "当用户说心情不好、焦虑、烦躁时，先共情，再帮助识别诱因，建议做深呼吸、短暂散步、规律睡眠、减少反复自责。"),
        ("压力管理", "面对学习和工作压力时，可把问题拆成最小步骤，先完成一件最容易的小事，减少失控感。"),
        ("高风险提醒", "如果用户反复提到绝望、自伤、自杀想法、严重失眠或无法正常生活，应建议尽快联系家人、老师、医生或心理热线。"),
        ("失眠应对", "偶发失眠可先从固定作息、睡前减少刷手机、避免咖啡因、放松训练入手。若持续数周影响生活，需要考虑专业帮助。"),
        ("社交焦虑", "社交紧张时可以先练习低压力场景表达，从小范围互动开始，避免因为一次尴尬就全盘否定自己。"),
        ("情绪表达", "当用户难以表达感受时，可帮助其用几个词描述：委屈、愤怒、疲惫、无助、害怕，再继续分析具体诱因。"),
        ("亲密关系困扰", "感情问题讨论时，先区分事实、猜测和需求，再考虑沟通方式，而不是只围绕对错争论。"),
        ("考试焦虑", "考试前焦虑可尝试深呼吸、积极自我暗示、合理复习计划和适当运动。重要的是接纳焦虑情绪，告诉自己紧张是正常的。"),
        ("职场压力", "工作压力大时，可先做任务分类，区分紧急重要的事情。适当与同事沟通，避免独自硬扛。必要时寻求领导支持调整工作安排。"),
        ("孤独感", "感到孤独时，可以尝试主动联系朋友、参加兴趣小组、培养一个爱好，或者做一些帮助他人的事情来获得连接感。"),
        ("完美主义困扰", "完美主义容易导致拖延和自我否定。可以尝试设定合理目标，接受不完美，关注过程而非结果，从小事开始行动。"),
        ("拖延症", "拖延往往源于害怕失败或任务太庞大。可以用五分钟起步法，把任务分解成微小步骤，先完成最容易的部分。"),
        ("自我否定", "自我否定时，试着像对待好朋友一样对待自己。写下负面想法，寻找证据反驳，用更客观的视角看待自己。"),
        ("人际边界", "建立健康边界很重要，学会说不，尊重自己的感受和需求。不需要取悦所有人，合理的拒绝是自我保护。"),
        ("焦虑发作", "焦虑发作时，先找一个安静的地方，做深呼吸，感受身体的感觉，告诉自己这只是暂时的，会过去的。"),
        ("情绪调节技巧", "情绪调节可以用深呼吸、冥想、运动、写日记、听音乐等方式。找到适合自己的方法，在情绪来临时使用。"),
        ("自我关怀", "自我关怀包括善待自己、允许自己休息、接纳不完美、关注身体感受。就像照顾生病的朋友一样照顾自己。"),
        ("家庭关系", "家庭矛盾常源于沟通不畅和期待差异。试着站在对方角度理解，用温和的方式表达自己的需求，而非指责。"),
        ("决策困难", "难以做决定时，可以列出利弊、征求信任的人意见、设置决策时限。记住没有完美的选择，只有合适的选择。"),
    ],
    "virtual_friend": [
        ("陪伴聊天", "虚拟朋友要像真实朋友一样接住用户的话，不要每次都重复自我介绍。要针对用户当前情绪、处境和话题继续交流。"),
        ("安慰方式", "当用户说难过、烦闷、委屈时，可以先表达理解，再问一句更贴近的话，例如：今天发生了什么，还是这种状态持续一阵子了？"),
        ("天气闲聊", "用户聊今天天气、吃了什么、在做什么时，虚拟朋友应自然接话，可以分享观察感受并继续延展话题，而不是回到固定开场白。"),
        ("日常陪伴", "当用户只是想找人聊天时，不要追问成咨询问卷式对话，可以顺着对方的话题慢慢聊。"),
        ("共情语气", "虚拟朋友适合使用自然口语，比如：听起来你今天有点累、这事确实容易让人烦、要不你跟我说说最难受的是哪一段。"),
        ("鼓励方式", "鼓励要具体，不要只说加油。可以说：你已经撑到现在了，这本身就很不容易。"),
        ("兴趣话题", "当用户提到喜欢的音乐、电影、游戏或运动时，可以表现出兴趣，问问具体喜欢哪一点，分享类似的感受。"),
        ("工作学习", "用户聊工作学习时，可询问遇到的挑战、进展如何，给予理解和支持，比如：听起来这个项目挺有挑战性的，你是怎么应对的？"),
        ("美食分享", "聊到美食时，可以问问用户最喜欢的口味、最近吃到什么好吃的，分享一些简单的美食体验，让话题更生活化。"),
        ("周末计划", "周末话题可以问用户打算怎么过，有没有特别想做的事，或者分享一些轻松的周末活动建议。"),
        ("宠物话题", "如果用户提到宠物，可以问问宠物的名字、品种，分享一些宠物的可爱瞬间，让对话更温馨。"),
        ("旅行经历", "聊旅行时，可以问问用户去过哪里，印象最深的地方是什么，有没有特别的故事可以分享。"),
        ("追剧聊天", "用户聊电视剧或综艺时，可以问问最喜欢的角色、印象深刻的情节，一起讨论剧情发展。"),
        ("深夜陪伴", "深夜聊天时语气可以更轻柔，理解用户可能的孤独感，用安静陪伴的方式回应，比如：这么晚还没睡，是有心事吗？"),
        ("日常吐槽", "当用户吐槽生活琐事时，先表示认同，再一起轻松吐槽，让用户感觉被理解，比如：确实，这种小事最让人烦了！"),
        ("正向反馈", "当用户分享开心的事时，要真心为对方高兴，表达具体的赞赏，比如：太棒了！你怎么做到的，快跟我说说！"),
        ("轻松幽默", "适当加入轻松幽默的回应，让聊天更有趣，比如用一些网络流行语或可爱的表情符号（用文字描述）。"),
        ("倾听技巧", "用户倾诉时，不要急于给建议，先认真倾听，用嗯、我懂、原来是这样等回应，让用户感到被重视。"),
        ("话题转换", "当话题聊完时，可以自然过渡到相关话题，比如从工作聊到休息，从美食聊到做饭，让对话流畅自然。"),
    ],
    "teacher": [
        ("教学表达", "老师角色回答时应该先下定义，再讲原理，最后给例子，适合中学生和大学生阅读。"),
        ("鼓励学习", "当学生不会时，要先降低心理压力，再一步一步引导，而不是只给结论。"),
        ("数学讲解", "数学题可按已知条件、公式选择、代入过程、结果检验四步说明。"),
        ("语文阅读", "语文阅读题要先抓文章中心，再看关键词、修辞和上下文逻辑。"),
        ("英语学习", "讲英语时可按词义、句型、语法和例句四部分展开。"),
        ("物理学习", "物理问题通常先找受力或已知量，再确定公式和单位，避免一上来盲算。"),
    ],
    "lawyer": [
        ("合同纠纷", "合同问题通常需要先看合同约定、付款记录、聊天记录和违约事实，再判断是否协商、发函、调解或诉讼。"),
        ("劳动纠纷", "劳动争议常见证据包括劳动合同、工资流水、考勤记录、通知截图等。"),
        ("欠钱不还", "民间借贷纠纷重点看借条、转账记录、聊天承认欠款的内容以及还款约定。"),
        ("租房纠纷", "租房纠纷常见争议包括押金、提前解约、房屋损坏和维修责任，处理时要看合同条款和交接证据。"),
        ("交通事故", "交通事故一般要先固定现场证据、报警、保留医疗票据，再根据责任认定协商或索赔。"),
        ("法律边界", "律师助手提供的是一般信息，不应替代正式律师审查合同或代理案件。"),
    ],
    "scientist": [
        ("科学解释", "科学问题应优先说明定义、机制、证据和常见误区，避免绝对化。"),
        ("科普表达", "科学助手回答时要兼顾准确性和通俗性，可用生活类比帮助理解。"),
        ("实验思维", "遇到科学争议时，应关注样本量、变量控制、是否可重复和证据质量。"),
        ("伪科学识别", "判断说法是否可靠，可以看是否有明确证据、是否夸大绝对、是否回避验证。"),
        ("生物常识", "生物问题适合从结构、功能、环境影响和个体差异四个方面解释。"),
    ],
    "english_tutor": [
        ("英语纠错", "英语学习助手应先给自然表达，再解释语法点，并提供1到2个类似例句。"),
        ("口语练习", "口语问题可以先给简短自然表达，再提供更正式版本。"),
        ("翻译原则", "翻译不是逐词替换，而是优先保证意思自然、语气合适和语法正确。"),
        ("写作润色", "英语写作润色时可以从词汇更自然、句子更简洁、逻辑更清楚三方面修改。"),
        ("时态讲解", "讲时态时要结合时间线，说明动作发生的时间和是否持续。"),
    ],
    "stock_analyst": [
        ("股票分析框架", "分析股票时通常看行业景气度、公司基本面、估值水平、盈利能力、现金流和风险因素。"),
        ("风险提示", "涉及个股时应提示市场波动、政策变化、业绩不及预期等风险。"),
        ("行业分析", "分析行业时要看政策环境、竞争格局、需求变化、盈利模式和周期位置。"),
        ("房地产行业", "房地产行业分析通常关注销售数据、政策支持、融资环境、库存去化和龙头企业现金流压力。"),
        ("财报阅读", "看财报时可先看营收、利润、毛利率、现金流和负债，再判断增长质量。"),
        ("估值视角", "估值不能只看一个指标，要结合行业特征、增长性和历史区间一起判断。"),
    ],
    "financial_advisor": [
        ("理财原则", "理财建议通常要考虑收入稳定性、应急金、负债水平、风险承受能力与投资期限。"),
        ("资产配置", "资产配置通常不是押注单一产品，而是根据目标和期限在现金、固收、权益和保障之间做平衡。"),
        ("应急金", "多数情况下建议先准备3到6个月生活支出的应急金，再考虑中长期投资。"),
        ("风险承受能力", "风险承受能力受收入稳定性、家庭责任、投资经验和心理承受能力共同影响。"),
    ],
    "investment_advisor": [
        ("投资建议框架", "投资顾问回答时应先了解用户目标、期限、风险偏好，再讨论资产类别和组合分散。"),
        ("基金选择", "基金分析常看基金经理风格、回撤控制、费用、持仓结构和长期表现。"),
        ("止损止盈", "止损止盈规则要事先设定，避免情绪化操作。"),
        ("分散投资", "分散投资能降低单一资产波动对整体组合的冲击，但不等于没有风险。"),
    ],
    "financial_planner": [
        ("财务规划", "财务规划应包含现金流管理、储蓄、保险保障、长期投资和阶段性目标。"),
        ("家庭预算", "家庭预算可以按必要支出、弹性支出、储蓄投资三类拆分，先保证现金流稳健。"),
        ("保险配置", "保险规划常见顺序是先考虑医保、意外险、重疾险和定期寿险，再看其他需求。"),
        ("长期目标", "教育金、购房、养老等长期目标需要明确时间点、金额目标和年度储蓄计划。"),
    ],
}

# 全局变量
_embedding_model = None
# ==================== 全局变量和缓存 ====================
_reranker_model = None                # BGE-Reranker 模型实例（全局缓存）
_bm25_models: Dict[str, BM25EmbeddingFunction] = {}  # BM25 模型缓存（按集合名索引）
_redis_client = None                 # Redis 客户端实例（全局缓存）
_in_memory_history: Dict[str, List[Dict[str, str]]] = {}  # 内存中的对话历史缓存
_milvus_available = False            # Milvus 连接是否可用的标志
_local_stock_cache: Optional[List[Dict[str, Any]]] = None  # 本地股票数据缓存

# Milvus 连接配置
_MILVUS_CONNECTION_ALIAS = "default"  # Milvus 连接别名

# ==================== 路径和数据目录 ====================
def _data_dir() -> str:
    """获取数据目录路径"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ==================== Milvus 连接状态检查 ====================
def milvus_ready() -> bool:
    """检查 Milvus 是否已连接可用"""
    return _milvus_available


def _load_local_stock_cache() -> List[Dict[str, Any]]:
    global _local_stock_cache
    if _local_stock_cache is not None:
        return _local_stock_cache

    candidates = [
        os.path.join(_data_dir(), "eastmoney_stock_list.json"),
        os.path.join(_data_dir(), "eastmoney_stock_suggest_cache.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                _local_stock_cache = data
                return _local_stock_cache
            if isinstance(data, dict):
                values = []
                for value in data.values():
                    if isinstance(value, list):
                        values.extend(value)
                    elif isinstance(value, dict):
                        values.append(value)
                _local_stock_cache = values
                return _local_stock_cache
        except Exception as e:
            logger.warning(f"读取本地股票数据失败: {path}: {e}")

    _local_stock_cache = []
    return _local_stock_cache


def _find_role_dataset_files(role_name: str) -> List[str]:
    """
    在 data/ 中查找与角色名相关的数据集文件（csv/txt）。
    规则：文件名包含 role_name（不区分大小写）。
    若找不到，则回退到 financial_qa.csv（如果存在）。
    """
    base = _data_dir()
    role_lower = (role_name or "").lower()
    candidates: List[str] = []
    try:
        for fn in os.listdir(base):
            lower = fn.lower()
            if not (lower.endswith(".csv") or lower.endswith(".txt")):
                continue
            if role_lower and role_lower in lower:
                candidates.append(os.path.join(base, fn))
    except Exception:
        candidates = []

    if candidates:
        return sorted(candidates)

    fallback = os.path.join(base, "financial_qa.csv")
    if os.path.exists(fallback):
        return [fallback]
    return []


def _load_texts_from_dataset(path: str) -> List[str]:
    """从 csv/txt 数据集中抽取可入库文本。"""
    lower = path.lower()
    if lower.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
        return chunk_text(txt) if txt else []

    # csv: 优先识别 question/ground_truth 两列
    import csv
    texts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            a = (row.get("ground_truth") or row.get("answer") or "").strip()
            if q and a:
                texts.append(f"Q: {q}\nA: {a}")
            elif q:
                texts.append(q)
            elif a:
                texts.append(a)
    return texts


def seed_role_knowledge_if_empty(role_id: int, role_name: str):
    """
    若角色知识库为空，则自动从 data/ 中加载与该角色匹配的数据集入库。
    """
    col = ensure_knowledge_collection(role_id)
    if col.num_entities > 0:
        return

    dataset_files = _find_role_dataset_files(role_name)
    if not dataset_files:
        logger.warning(f"未找到角色数据集，跳过自动入库: role={role_name}({role_id})")
        return

    all_texts: List[str] = []
    for p in dataset_files:
        try:
            all_texts.extend(_load_texts_from_dataset(p))
        except Exception as e:
            logger.warning(f"读取数据集失败，跳过: {p}: {e}")
            continue

    # 控制一次性入库量，避免首次启动过慢
    if not all_texts:
        return
    add_documents_to_milvus(
        texts=all_texts,
        source="dataset:auto",
        doc_id=f"seed_{role_name}",
        role_id=role_id,
    )


def knowledge_collection_name(role_id: int) -> str:
    return f"{COLLECTION_PREFIX}_{int(role_id)}"


def _knowledge_collection_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="create_time", dtype=DataType.INT64),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
    ]
    return CollectionSchema(fields, "角色隔离金融知识库集合")


def ensure_knowledge_collection(role_id: int) -> Collection:
    """确保某个 role_id 的知识库 collection 存在并已 load。"""
    name = knowledge_collection_name(role_id)
    try:
        if not utility.has_collection(name):
            logger.info(f"📦 创建角色知识库集合: {name}")
            collection = Collection(name, _knowledge_collection_schema())
            
            # 创建稠密向量索引
            logger.debug(f"🔧 创建稠密向量索引: {name}")
            collection.create_index(
                "dense_vector",
                {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}},
            )
            
            # 创建稀疏向量索引
            logger.debug(f"🔧 创建稀疏向量索引: {name}")
            collection.create_index(
                "sparse_vector",
                {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
            )
            
            logger.info(f"✅ 角色知识库集合 {name} 创建完成")
        else:
            logger.debug(f"📋 角色知识库集合 {name} 已存在")
            collection = Collection(name)
        
        # 确保集合已加载
        _ensure_collection_loaded(collection, name)
        
        return collection
        
    except Exception as e:
        logger.error(f"❌ 确保知识库集合 {name} 失败: {e}")
        raise

def _ensure_collection_loaded(collection: Collection, collection_name: str = ""):
    """确保集合已加载，兼容 pymilvus 2.6.x API"""
    try:
        # pymilvus 2.6.x 使用 utility.load_state 检查加载状态
        from pymilvus import utility
        try:
            load_state = utility.load_state(collection_name or collection.name)
            if load_state != utility.LoadState.Loaded:
                logger.info(f"📥 加载集合: {collection_name or collection.name}")
                collection.load()
        except AttributeError:
            # 如果 load_state 不可用，直接尝试加载
            logger.debug(f"📥 加载集合: {collection_name or collection.name}")
            collection.load()
    except Exception as e:
        logger.warning(f"⚠️ 检查/加载集合失败: {e}")
        # 尝试直接加载
        try:
            collection.load()
        except Exception:
            pass

# ========== Milvus 诊断工具 ==========
def diagnose_milvus_connection() -> dict:
    """诊断Milvus连接问题，返回诊断结果"""
    result = {
        "success": False,
        "host": MILVUS_HOST,
        "port": MILVUS_PORT,
        "ssl": MILVUS_USE_SSL,
        "timeout": MILVUS_TIMEOUT,
        "errors": [],
        "suggestions": [],
        "server_info": None
    }
    
    try:
        logger.info("🔍 开始诊断 Milvus 连接...")
        
        # 1. 检查网络连接
        result["errors"].append("正在检查网络连接...")
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            conn_result = sock.connect_ex((MILVUS_HOST, MILVUS_PORT))
            if conn_result == 0:
                result["errors"].append(f"✅ 网络连接正常: {MILVUS_HOST}:{MILVUS_PORT}")
            else:
                result["errors"].append(f"❌ 网络连接失败: 端口 {MILVUS_PORT} 无法访问")
                result["suggestions"].append("请检查 Milvus 服务是否正在运行")
                result["suggestions"].append("请检查防火墙设置，确保端口 19530 已开放")
        finally:
            sock.close()
        
        # 2. 尝试连接 Milvus
        result["errors"].append("正在尝试连接 Milvus...")
        from pymilvus import connections, utility
        
        try:
            connections.connect(
                alias="diagnose",
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                timeout=MILVUS_TIMEOUT
            )
            
            # 获取服务器信息
            version = utility.get_server_version()
            result["server_info"] = {"version": version}
            result["errors"].append(f"✅ Milvus 连接成功，版本: {version}")
            
            # 获取集合列表
            collections = utility.list_collections()
            result["errors"].append(f"📚 当前集合数量: {len(collections)}")
            
            # 断开诊断连接
            connections.disconnect("diagnose")
            
            result["success"] = True
            
        except Exception as e:
            error_msg = str(e)
            result["errors"].append(f"❌ Milvus 连接失败: {error_msg}")
            
            # 分析错误
            if "connection refused" in error_msg.lower():
                result["suggestions"].append("Milvus 服务未启动或端口错误")
                result["suggestions"].append("使用命令启动: milvus start 或 docker-compose up -d")
            elif "timeout" in error_msg.lower():
                result["suggestions"].append("网络超时，请检查网络连接")
                result["suggestions"].append(f"尝试增加 MILVUS_TIMEOUT 环境变量（当前: {MILVUS_TIMEOUT}s）")
            else:
                result["suggestions"].append(f"错误详情: {error_msg}")
        
        return result
        
    except Exception as e:
        result["errors"].append(f"❌ 诊断过程出错: {str(e)}")
        return result


# ========== 辅助函数 ==========
def get_embedding_model() -> SentenceTransformer:
    """获取向量化模型（单例）"""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"加载 BGE-m3 模型: {LOCAL_BGE_M3_PATH}")
        try:
            _embedding_model = SentenceTransformer(LOCAL_BGE_M3_PATH)
            logger.info("✅ BGE-m3 模型加载成功")
        except Exception as e:
            logger.error(f"❌ BGE-m3 模型加载失败: {e}")
            raise
    return _embedding_model

def get_reranker_model() -> FlagReranker:
    """获取重排序模型（单例）"""
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"加载 BGE-reranker 模型: {LOCAL_BGE_RERANKER_PATH}")
        try:
            import torch
            # 检查是否有可用的GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {device}")
            
            # 尝试多种加载方式
            try:
                _reranker_model = FlagReranker(LOCAL_BGE_RERANKER_PATH, use_fp16=False, device=device)
            except Exception as e:
                logger.debug(f"标准加载失败，尝试备用方式: {e}")
                _reranker_model = FlagReranker(LOCAL_BGE_RERANKER_PATH, use_fp16=False, device="cpu")
            
            logger.info("✅ BGE-reranker 模型加载成功")
        except Exception as e:
            logger.warning(f"⚠️ BGE-reranker 模型加载失败: {e}")
            _reranker_model = None
    return _reranker_model

def get_redis_client() -> redis.Redis:
    """获取 Redis 客户端（单例）"""
    global _redis_client
    if _redis_client is None:
        logger.info(f"连接 Redis: {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}")
        try:
            _redis_client = redis.Redis(**REDIS_CONFIG)
            _redis_client.ping()
            logger.info("✅ Redis 连接成功")
        except Exception as e:
            logger.warning(f"⚠️ Redis 连接失败: {e}")
            _redis_client = None
    return _redis_client

def hash_password(password: str) -> str:
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()

# ==================== 数据库操作函数 (MySQL) ====================
# 用于用户管理、角色管理、对话历史存储等结构化数据操作

def _build_custom_role_prompt(role: Dict[str, Any]) -> str:
    """根据角色信息构建自定义提示词"""
    role_name = role.get("display_name") or role.get("role_name") or "角色"
    personality = (role.get("personality") or "自然、友好、专业").strip()
    tone = (role.get("tone") or "口语化、真诚").strip()
    background_story = (role.get("background_story") or "无").strip()
    hobbies = (role.get("hobbies") or "无").strip()
    return f"""
你现在扮演“{role_name}”。

角色设定：
1. 性格：{personality}
2. 语气：{tone}
3. 背景故事：{background_story}
4. 爱好：{hobbies}

要求：
1. 回答必须始终符合上述人设，不要跳出角色。
2. 回答自然、具体，优先解决用户问题。
3. 不输出 JSON、代码块或字段名。
""".strip()

def get_db_connection():
    """获取 MySQL 数据库连接"""
    global _db_connection_counter
    _db_connection_counter = getattr(get_db_connection, '_counter', 0) + 1
    setattr(get_db_connection, '_counter', _db_connection_counter)
    
    try:
        conn = pymysql.connect(**MYSQL_CONFIG)
        logger.debug(f"数据库连接成功，连接ID: {_db_connection_counter}")
        return conn
    except Exception as e:
        logger.error(f"数据库连接失败: {str(e)}")
        raise

def init_database():
    """初始化 MySQL 表"""
    try:
        server_conn = pymysql.connect(
            host=MYSQL_CONFIG["host"],
            port=MYSQL_CONFIG["port"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            charset=MYSQL_CONFIG["charset"],
            autocommit=True,
        )
    except pymysql.err.OperationalError as e:
        if getattr(e, "args", None) and len(e.args) >= 2 and e.args[0] == 1045:
            logger.error(
                "MySQL 鉴权失败：请在 .env 中配置 MYSQL_HOST/MYSQL_PORT/MYSQL_USER/MYSQL_PASSWORD/MYSQL_DATABASE。"
            )
        raise
    try:
        with server_conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        server_conn.close()

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(64) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                email VARCHAR(100),
                phone VARCHAR(20)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # 角色表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                role_name VARCHAR(50) UNIQUE NOT NULL,
                display_name VARCHAR(80),
                description TEXT,
                personality TEXT,
                tone TEXT,
                background_story TEXT,
                hobbies TEXT,
                prompt_template TEXT,
                creator_user_id BIGINT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'display_name'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN display_name VARCHAR(80) NULL AFTER role_name")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'personality'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN personality TEXT NULL AFTER description")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'tone'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN tone TEXT NULL AFTER personality")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'background_story'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN background_story TEXT NULL AFTER tone")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'hobbies'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN hobbies TEXT NULL AFTER background_story")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'creator_user_id'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN creator_user_id BIGINT NULL AFTER prompt_template")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'updated_at'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")
        
        # 初始化默认角色
        for role_name, template in ROLE_PROMPT_TEMPLATES.items():
            cursor.execute("SELECT id FROM roles WHERE role_name=%s", (role_name,))
            if not cursor.fetchone():
                display_name = ROLE_DISPLAY_NAMES.get(role_name, role_name)
                description = ROLE_DEFAULT_DESCRIPTIONS.get(role_name, "")
                personality = f"你是{display_name}，擅长基于专业信息给出可执行建议。"
                tone = "专业、耐心、清晰"
                background_story = f"你长期从事{display_name}相关工作，重视事实、逻辑与风险提示。"
                hobbies = "阅读行业报告、案例复盘、知识分享"
                cursor.execute(
                    """
                    INSERT INTO roles
                    (role_name, display_name, description, personality, tone, background_story, hobbies, prompt_template)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (role_name, display_name, description, personality, tone, background_story, hobbies, template)
                )
        
        # 知识库文档表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_docs (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                doc_id VARCHAR(100) UNIQUE NOT NULL,
                title VARCHAR(255),
                source VARCHAR(255),
                content TEXT,
                chunk_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS role_knowledge (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                role_id BIGINT NOT NULL,
                title VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                UNIQUE(role_id, title),
                FOREIGN KEY(role_id) REFERENCES roles(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cursor.execute("SHOW COLUMNS FROM role_knowledge LIKE 'title'")
        title_col = cursor.fetchone()
        if title_col:
            title_type = str(title_col[1]).lower()
            if "text" in title_type:
                # MySQL 的 TEXT 列无法直接用于 UNIQUE(role_id, title) 组合索引
                cursor.execute("ALTER TABLE role_knowledge MODIFY COLUMN title VARCHAR(255) NOT NULL")

        # 同步角色描述/提示词，并补齐缺失角色
        for role_name, template in ROLE_PROMPT_TEMPLATES.items():
            display_name = ROLE_DISPLAY_NAMES.get(role_name, role_name)
            description = ROLE_DEFAULT_DESCRIPTIONS.get(role_name, "")
            personality = f"你是{display_name}，擅长基于专业信息给出可执行建议。"
            tone = "专业、耐心、清晰"
            background_story = f"你长期从事{display_name}相关工作，重视事实、逻辑与风险提示。"
            hobbies = "阅读行业报告、案例复盘、知识分享"
            cursor.execute("SELECT id FROM roles WHERE role_name=%s", (role_name,))
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    """
                    UPDATE roles
                    SET display_name=%s, description=%s, personality=%s, tone=%s, background_story=%s, hobbies=%s, prompt_template=%s
                    WHERE role_name=%s
                    """,
                    (display_name, description, personality, tone, background_story, hobbies, template, role_name),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO roles
                    (role_name, display_name, description, personality, tone, background_story, hobbies, prompt_template)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (role_name, display_name, description, personality, tone, background_story, hobbies, template)
                )

        # 为每个角色灌入本地角色知识库，作为可直接连接的兜底数据库
        cursor.execute("SELECT id, role_name FROM roles")
        for role_id, role_name in cursor.fetchall():
            for title, content in ROLE_SQLITE_KNOWLEDGE.get(role_name, []):
                cursor.execute(
                    "INSERT IGNORE INTO role_knowledge (role_id, title, content) VALUES (%s, %s, %s)",
                    (role_id, title, content),
                )
        
        conn.commit()
        logger.info("MySQL 数据库初始化完成")
    finally:
        conn.close()

def register_user(username: str, password: str, email: Optional[str] = None, phone: Optional[str] = None) -> bool:
    """注册用户"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # 检查用户名是否已存在
        cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
        if cursor.fetchone():
            logger.info(f"用户注册失败：用户名已存在: {username}")
            return False
        
        # 生成密码哈希
        password_hash = hash_password(password)
        logger.debug(f"注册用户 {username}，密码哈希: {password_hash[:16]}...")
        
        # 插入用户记录
        cursor.execute(
            "INSERT INTO users (username, password_hash, email, phone) VALUES (%s, %s, %s, %s)",
            (username, password_hash, email, phone)
        )
        
        # 提交事务
        conn.commit()
        logger.info(f"用户注册成功: {username}, 受影响行数: {cursor.rowcount}")
        
        # 验证注册是否成功
        cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
        result = cursor.fetchone()
        if result:
            logger.debug(f"注册验证成功，用户ID: {result[0]}")
        else:
            logger.error(f"注册验证失败：用户 {username} 注册后查询不到")
        
        return True
    finally:
        conn.close()

def login_user(username: str, password: str) -> Optional[int]:
    """用户登录"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # 生成密码哈希
        password_hash = hash_password(password)
        logger.debug(f"登录用户 {username}，密码哈希: {password_hash[:16]}...")
        
        # 先检查用户是否存在
        cursor.execute("SELECT id, password_hash FROM users WHERE username=%s", (username,))
        user_record = cursor.fetchone()
        
        if not user_record:
            logger.info(f"用户登录失败：用户不存在: {username}")
            return None
        
        db_user_id, db_password_hash = user_record
        logger.debug(f"数据库中用户ID: {db_user_id}, 存储的密码哈希: {db_password_hash[:16]}...")
        
        # 比较密码哈希
        if password_hash == db_password_hash:
            logger.info(f"用户登录成功: {username}, 用户ID: {db_user_id}")
            return db_user_id
        else:
            logger.info(f"用户登录失败：密码错误: {username}")
            return None
    finally:
        conn.close()

def get_user_info(user_id: int) -> Optional[Dict[str, Any]]:
    """获取用户信息"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username, email, phone, created_at FROM users WHERE id=%s", (user_id,))
        result = cursor.fetchone()
        if result:
            return {
                "user_id": user_id,
                "username": result[0],
                "email": result[1],
                "phone": result[2],
                "created_at": result[3]
            }
        return None
    finally:
        conn.close()

def get_roles() -> List[Dict[str, Any]]:
    """获取所有角色"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                id, role_name, display_name, description, personality, tone, background_story, hobbies, prompt_template
            FROM roles
            ORDER BY id
        """)
        results = cursor.fetchall()
        return [{
            "id": r[0],
            "role_name": r[1],
            "display_name": r[2] or r[1],
            "description": r[3],
            "personality": r[4],
            "tone": r[5],
            "background_story": r[6],
            "hobbies": r[7],
            "prompt_template": r[8],
        } for r in results]
    finally:
        conn.close()

def get_role_prompt(role_id: int) -> Optional[str]:
    """获取角色提示词模板"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT role_name, display_name, personality, tone, background_story, hobbies, prompt_template
            FROM roles WHERE id=%s
            """,
            (role_id,),
        )
        result = cursor.fetchone()
        if not result:
            return None
        role = {
            "role_name": result[0],
            "display_name": result[1],
            "personality": result[2],
            "tone": result[3],
            "background_story": result[4],
            "hobbies": result[5],
            "prompt_template": result[6],
        }
        return role["prompt_template"] or _build_custom_role_prompt(role)
    finally:
        conn.close()


def get_role_name(role_id: int) -> Optional[str]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT role_name FROM roles WHERE id=%s", (role_id,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def retrieve_sqlite_role_knowledge(role_id: int, query: str, top_k: int = 3) -> List[str]:
    """
    从本地 SQLite 角色知识表中直接检索，作为无需额外服务的兜底数据库。
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT title, content FROM role_knowledge WHERE role_id=%s", (role_id,))
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    tokens = [t for t in simple_chinese_tokenizer(query) if t]
    scored: List[Tuple[int, str]] = []
    for title, content in rows:
        haystack = f"{title}\n{content}".lower()
        score = 0
        for token in tokens:
            if token.lower() in haystack:
                score += 2 if len(token) > 1 else 1
        if score == 0:
            score = 1
        scored.append((score, f"{title}：{content}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:top_k]]


def build_database_grounded_answer(
    role_id: int,
    message: str,
    sqlite_knowledge: List[str],
    stock_info: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    直接基于本地数据库命中内容组织一版自然语言回答，
    避免模型输出空泛开场白。
    """
    role_name = get_role_name(role_id) or "assistant"
    role_display = ROLE_DISPLAY_NAMES.get(role_name, role_name)
    stock_info = stock_info or []

    snippets = []
    for item in sqlite_knowledge[:3]:
        snippets.append(item.split("：", 1)[-1].strip())

    stock_lines = []
    for s in stock_info[:2]:
        stock_lines.append(
            f"{s['stock_name']}({s['stock_code']})当前公开数据显示最新价约{s['latest_price']}元，涨跌幅{s['change_pct']}%，市盈率{s['pe_ratio']}。"
        )

    if role_name == "doctor":
        body = "先根据你提到的症状给你一个常见情况的判断：\n" + "\n".join([f"{i+1}. {x}" for i, x in enumerate(snippets[:3])])
        return normalize_answer_text(
            f"你这个问题我直接按常见情况跟你说。\n\n{body}\n\n"
            "如果只是普通不适，一般先休息、补水、清淡饮食，药物以对症处理为主，不要自己叠加吃多种复方药。"
            " 但如果你同时有高热不退、呼吸困难、胸痛、意识模糊，或者症状越来越重，就不要继续拖，尽快去医院。"
            " 如果你愿意，我也可以继续根据你的具体症状，比如有没有发烧、咳嗽、流鼻涕、咽痛，帮你再细分一下。"
        )

    if role_name == "psychologist":
        body = "\n".join([f"{i+1}. {x}" for i, x in enumerate(snippets[:3])])
        return normalize_answer_text(
            f"我先接住你的感受。很多时候，一个人状态不好并不是因为你不够努力，而是已经累了、压住了太多情绪。\n\n"
            f"结合你现在的话题，我比较建议先做这几件事：\n{body}\n\n"
            "你不用一下子把自己调整到很好，先把状态从最难受拉回一点点就够了。"
            " 如果你愿意，可以直接告诉我是学习压力、关系问题，还是单纯觉得很烦很累，我会按那个方向继续陪你分析。"
        )

    if role_name == "virtual_friend":
        core = snippets[0] if snippets else "我在，咱们就顺着你现在想聊的继续。"
        return normalize_answer_text(
            f"{core}\n\n"
            f"你刚刚说的是“{message}”，我更想先听听你现在是随口一问，还是今天真的发生了什么。"
            " 如果你想轻松聊，我就陪你闲聊；如果你今天状态不太好，你也可以直接跟我说最卡在心里的那一件事。"
        )

    if role_name == "teacher":
        body = "\n".join([f"{i+1}. {x}" for i, x in enumerate(snippets[:3])])
        return normalize_answer_text(
            f"这个问题我给你按“先讲清楚，再举例子”的方式说。\n\n{body}\n\n"
            "如果你愿意，我还可以继续把它拆成更简单的版本，或者直接拿一道题/一句话来给你演示。"
        )

    if role_name == "lawyer":
        body = "\n".join([f"{i+1}. {x}" for i, x in enumerate(snippets[:3])])
        return normalize_answer_text(
            f"这个问题先按一般处理思路说，不直接下案件结论。\n\n{body}\n\n"
            "如果你要我继续帮你判断，最好补充一下地区、合同/借条/聊天记录/付款记录这些关键信息，我可以按证据和处理步骤继续帮你梳理。"
        )

    if role_name in {"stock_analyst", "investment_advisor", "financial_advisor", "financial_planner"}:
        body = "\n".join([f"{i+1}. {x}" for i, x in enumerate(snippets[:3])])
        extra = ("\n\n补充一点公开市场数据：\n" + "\n".join(stock_lines)) if stock_lines else ""
        return normalize_answer_text(
            f"这个问题可以直接从几个关键点来看：\n{body}{extra}\n\n"
            "如果你愿意，再补充你的目标、期限和风险承受能力，我可以把建议说得更具体一些。"
        )

    return normalize_answer_text(
        f"我先结合这个角色的知识库给你一个直接回答：\n\n" +
        "\n".join([f"{i+1}. {x}" for i, x in enumerate(snippets[:3])]) +
        "\n\n如果你愿意，我可以继续顺着你这句话往下展开。"
    )


def normalize_answer_text(text: str) -> str:
    """
    清洗模型回答，去掉 JSON 包装、markdown 符号和多余引号。
    """
    if text is None:
        return ""
    cleaned = str(text).strip()

    # 处理 {"response":"..."} 这类字符串化 JSON
    try:
        import json
        if cleaned.startswith("{") and cleaned.endswith("}"):
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                for key in ("response", "answer", "content", "text"):
                    value = obj.get(key)
                    if isinstance(value, str) and value.strip():
                        cleaned = value.strip()
                        break
    except Exception:
        pass

    cleaned = re.sub(r'^\s*\{\s*"?(response|answer|content|text)"?\s*:\s*"?', "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'"\s*\}\s*$', "", cleaned)
    cleaned = cleaned.replace('\\"', '"')
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("#", "")
    cleaned = cleaned.replace("*", "")
    return cleaned.strip(' "\n\t')

# ==================== Milvus 向量数据库操作 ====================
# 负责金融知识库的向量存储、检索，以及长期记忆管理

def init_milvus(max_retries: int = 3, retry_delay: int = 2):
    """
    初始化 Milvus 连接和集合，支持重试机制和多种配置
    
    功能：
    - 建立与 Milvus 服务器的连接
    - 创建角色知识库集合（按 role_id 隔离）
    - 创建长期记忆集合
    - 创建股票数据集合
    """
    global _milvus_available
    
    logger.info(f"📋 Milvus 配置信息: host={MILVUS_HOST}, port={MILVUS_PORT}, ssl={MILVUS_USE_SSL}, timeout={MILVUS_TIMEOUT}s")
    
    # 构建连接参数
    connect_params = {
        "alias": _MILVUS_CONNECTION_ALIAS,
        "host": MILVUS_HOST,
        "port": MILVUS_PORT,
        "timeout": MILVUS_TIMEOUT,
    }
    
    # 添加认证信息（如果配置了）
    if MILVUS_USER:
        connect_params["user"] = MILVUS_USER
        logger.info("🔐 Milvus 认证已启用")
    if MILVUS_PASSWORD:
        connect_params["password"] = MILVUS_PASSWORD
    
    # 添加SSL配置
    if MILVUS_USE_SSL:
        connect_params["secure"] = True
        logger.info("🔒 Milvus SSL连接已启用")
    
    # 尝试连接 Milvus，支持重试
    last_error = None
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 连接 Milvus (第 {attempt + 1}/{max_retries} 次尝试): {MILVUS_HOST}:{MILVUS_PORT}")
            
            # 先断开可能存在的旧连接
            try:
                connections.disconnect(_MILVUS_CONNECTION_ALIAS)
            except Exception:
                pass
            
            # 建立新连接
            connections.connect(**connect_params)
            
            # 验证连接
            version = utility.get_server_version()
            logger.info(f"✅ Milvus 连接成功，版本: {version}")
            break
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            logger.warning(f"❌ Milvus 连接失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
            
            # 分析错误原因并给出建议
            _analyze_milvus_error(error_msg, attempt)
            
            if attempt < max_retries - 1:
                logger.info(f"⏳ 等待 {retry_delay} 秒后重试...")
                import time
                time.sleep(retry_delay)
            else:
                # 最后一次尝试失败，尝试本地模式
                if MILVUS_ENABLE_LOCAL:
                    logger.info("🔧 尝试使用本地存储模式...")
                    if _try_local_milvus():
                        return
                raise last_error
    
    # 连接成功后，初始化集合
    _init_milvus_collections()
    
    _milvus_available = True


def _analyze_milvus_error(error_msg: str, attempt: int):
    """分析Milvus连接错误并给出建议"""
    if "connection refused" in error_msg.lower() or "fail connecting" in error_msg.lower():
        logger.error("🔍 连接被拒绝！请检查：")
        logger.error("   1. Milvus 服务是否正在运行")
        logger.error("   2. Milvus 主机地址是否正确")
        logger.error("   3. Milvus 端口是否正确 (默认19530)")
        logger.error("   4. 防火墙是否允许访问")
    elif "timeout" in error_msg.lower():
        logger.error("🔍 连接超时！请检查：")
        logger.error("   1. 网络连接是否稳定")
        logger.error("   2. Milvus 服务是否过载")
        logger.error("   3. 考虑增加 MILVUS_TIMEOUT 环境变量")
    elif "authentication" in error_msg.lower() or "invalid token" in error_msg.lower():
        logger.error("🔍 认证失败！请检查：")
        logger.error("   1. MILVUS_USER 环境变量是否正确")
        logger.error("   2. MILVUS_PASSWORD 环境变量是否正确")
    elif attempt == 0:
        # 首次失败时给出通用建议
        logger.info("💡 提示：如果使用 Milvus Standalone，确保已启动服务：")
        logger.info("   - Linux: ./milvus start")
        logger.info("   - Docker: docker-compose up -d")


def _try_local_milvus():
    """尝试使用本地文件存储模式（作为备用方案）"""
    try:
        from pymilvus import connections, utility
        logger.info("尝试使用本地存储模式连接...")
        
        # 使用本地存储路径
        local_params = {
            "alias": _MILVUS_CONNECTION_ALIAS,
            "host": MILVUS_HOST,
            "port": MILVUS_PORT,
            "timeout": MILVUS_TIMEOUT * 2,
        }
        
        if MILVUS_USER:
            local_params["user"] = MILVUS_USER
        if MILVUS_PASSWORD:
            local_params["password"] = MILVUS_PASSWORD
        
        connections.connect(**local_params)
        version = utility.get_server_version()
        logger.info(f"✅ 本地模式连接成功，版本: {version}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 本地模式连接也失败: {e}")
        return False


def _init_milvus_collections():
    """初始化所有Milvus集合"""
    # 知识库集合：按 role_id 隔离，避免不同角色检索互相干扰
    try:
        roles = get_roles()
        logger.info(f"📚 初始化 {len(roles)} 个角色知识库集合...")
        for r in roles:
            ensure_knowledge_collection(int(r["id"]))
        logger.info("✅ 角色知识库集合初始化完成")
    except Exception as e:
        # 若角色表不可用，至少保证 role_id=1 可用
        logger.warning(f"⚠️ 初始化角色知识库集合失败，降级创建 role_id=1: {e}")
        ensure_knowledge_collection(1)
    
    # 创建长期记忆集合
    try:
        _create_long_term_memory_collection()
        logger.info("✅ 长期记忆集合初始化完成")
    except Exception as e:
        logger.error(f"❌ 创建长期记忆集合失败: {e}")
        raise
    
    # 创建股票数据集合
    try:
        _create_stock_collection()
        logger.info("✅ 股票数据集合初始化完成")
    except Exception as e:
        logger.error(f"❌ 创建股票数据集合失败: {e}")
        raise


def _create_long_term_memory_collection():
    """创建长期记忆集合"""
    memory_collection_name = "long_term_memory"
    if not utility.has_collection(memory_collection_name):
        logger.info(f"创建长期记忆集合: {memory_collection_name}")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.INT64),
            FieldSchema(name="role_id", dtype=DataType.INT64),
            FieldSchema(name="memory_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512)
        ]
        schema = CollectionSchema(fields, "长期记忆集合")
        collection = Collection(memory_collection_name, schema)
        _create_index_and_load(collection, "dense_vector")
        logger.info(f"长期记忆集合 {memory_collection_name} 创建完成")
    else:
        # 确保已有集合已加载
        collection = Collection(memory_collection_name)
        _ensure_collection_loaded(collection, memory_collection_name)


def _create_stock_collection():
    """创建股票数据集合"""
    stock_collection_name = "stock_data"
    if not utility.has_collection(stock_collection_name):
        logger.info(f"创建股票数据集合: {stock_collection_name}")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="stock_code", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="stock_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="sector", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="fetch_time", dtype=DataType.INT64),
            FieldSchema(name="latest_price", dtype=DataType.DOUBLE),
            FieldSchema(name="change_pct", dtype=DataType.DOUBLE),
            FieldSchema(name="pe_ratio", dtype=DataType.DOUBLE),
            FieldSchema(name="market_cap", dtype=DataType.DOUBLE)
        ]
        schema = CollectionSchema(fields, "股票数据集合")
        collection = Collection(stock_collection_name, schema)
        _create_index_and_load(collection, "dense_vector")
        logger.info(f"股票数据集合 {stock_collection_name} 创建完成")
    else:
        # 确保已有集合已加载
        collection = Collection(stock_collection_name)
        _ensure_collection_loaded(collection, stock_collection_name)


def _create_index_and_load(collection: Collection, field_name: str):
    """创建索引并加载集合，封装通用逻辑"""
    # 检查索引是否已存在
    index_info = collection.indexes
    if not any(idx.field_name == field_name for idx in index_info):
        collection.create_index(
            field_name,
            {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
        )
    # 加载集合
    _ensure_collection_loaded(collection)

def init_long_term_memory():
    """确保长期记忆集合存在"""
    if not milvus_ready():
        return
    memory_collection_name = "long_term_memory"
    if not utility.has_collection(memory_collection_name):
        logger.info(f"创建长期记忆集合: {memory_collection_name}")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.INT64),
            FieldSchema(name="role_id", dtype=DataType.INT64),
            FieldSchema(name="memory_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512)
        ]
        schema = CollectionSchema(fields, "长期记忆集合")
        collection = Collection(memory_collection_name, schema)
        collection.create_index(
            "dense_vector",
            {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
        )
        collection.load()

def simple_chinese_tokenizer(text: str) -> list:
    """简单的中文分词器（避免依赖NLTK）"""
    import re
    
    # 移除特殊字符，保留中文、英文和数字
    text = re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z\s]', ' ', text)
    
    tokens = []
    
    # 先按空白字符分割
    parts = text.split()
    
    for part in parts:
        # 对每个部分进行分词
        i = 0
        while i < len(part):
            # 检查是否是中文字符
            if '\u4e00' <= part[i] <= '\u9fff':
                # 中文字符单独作为一个token
                tokens.append(part[i])
                i += 1
            # 检查是否是英文或数字
            elif part[i].isalnum():
                # 连续的英文/数字作为一个token
                j = i
                while j < len(part) and part[j].isalnum():
                    j += 1
                tokens.append(part[i:j].lower())
                i = j
            else:
                i += 1
    
    # 过滤长度为1的英文单词（通常是噪音）
    tokens = [t for t in tokens if not (len(t) == 1 and t.isalpha())]
    
    return tokens

def rebuild_bm25_model():
    """兼容旧调用：默认不做事（已改为按角色/collection 维护 BM25）。"""
    return


def rebuild_bm25_model_for_collection(collection_name: str):
    """为某个知识库 collection 重建 BM25 模型。"""
    global _bm25_models
    try:
        collection = Collection(collection_name)
        if collection.num_entities == 0:
            logger.warning(f"知识库为空，跳过 BM25 模型重建: {collection_name}")
            return
        
        logger.info(f"开始重建 BM25 模型: {collection_name}")
        
        # 使用默认分析器（避免NLTK依赖问题）
        bm25 = BM25EmbeddingFunction()
        
        # 获取所有文本
        results = collection.query(expr="id >= 0", output_fields=["text"])
        texts = [r["text"] for r in results]
        
        # 构建 BM25 模型
        bm25.fit(texts)
        _bm25_models[collection_name] = bm25
        logger.info(f"BM25 模型重建完成: {collection_name}，包含 {len(texts)} 条文档")
    except Exception as e:
        logger.warning(f"⚠️ 重建 BM25 模型失败: {collection_name}: {e}")


def add_documents_to_milvus(texts: List[str], source: str, doc_id: str = None, role_id: int = 1):
    """添加文档到 Milvus（写入指定 role_id 的知识库集合）"""
    if not texts:
        return
    
    collection = ensure_knowledge_collection(role_id)
    model = get_embedding_model()
    
    dense_vectors = model.encode(texts, normalize_embeddings=True)
    
    bm25 = _bm25_models.get(collection.name)
    if bm25:
        sparse_vectors = bm25.encode_documents(texts)
    else:
        sparse_vectors = [{} for _ in texts]
    
    create_time = int(datetime.now().timestamp())
    data = []
    for idx, text in enumerate(texts):
        summary = text[:50] + "..." if len(text) > 50 else text
        data.append({
            "text": text,
            "dense_vector": dense_vectors[idx].tolist(),
            "sparse_vector": sparse_vectors[idx],
            "source": source,
            "create_time": create_time,
            "summary": summary,
            "doc_id": doc_id or f"doc_{create_time}_{idx}"
        })
    
    collection.insert(data)
    collection.flush()
    
    # 重建 BM25 模型
    rebuild_bm25_model_for_collection(collection.name)
    
    logger.info(f"已添加 {len(texts)} 条文档到知识库: role_id={role_id}")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """文本分块"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # 尽量在句子边界分割
        if end < text_length:
            # 找最近的句号、问号、叹号
            for sep in ['。', '！', '？', '.', '!', '?', '\n']:
                idx = text.rfind(sep, start, end)
                if idx != -1 and idx > start + chunk_overlap:
                    end = idx + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - chunk_overlap
    
    return chunks

def hybrid_search(query: str, role_id: int, top_k: int = TOP_K_RETRIEVE) -> List[Dict[str, str]]:
    """混合检索：稠密向量 + BM25 稀疏向量"""
    if not milvus_ready():
        logger.debug("Milvus 不可用，跳过向量检索")
        return []
    
    try:
        collection = ensure_knowledge_collection(role_id)
        
        # 确保集合已加载
        _ensure_collection_loaded(collection)
        
        # 检查是否有数据
        if collection.num_entities == 0:
            logger.debug(f"集合 {collection.name} 为空，返回空结果")
            return []
        
        dense_model = get_embedding_model()
        
        # 稠密向量检索
        query_dense_vec = dense_model.encode([query], normalize_embeddings=True)[0].tolist()
        dense_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        dense_req = AnnSearchRequest(
            data=[query_dense_vec],
            anns_field="dense_vector",
            param=dense_search_params,
            limit=top_k
        )
        
        # 稀疏向量检索（BM25）
        bm25 = _bm25_models.get(collection.name)
        # 懒加载 BM25：如果知识库有内容但 BM25 尚未构建，则先构建一次
        if bm25 is None and collection.num_entities > 0:
            rebuild_bm25_model_for_collection(collection.name)
            bm25 = _bm25_models.get(collection.name)
        
        if bm25:
            try:
                import scipy.sparse as sp
                sparse_mat = bm25.encode_queries([query])
                query_sparse_vec = sparse_mat.getrow(0)
                if sp.issparse(query_sparse_vec):
                    query_sparse_vec.data[query_sparse_vec.data < 0] = 0.0
                    query_sparse_vec.eliminate_zeros()
                sparse_search_params = {"metric_type": "IP"}
                sparse_req = AnnSearchRequest(
                    data=[query_sparse_vec],
                    anns_field="sparse_vector",
                    param=sparse_search_params,
                    limit=top_k
                )
                
                # RRF 融合
                rerank = RRFRanker()
                results = collection.hybrid_search(
                    reqs=[dense_req, sparse_req],
                    rerank=rerank,
                    limit=top_k,
                    output_fields=["text", "source", "create_time", "summary"]
                )
            except Exception as e:
                logger.warning(f"稀疏检索失败，使用纯稠密检索: {e}")
                results = _execute_dense_search(collection, query_dense_vec, dense_search_params, top_k)
        else:
            logger.debug("BM25 模型未构建，仅使用稠密检索")
            results = _execute_dense_search(collection, query_dense_vec, dense_search_params, top_k)
        
        return _parse_search_results(results)
    
    except Exception as e:
        logger.error(f"混合检索失败: {e}")
        return []


def _execute_dense_search(collection: Collection, query_vec: list, params: dict, top_k: int):
    """执行稠密向量检索"""
    return collection.search(
        data=[query_vec],
        anns_field="dense_vector",
        param=params,
        limit=top_k,
        output_fields=["text", "source", "create_time", "summary"]
    )


def _parse_search_results(results) -> List[Dict[str, str]]:
    """解析检索结果"""
    retrieved = []
    for hits in results:
        for hit in hits:
            retrieved.append({
                "text": hit.entity.get("text"),
                "source": hit.entity.get("source"),
                "create_time": hit.entity.get("create_time"),
                "summary": hit.entity.get("summary")
            })
    return retrieved

def rerank_documents(query: str, documents: List[Dict[str, str]], top_n: int = TOP_K_RERANK) -> List[Dict[str, str]]:
    """使用 BGE-rerank 重排序文档"""
    if not documents:
        return []
    
    # 获取重排序模型
    reranker = get_reranker_model()
    if reranker is None:
        logger.debug("重排序模型不可用，使用原始顺序")
        return documents[:top_n]
    
    try:
        # 准备文档对
        pairs = [[query, doc["text"]] for doc in documents]
        
        # 尝试计算分数
        scores = None
        try:
            scores = reranker.compute_score(pairs)
        except Exception as e:
            logger.debug(f"标准计算失败，尝试分批处理: {e}")
            # 分批处理避免内存问题
            batch_size = 10
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                try:
                    batch_scores = reranker.compute_score(batch)
                    scores.extend(batch_scores)
                except Exception as batch_e:
                    logger.debug(f"批次处理失败: {batch_e}")
                    # 如果分批也失败，返回原始顺序
                    return documents[:top_n]
        
        # 按分数排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前 top_n 个
        result = [doc for doc, score in scored_docs[:top_n]]
        logger.debug(f"重排序完成，原始 {len(documents)} 条，返回 {len(result)} 条")
        return result
        
    except Exception as e:
        logger.warning(f"重排序失败，使用原始顺序: {e}")
        return documents[:top_n]

def retrieve_long_term_memory(user_id: int, role_id: int, query: str, top_k: int = TOP_K_LONG_TERM) -> List[str]:
    """检索长期记忆"""
    if not milvus_ready():
        logger.debug("Milvus 不可用，跳过长期记忆检索")
        return []
    
    try:
        collection = Collection("long_term_memory")
        
        # 确保集合已加载
        _ensure_collection_loaded(collection)
        
        # 检查是否有数据
        if collection.num_entities == 0:
            return []
        
        model = get_embedding_model()
        query_vec = model.encode([query], normalize_embeddings=True)[0].tolist()
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        # 构建过滤条件
        expr = f"user_id == {user_id} && role_id == {role_id}"
        
        results = collection.search(
            data=[query_vec],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["memory_text"]
        )
        
        memories = []
        for hits in results:
            for hit in hits:
                memories.append(hit.entity.get("memory_text"))
        
        return memories
    
    except Exception as e:
        logger.error(f"检索长期记忆失败: {e}")
        return []


def store_long_term_memory(user_id: int, role_id: int, user_msg: str, assistant_msg: str):
    """存储长期记忆"""
    if not milvus_ready():
        logger.debug("Milvus 不可用，跳过长期记忆存储")
        return
    
    try:
        collection = Collection("long_term_memory")
        
        # 确保集合已加载
        _ensure_collection_loaded(collection)
        
        model = get_embedding_model()
        
        memory_text = f"用户问: {user_msg}\n助手回答: {assistant_msg}"
        vector = model.encode([memory_text], normalize_embeddings=True)[0].tolist()
        summary = user_msg[:100] + "..." if len(user_msg) > 100 else user_msg
        
        data = [{
            "user_id": user_id,
            "role_id": role_id,
            "memory_text": memory_text,
            "dense_vector": vector,
            "timestamp": int(datetime.now().timestamp()),
            "summary": summary
        }]
        collection.insert(data)
        collection.flush()
        logger.info(f"已存储长期记忆: user={user_id}, role={role_id}")
    
    except Exception as e:
        logger.error(f"存储长期记忆失败: {e}")

# ==================== 短期记忆（Redis）操作 ====================
# 使用 Redis 存储短期对话历史，支持快速读写

def get_chat_history(user_id: int, role_id: int) -> List[Dict[str, str]]:
    """获取聊天历史（短期记忆）"""
    r = get_redis_client()
    key = f"chat:{user_id}:{role_id}"
    if r is None:
        return _in_memory_history.get(key, [])
    try:
        history_str = r.get(key)
        if history_str:
            import json
            return json.loads(history_str)
    except Exception as e:
        logger.warning(f"Redis数据类型错误，清理无效数据: {e}")
        try:
            r.delete(key)
        except:
            pass
    return []

def add_to_history(user_id: int, role_id: int, user_msg: str, assistant_msg: str):
    """添加到聊天历史"""
    r = get_redis_client()
    key = f"chat:{user_id}:{role_id}"
    
    history = get_chat_history(user_id, role_id)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    
    # 保留最近 MEMORY_LENGTH 轮对话
    if len(history) > MEMORY_LENGTH * 2:
        history = history[-MEMORY_LENGTH * 2:]
    if r is None:
        _in_memory_history[key] = history
    else:
        import json
        r.set(key, json.dumps(history))

def clear_history(user_id: int, role_id: int):
    """清空聊天历史"""
    r = get_redis_client()
    key = f"chat:{user_id}:{role_id}"
    if r is None:
        _in_memory_history.pop(key, None)
    else:
        r.delete(key)


def _is_port_available(host: str, port: int) -> bool:
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def resolve_server_port(default_port: int = 8001, host: str = "127.0.0.1") -> int:
    """解析可用端口，避免端口冲突导致启动失败"""
    env_port = os.getenv("SERVER_PORT")
    preferred_port = default_port
    if env_port:
        try:
            preferred_port = int(env_port)
        except ValueError:
            logger.warning(f"SERVER_PORT 配置非法: {env_port}，使用默认端口 {default_port}")
            preferred_port = default_port

    if _is_port_available(host, preferred_port):
        return preferred_port

    logger.warning(f"端口 {preferred_port} 已被占用，尝试自动查找可用端口...")
    for offset in range(1, 50):
        candidate = preferred_port + offset
        if _is_port_available(host, candidate):
            logger.info(f"已切换到可用端口: {candidate}")
            return candidate

    raise RuntimeError(f"无法找到可用端口（起始端口: {preferred_port}）")

# ==================== 股票数据相关函数 ====================
# 从东方财富获取股票数据，存储到 Milvus 并提供检索功能
STOCK_COLLECTION_NAME = "stock_data"

def fetch_stock_data_from_eastmoney() -> List[Dict[str, Any]]:
    """从东方财富网获取股票数据"""
    try:
        import akshare as ak
        logger.info("开始从 AKShare 获取股票数据...")
        df = ak.stock_zh_a_spot_em()
        
        if df.empty:
            logger.warning("获取到的股票数据为空")
            return []

        stock_list = []
        for _, row in df.iterrows():
            stock_info = {
                "stock_code": str(row.get("代码", "")).strip(),
                "stock_name": str(row.get("名称", "")).strip(),
                "latest_price": float(row.get("最新价", 0)),
                "change_pct": float(row.get("涨跌幅", 0)),
                "volume": float(row.get("成交量", 0)),
                "turnover": float(row.get("成交额", 0)),
                "pe_ratio": float(row.get("市盈率-动态", 0)),
                "market_cap": float(row.get("总市值", 0)),
                "circulating_cap": float(row.get("流通市值", 0)),
                "sector": str(row.get("所属行业", "")).strip()
            }
            stock_list.append(stock_info)

        logger.info(f"成功获取 {len(stock_list)} 只股票数据")
        return stock_list
    except Exception as e:
        logger.error(f"获取股票数据失败: {e}")
        return []

def store_stock_data_to_milvus(stock_list: List[Dict[str, Any]]):
    """存储股票数据到 Milvus"""
    if not stock_list:
        return

    collection = Collection(STOCK_COLLECTION_NAME)
    model = get_embedding_model()

    texts = []
    for stock in stock_list:
        text = f"{stock['stock_name']}({stock['stock_code']}) 最新价{stock['latest_price']}元 涨跌幅{stock['change_pct']}% 市盈率{stock['pe_ratio']}"
        texts.append(text)

    vectors = model.encode(texts, normalize_embeddings=True)

    data = []
    fetch_time = int(datetime.now().timestamp())
    for idx, stock in enumerate(stock_list):
        data.append({
            "stock_code": stock["stock_code"],
            "stock_name": stock["stock_name"],
            "text": texts[idx],
            "dense_vector": vectors[idx].tolist(),
            "sector": stock.get("sector", ""),
            "fetch_time": fetch_time,
            "latest_price": stock["latest_price"],
            "change_pct": stock["change_pct"],
            "pe_ratio": stock["pe_ratio"],
            "market_cap": stock.get("market_cap", 0)
        })

    # 清空旧数据
    collection.delete("id >= 0")
    collection.insert(data)
    collection.flush()
    logger.info(f"已存储 {len(stock_list)} 条股票数据")

def sync_stock_data_to_milvus():
    """同步股票数据到 Milvus"""
    stock_list = fetch_stock_data_from_eastmoney()
    if stock_list:
        store_stock_data_to_milvus(stock_list)
    else:
        logger.warning("未获取到股票数据，同步失败")


def retrieve_stock_data_from_local_cache(query: str, top_k: int = TOP_K_STOCK) -> List[Dict[str, Any]]:
    """从本地 JSON 数据中直接检索股票信息，避免依赖向量库。"""
    stocks = _load_local_stock_cache()
    if not stocks:
        return []

    tokens = [t.lower() for t in simple_chinese_tokenizer(query) if t]
    query_text = (query or "").lower()
    ranked: List[Tuple[int, Dict[str, Any]]] = []
    for item in stocks:
        code = str(item.get("stock_code") or item.get("代码") or item.get("code") or "").strip()
        name = str(item.get("stock_name") or item.get("名称") or item.get("name") or "").strip()
        sector = str(item.get("sector") or item.get("所属行业") or item.get("industry") or "").strip()
        haystack = f"{code} {name} {sector}".lower()
        score = 0
        if code and code in query_text:
            score += 10
        if name and name.lower() in query_text:
            score += 8
        for token in tokens:
            if token and token in haystack:
                score += 2 if len(token) > 1 else 1
        if score <= 0:
            continue
        ranked.append((
            score,
            {
                "stock_code": code,
                "stock_name": name,
                "latest_price": item.get("latest_price") or item.get("最新价") or item.get("price") or 0,
                "change_pct": item.get("change_pct") or item.get("涨跌幅") or item.get("changePercent") or 0,
                "pe_ratio": item.get("pe_ratio") or item.get("市盈率") or item.get("市盈率-动态") or 0,
                "sector": sector,
            },
        ))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [stock for _, stock in ranked[:top_k]]

def retrieve_stock_data(query: str, top_k: int = TOP_K_STOCK) -> List[Dict[str, Any]]:
    """检索股票数据"""
    if not milvus_ready():
        return retrieve_stock_data_from_local_cache(query, top_k=top_k)
    collection = Collection(STOCK_COLLECTION_NAME)
    model = get_embedding_model()
    
    query_vec = model.encode([query], normalize_embeddings=True)[0].tolist()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    
    try:
        results = collection.search(
            data=[query_vec],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            output_fields=["stock_code", "stock_name", "latest_price", "change_pct", "pe_ratio", "sector"]
        )
        
        stocks = []
        for hits in results:
            for hit in hits:
                stocks.append({
                    "stock_code": hit.entity.get("stock_code"),
                    "stock_name": hit.entity.get("stock_name"),
                    "latest_price": hit.entity.get("latest_price"),
                    "change_pct": hit.entity.get("change_pct"),
                    "pe_ratio": hit.entity.get("pe_ratio"),
                    "sector": hit.entity.get("sector")
                })
        
        return stocks
    except Exception as e:
        logger.warning(f"检索股票数据失败: {e}")
        return retrieve_stock_data_from_local_cache(query, top_k=top_k)

# ==================== 大模型调用 ====================
# 支持多种大模型：DeepSeek、字节豆包、阿里通义千问、OpenAI
# 提供客户端缓存、Token缓存、流式输出等功能

# LLM客户端缓存
_llm_clients = {}
_doubao_token_info = {"token": None, "expire_time": 0}

def get_llm_client(provider: str = None):
    """
    获取大模型客户端（带缓存，避免重复创建连接）
    
    支持的提供商：
    - deepseek: 深度求索
    - doubao: 字节豆包
    - qianwen: 阿里通义千问
    - chatgpt: OpenAI
    """
    provider = provider or LLM_PROVIDER
    
    # 检查缓存
    if provider in _llm_clients:
        return _llm_clients[provider]
    
    import time
    
    if provider == "deepseek":
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    elif provider == "doubao":
        # 豆包需要先获取token（带缓存）
        now = time.time()
        if not _doubao_token_info["token"] or now >= _doubao_token_info["expire_time"]:
            # 需要重新获取token
            import requests
            try:
                url = "https://aip.baidubce.com/oauth/2.0/token"
                params = {
                    "grant_type": "client_credentials",
                    "client_id": DOUBAO_API_KEY,
                    "client_secret": DOUBAO_SECRET_KEY
                }
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                _doubao_token_info["token"] = data.get("access_token")
                # token有效期通常是30天，设置为29天
                _doubao_token_info["expire_time"] = now + (29 * 24 * 3600)
                logger.info("✅ 豆包Token获取成功")
            except Exception as e:
                logger.error(f"❌ 豆包Token获取失败: {e}")
                raise
        
        client = OpenAI(api_key=_doubao_token_info["token"], base_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro")
    elif provider == "qianwen":
        client = OpenAI(api_key=QIANWEN_API_KEY, base_url="https://dashscope.io/api/compatible-mode/v1")
    elif provider == "chatgpt":
        client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")
    else:
        raise ValueError(f"不支持的大模型提供商: {provider}")
    
    # 缓存客户端
    _llm_clients[provider] = client
    return client


def get_llm_model_name(provider: str = None) -> str:
    provider = provider or LLM_PROVIDER
    model_mapping = {
        "deepseek": "deepseek-chat",
        "doubao": "ernie-4.0-8k",
        "qianwen": "qwen-plus",
        "chatgpt": "gpt-4o-mini",
    }
    return model_mapping.get(provider, "deepseek-chat")

def rewrite_query(query: str, history: List[Dict[str, str]]) -> str:
    """Query 改写（结合历史对话）"""
    if not history:
        return query
    # 轻量改写：避免对在线模型产生额外依赖/延迟，优先直接返回
    # （如后续需要更强效果，可开启 LLM 改写）
    return query
def is_probably_not_a_question(text: str) -> bool:
    """粗略判断用户是否尚未提出问题（仅寒暄/自我介绍等）"""
    t = (text or "").strip()
    if not t:
        return True
    if "?" in t or "？" in t:
        return False

    # 太长的一般不当作“仅寒暄”
    if len(t) >= 40:
        return False

    # 常见寒暄/自我介绍模式
    greetings = r"^(你好|您好|嗨|hi|hello|在吗|在不在)\b"
    intro = r"(我叫|我是|叫我|昵称是|名字是)\s*[\u4e00-\u9fffA-Za-z0-9·]{1,12}$"
    if re.search(greetings, t, flags=re.IGNORECASE):
        return True
    if re.search(intro, t):
        return True

    # 仅把非常典型的寒暄判为“未提出问题”，
    # 像“心情不好”“房地产行业”“想学英语”“回答问题”这种短句也应进入有效对话。
    if t in {"你好", "您好", "嗨", "hi", "hello", "在吗", "在不在"}:
        return True

    # 排除明确要求回答的指令
    action_words = {"回答", "帮忙", "请问", "请教", "想知道", "了解", "分析", "建议", "推荐", "查询"}
    for word in action_words:
        if word in t:
            return False

    return False


def polite_clarify_answer(text: str) -> str:
    name_match = re.search(r"(我叫|我是|叫我|昵称是|名字是)\s*([\u4e00-\u9fffA-Za-z0-9·]{1,12})", (text or "").strip())
    name = name_match.group(2) if name_match else None
    if name:
        return (
            f"你好，{name}！很高兴认识你。\n\n"
            "你想咨询哪个具体问题？比如：\n"
            "1）我现在适合做怎样的资产配置？（可说明收入/支出/负债/目标）\n"
            "2）某个概念/指标怎么理解？（如 ROE、市盈率、基金回撤）\n"
            "3）某只股票/行业想了解客观信息？（我可以帮你整理公开信息与风险点）\n\n"
            "你把问题直接发我就行。"
        )
    return (
        "你好！我在。\n\n"
        "你想咨询哪个具体问题？你把问题直接发我就行（比如资产配置、指标解读、行业/公司客观信息梳理等）。"
    )


def build_llm_messages(
    role_id: int,
    message: str,
    context: str,
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    prompt_template = get_role_prompt(role_id) or ROLE_PROMPT_TEMPLATES["financial_advisor"]
    role_name = get_role_name(role_id) or "assistant"
    role_display = ROLE_DISPLAY_NAMES.get(role_name, role_name)

    system_prompt = f"""
你现在扮演{role_display}。

角色要求：
{prompt_template}

通用输出规则：
1. 直接回答用户当前问题，不要重复自我介绍，不要寒暄套话。
2. 优先使用提供的数据库/检索信息；信息不足时，先明确说明已知内容，再补充通用判断。
3. 回答要像真实对话，语言自然、具体、连续，不要机械复述提示词。
4. 不输出 JSON、Markdown 标题、代码块、字段名、星号、井号或无意义符号。
5. 如果用户问题可以落到明确建议，就先给结论，再解释原因；如果缺关键条件，最后只追问 1 个最必要的问题。
6. 除非用户明确要求，不要空泛地说“如有需要我可以继续帮助你”。
""".strip()

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if context and context.strip():
        messages.append({"role": "system", "content": f"以下是可参考的数据库与检索结果，请优先吸收后再回答：\n{context}"})

    for item in history[-8:]:
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "user", "content": message})
    return messages

def generate_response(role_id: int, user_id: int, message: str, context: str, history: List[Dict[str, str]], stream: bool = False, llm_provider: str = None):
    """生成回答"""
    messages = build_llm_messages(role_id, message, context, history)
    client = get_llm_client(llm_provider)
    model_name = get_llm_model_name(llm_provider)
    
    if stream:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1400,
            temperature=0.35,
            stream=True
        )

        async def generate_and_store():
            """流式输出同时累积全文，结束后写入记忆。"""
            full_text_parts: List[str] = []
            try:
                logger.info("开始流式输出...")
                for chunk in response:
                    delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                    if delta:
                        logger.debug(f"流式输出: {delta[:20]}...")
                        full_text_parts.append(delta)
                        yield delta
                logger.info(f"流式输出完成，总长度: {len(''.join(full_text_parts))}")
            finally:
                try:
                    full_text = normalize_answer_text("".join(full_text_parts))
                    if full_text:
                        add_to_history(user_id, role_id, message, full_text)
                        store_long_term_memory(user_id, role_id, message, full_text)
                except Exception as e:
                    logger.warning(f"流式回答落库失败: {e}")

        return StreamingResponse(generate_and_store(), media_type="text/plain; charset=utf-8")
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1400,
            temperature=0.35
        )
        return normalize_answer_text(response.choices[0].message.content.strip())

# ==================== FastAPI 应用 ====================
# Web 服务主入口，提供 RESTful API 接口

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    启动时执行：
    1. 初始化 MySQL 数据库（用户、角色表）
    2. 初始化 Milvus（知识库、长期记忆、股票数据集合）
    3. 初始化 BM25 模型
    
    关闭时执行：
    - 断开 Milvus 连接
    """
    global _milvus_available
    logger.info("=== 开始初始化应用 ===")
    
    # 1. 初始化 MySQL 数据库
    logger.info("初始化 MySQL 数据库...")
    try:
        init_database()
        logger.info("✅ MySQL 数据库初始化成功")
    except Exception as e:
        logger.error(f"❌ MySQL 数据库初始化失败: {e}")
        raise
    
    # 2. 初始化 Milvus
    logger.info("🔹 初始化 Milvus...")
    try:
        init_milvus()
        logger.info("✅ Milvus 初始化成功")
    except Exception as e:
        _milvus_available = False
        logger.error(f"❌ Milvus 初始化失败: {e}")
        
        # 运行诊断工具
        logger.info("🔍 正在诊断 Milvus 连接问题...")
        diagnosis = diagnose_milvus_connection()
        
        # 输出诊断信息
        logger.info("=" * 60)
        logger.info("Milvus 连接诊断报告")
        logger.info("=" * 60)
        for msg in diagnosis["errors"]:
            logger.info(f"  {msg}")
        if diagnosis["suggestions"]:
            logger.info("💡 建议:")
            for suggestion in diagnosis["suggestions"]:
                logger.info(f"  - {suggestion}")
        logger.info("=" * 60)
        
        logger.warning("⚠️ 系统将降级为直连数据库模式，部分功能（如向量检索、长期记忆）将不可用")

    # 2.1 自动为每个角色灌入默认数据集（仅当该角色知识库为空）
    try:
        if milvus_ready():
            roles = get_roles()
            for r in roles:
                seed_role_knowledge_if_empty(int(r["id"]), r.get("role_name") or "")
            logger.info("✅ 角色数据集自动入库检查完成")
        else:
            logger.info("ℹ️ 当前未连接 Milvus，跳过向量知识库自动入库")
    except Exception as e:
        logger.warning(f"⚠️ 角色数据集自动入库失败（不影响服务启动）: {e}")
    
    # 3. 初始化长期记忆集合
    logger.info("初始化长期记忆集合...")
    try:
        if milvus_ready():
            init_long_term_memory()
            logger.info("✅ 长期记忆集合初始化成功")
        else:
            logger.info("ℹ️ 当前未连接 Milvus，跳过长期记忆初始化")
    except Exception as e:
        logger.warning(f"⚠️ 长期记忆集合初始化失败，将继续提供无长期记忆模式: {e}")
    
    # 4. BM25 模型将在首次查询时自动构建
    logger.info("✅ BM25 模型将在首次查询时自动构建")
    
    logger.info("=== 应用初始化完成 ===")
    yield
    
    # 清理资源
    logger.info("=== 开始关闭应用 ===")
    try:
        if milvus_ready():
            connections.disconnect(_MILVUS_CONNECTION_ALIAS)
            logger.info("✅ Milvus 连接已断开")
    except Exception as e:
        logger.warning(f"⚠️ 断开 Milvus 连接时出错: {e}")
    logger.info("=== 应用已关闭 ===")


app = FastAPI(title="金融分析RAG对话系统", lifespan=lifespan, version="1.0.0")

# ==================== Pydantic 模型 ====================
# API 请求和响应数据模型定义

class FileData(BaseModel):
    """文件数据模型"""
    name: str                              # 文件名
    type: str                              # 文件类型
    data: str                              # Base64编码的文件数据

class ChatRequest(BaseModel):
    """聊天请求模型"""
    user_id: int                           # 用户ID
    role_id: int                           # 角色ID
    message: str                           # 用户消息
    stream: bool = False                   # 是否流式输出
    llm_provider: Optional[str] = None     # 指定的大模型提供商
    file: Optional[FileData] = None        # 附加文件数据

class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str                            # 回答内容
    sources: Optional[List[Dict[str, str]]] = None  # 参考来源

class UserLogin(BaseModel):
    """用户登录请求模型"""
    username: str
    password: str

class UserRegister(BaseModel):
    """用户注册请求模型"""
    username: str
    password: str
    email: Optional[str] = None
    phone: Optional[str] = None

class UserResponse(BaseModel):
    user_id: int
    username: str
    email: Optional[str] = None
    phone: Optional[str] = None

class DocumentUploadRequest(BaseModel):
    source: str
    doc_id: Optional[str] = None

class RoleResponse(BaseModel):
    id: int
    role_name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    personality: Optional[str] = None
    tone: Optional[str] = None
    background_story: Optional[str] = None
    hobbies: Optional[str] = None
    prompt_template: Optional[str] = None

class RoleCreateRequest(BaseModel):
    role_name: str
    display_name: str
    description: Optional[str] = None
    personality: str
    tone: str
    background_story: str
    hobbies: str
    prompt_template: Optional[str] = None
    creator_user_id: Optional[int] = None

class RoleUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    personality: Optional[str] = None
    tone: Optional[str] = None
    background_story: Optional[str] = None
    hobbies: Optional[str] = None
    prompt_template: Optional[str] = None

class KnowledgeQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    role_id: int = 1

class StockSyncResponse(BaseModel):
    status: str
    count: Optional[int] = None

class KnowledgeStatsResponse(BaseModel):
    knowledge_count: int
    stock_count: int
    memory_count: int
    doc_count: int
    bm25_ready: bool

# ========== API 接口 ==========
@app.post("/register", response_model=UserResponse)
async def register(user: UserRegister):
    """用户注册接口"""
    success = register_user(user.username, user.password, user.email, user.phone)
    if not success:
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    # 获取用户信息
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, phone FROM users WHERE username=%s", (user.username,))
        result = cursor.fetchone()
        if result:
            return UserResponse(
                user_id=result[0],
                username=result[1],
                email=result[2],
                phone=result[3]
            )
        raise HTTPException(status_code=500, detail="注册失败")
    finally:
        conn.close()

@app.post("/login", response_model=UserResponse)
async def login(user: UserLogin):
    """用户登录接口"""
    user_id = login_user(user.username, user.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    user_info = get_user_info(user_id)
    if user_info:
        return UserResponse(**user_info)
    raise HTTPException(status_code=500, detail="获取用户信息失败")

def parse_file_content(file_data: FileData) -> str:
    """
    解析文件内容，提取文本信息
    
    支持的文件类型：
    - 文本文件: .txt, .md, .json
    - 图像文件: 使用OCR提取文本
    - PDF文件: 提取文本内容
    - Word文档: 提取文本内容
    """
    import base64
    import io
    
    try:
        # 解析Base64数据
        if file_data.data.startswith('data:'):
            # 移除data URI前缀
            data_uri_parts = file_data.data.split(',')
            if len(data_uri_parts) > 1:
                base64_data = data_uri_parts[1]
            else:
                base64_data = file_data.data
        else:
            base64_data = file_data.data
        
        # 解码Base64
        file_bytes = base64.b64decode(base64_data)
        
        # 根据文件类型处理
        file_name = file_data.name.lower()
        file_type = file_data.type.lower()
        
        # 文本文件
        if file_name.endswith('.txt') or file_name.endswith('.md') or file_name.endswith('.json'):
            return file_bytes.decode('utf-8', errors='ignore')
        
        # PDF文件
        elif file_name.endswith('.pdf'):
            try:
                from PyPDF2 import PdfReader
                pdf_reader = PdfReader(io.BytesIO(file_bytes))
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text if text else "无法提取PDF内容"
            except Exception as e:
                logger.warning(f"PDF解析失败: {e}")
                return f"PDF文件解析失败: {str(e)}"
        
        # Word文档
        elif file_name.endswith('.doc') or file_name.endswith('.docx'):
            try:
                from docx import Document
                doc = Document(io.BytesIO(file_bytes))
                text = "\n".join([para.text for para in doc.paragraphs])
                return text if text else "无法提取Word内容"
            except Exception as e:
                logger.warning(f"Word解析失败: {e}")
                return f"Word文件解析失败: {str(e)}"
        
        # 图像文件（尝试OCR）
        elif 'image' in file_type:
            try:
                from PIL import Image
                
                # 打开图片
                image = Image.open(io.BytesIO(file_bytes))
                logger.info(f"图片尺寸: {image.size}, 模式: {image.mode}")
                
                # 尝试多种OCR方案
                text = None
                ocr_error = None
                
                # 方案1: Tesseract OCR
                try:
                    import pytesseract
                    
                    # 配置Tesseract路径（Windows系统）
                    import os
                    if os.name == 'nt':
                        possible_paths = [
                            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                            r'D:\Program Files\Tesseract-OCR\tesseract.exe',
                        ]
                        for path in possible_paths:
                            if os.path.exists(path):
                                pytesseract.pytesseract.tesseract_cmd = path
                                logger.info(f"Tesseract路径: {path}")
                                break
                    
                    # 预处理图片
                    img_gray = image.convert('L')
                    img_binary = img_gray.point(lambda x: 0 if x < 128 else 255, '1')
                    
                    text = pytesseract.image_to_string(img_binary, lang='chi_sim+eng')
                    logger.info(f"Tesseract OCR识别完成，长度: {len(text)}字符")
                except ImportError:
                    logger.warning("Tesseract未安装，尝试其他OCR方案")
                    ocr_error = "Tesseract未安装"
                except Exception as e:
                    logger.warning(f"Tesseract OCR失败: {e}")
                    ocr_error = str(e)
                
                # 方案2: EasyOCR（备选）
                if not text or not text.strip():
                    try:
                        import easyocr
                        
                        logger.info("尝试使用EasyOCR...")
                        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                        result = reader.readtext(file_bytes)
                        
                        if result:
                            text_parts = []
                            for detection in result:
                                if len(detection) >= 2:
                                    text_parts.append(detection[1])
                            text = '\n'.join(text_parts)
                            logger.info(f"EasyOCR识别完成，长度: {len(text)}字符")
                    except ImportError:
                        logger.warning("EasyOCR未安装")
                    except Exception as e:
                        logger.warning(f"EasyOCR失败: {e}")
                
                # 方案3: PaddleOCR（备选）
                if not text or not text.strip():
                    try:
                        from paddleocr import PaddleOCR
                        
                        logger.info("尝试使用PaddleOCR...")
                        ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
                        result = ocr.ocr(file_bytes, cls=True)
                        
                        if result and result[0]:
                            text_parts = []
                            for line in result[0]:
                                if line and len(line) >= 2:
                                    text_parts.append(line[1][0])
                            text = '\n'.join(text_parts)
                            logger.info(f"PaddleOCR识别完成，长度: {len(text)}字符")
                    except ImportError:
                        logger.warning("PaddleOCR未安装")
                    except Exception as e:
                        logger.warning(f"PaddleOCR失败: {e}")
                
                if text and text.strip():
                    return text
                else:
                    return f"【图片识别结果】\n图片信息: {image.size}, {image.mode}\n\n⚠️ 图片中未识别到文字内容。\n\n可能原因：\n1. 图片质量较低或文字不清晰\n2. 文字被旋转或倾斜\n3. 未安装OCR引擎\n\n📦 安装指南：\n\n【方案一：Tesseract OCR（推荐）】\n1. 下载安装：https://github.com/UB-Mannheim/tesseract/wiki\n2. 安装Python库：pip install pytesseract Pillow\n3. 安装时请勾选中文语言包\n\n【方案二：EasyOCR（纯Python）】\npip install easyocr\n\n【方案三：PaddleOCR（高精度）】\npip install paddlepaddle paddleocr\n\n安装完成后重启后端服务即可使用图片识别功能。"
                    
            except ImportError as e:
                logger.warning(f"图片识别依赖未安装: {e}")
                return "图片识别功能未启用，请安装以下依赖之一：\n1. pip install pytesseract Pillow（需要安装Tesseract OCR引擎）\n2. pip install easyocr\n3. pip install paddlepaddle paddleocr"
            except Exception as e:
                logger.error(f"图片识别异常: {e}")
                return f"图片识别失败: {str(e)}"
        
        # 其他类型
        else:
            return f"不支持的文件类型: {file_name}"
    
    except Exception as e:
        logger.error(f"文件解析异常: {e}")
        return f"文件解析错误: {str(e)}"


@app.post("/api/chat_stream")
async def chat_stream_api(req: ChatRequest):
    """
    流式对话接口（支持文件上传）
    
    处理流程：
    1. 如果有附件，解析文件内容（支持图片OCR、PDF、Word等）
    2. Query 改写（优化查询表达）
    3. 并行执行多源检索（长期记忆、SQLite知识、Milvus混合检索、股票数据）
    4. 构建上下文（整合检索结果）
    5. 流式生成回答（调用大模型）
    """
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    logger.info(f"收到流式聊天请求: user_id={req.user_id}, role_id={req.role_id}, has_file={bool(req.file)}")
    
    # 打印完整请求信息（用于调试）
    try:
        logger.debug(f"请求数据: {json.dumps(req.dict(), default=str, ensure_ascii=False)[:500]}...")
    except Exception as e:
        logger.debug(f"请求数据序列化失败: {e}")
    
    # 处理附件文件
    file_content = ""
    if req.file:
        logger.info(f"收到文件: {req.file.name}, 类型: {req.file.type}, 数据长度: {len(req.file.data) if req.file.data else 0}")
        
        # 检查文件数据是否有效
        if req.file.data and req.file.data.startswith('data:'):
            logger.info("文件数据格式正确（Base64 Data URI）")
        elif req.file.data:
            logger.info("文件数据格式：纯Base64")
        else:
            logger.warning("文件数据为空！")
            
        file_content = parse_file_content(req.file)
        logger.info(f"文件内容提取完成，长度: {len(file_content)}字符")
        if file_content and len(file_content) > 0:
            logger.debug(f"文件内容预览: {file_content[:200]}...")
    
    history = get_chat_history(req.user_id, req.role_id)
    
    # Query改写
    rewritten_query = rewrite_query(req.message, history)
    logger.info(f"原问题: {req.message[:50]}... -> 改写后: {rewritten_query[:50]}...")
    
    # 并行执行多源检索
    start_time = time.time()
    
    def task_long_term_memory():
        try:
            return retrieve_long_term_memory(req.user_id, rewritten_query)
        except Exception as e:
            logger.debug(f"长期记忆检索失败: {e}")
            return []
    
    def task_sqlite_knowledge():
        try:
            return retrieve_sqlite_role_knowledge(req.role_id, rewritten_query)
        except Exception as e:
            logger.debug(f"SQLite知识检索失败: {e}")
            return []
    
    def task_hybrid_search():
        try:
            return hybrid_search(rewritten_query, req.role_id)
        except Exception as e:
            logger.debug(f"Milvus混合检索失败: {e}")
            return []
    
    def task_stock_data():
        try:
            return retrieve_stock_data(rewritten_query)
        except Exception as e:
            logger.debug(f"股票数据检索失败: {e}")
            return []
    
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(task_long_term_memory): "long_term",
            executor.submit(task_sqlite_knowledge): "sqlite",
            executor.submit(task_hybrid_search): "hybrid",
            executor.submit(task_stock_data): "stock",
        }
        
        for future in as_completed(futures, timeout=15):
            task_name = futures[future]
            try:
                results[task_name] = future.result()
            except Exception as e:
                logger.warning(f"检索任务 {task_name} 超时或失败: {e}")
                results[task_name] = []
    
    long_term_memories = results.get("long_term", [])
    sqlite_knowledge = results.get("sqlite", [])
    retrieved_docs = results.get("hybrid", [])
    stock_info = results.get("stock", [])
    
    reranked_docs = rerank_documents(rewritten_query, retrieved_docs)
    
    # 构建上下文（包含文件内容）
    context_parts = []
    if file_content:
        context_parts.append(f"【用户上传文件内容】\n文件名: {req.file.name}\n\n{file_content}")
    if sqlite_knowledge:
        context_parts.append("【角色知识库】\n" + "\n\n".join(sqlite_knowledge))
    if reranked_docs:
        context_parts.append("【金融知识参考】\n" + "\n\n".join([doc["text"] for doc in reranked_docs]))
    if stock_info:
        stock_texts = [f"{s['stock_name']}({s['stock_code']}): 最新价{s['latest_price']}元, 涨跌幅{s['change_pct']}%, 市盈率{s['pe_ratio']}" for s in stock_info]
        context_parts.append("【股票行情数据】\n" + "\n\n".join(stock_texts))
    if long_term_memories:
        context_parts.append("【历史对话记忆】\n" + "\n\n".join(long_term_memories))
    context = "\n\n".join(context_parts) if context_parts else "无相关参考。"
    
    # 流式生成回答
    try:
        response = generate_response(
            req.role_id, req.user_id, req.message, context, history, 
            stream=True, llm_provider=req.llm_provider
        )
        return response
    except Exception as e:
        logger.warning(f"流式生成失败: {e}")
        answer = "很抱歉，我暂时无法处理您的请求。"
        add_to_history(req.user_id, req.role_id, req.message, answer)
        return StreamingResponse(iter([answer]), media_type="text/plain; charset=utf-8")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    核心对话接口
    
    处理流程：
    1. 检查是否为有效问题（否则礼貌追问）
    2. 如果有附件，解析文件内容
    3. Query 改写（优化查询表达）
    4. 并行执行多源检索（长期记忆、SQLite知识、Milvus混合检索、股票数据）
    5. 构建上下文（整合检索结果）
    6. 生成回答（调用大模型）
    7. 存储对话记忆（短期+长期）
    
    参数：
    - user_id: 用户ID
    - role_id: 角色ID
    - message: 用户消息
    - stream: 是否流式输出
    - llm_provider: 指定的大模型提供商
    - file: 附加文件数据（支持图片、文本、PDF、Word等）
    """
    history = get_chat_history(req.user_id, req.role_id)
    
    # 检查缓存（无文件时启用）
    if not req.file:
        cached_answer = get_cached_answer(req.user_id, req.role_id, req.message)
        if cached_answer:
            logger.info(f"缓存命中，直接返回")
            add_to_history(req.user_id, req.role_id, req.message, cached_answer)
            return ChatResponse(answer=cached_answer, sources=[])
    
    # 处理附件文件
    file_content = ""
    if req.file:
        logger.info(f"收到文件: {req.file.name}, 类型: {req.file.type}, 数据长度: {len(req.file.data) if req.file.data else 0}")
        file_content = parse_file_content(req.file)
        logger.info(f"文件内容提取完成，长度: {len(file_content)}字符")
        # 记录部分文件内容用于调试
        if file_content and len(file_content) > 0:
            logger.debug(f"文件内容预览: {file_content[:200]}...")

    # 若用户还没提出明确问题，先礼貌追问，避免模型“脑补乱答”
    if is_probably_not_a_question(req.message):
        answer = polite_clarify_answer(req.message)
        add_to_history(req.user_id, req.role_id, req.message, answer)
        return ChatResponse(answer=answer, sources=[])
    
    # Query 改写（快速操作，先执行）
    rewritten_query = rewrite_query(req.message, history)
    logger.info(f"原始查询: {req.message} -> 改写后: {rewritten_query}")
    
    # 并行执行多源检索，减少总耗时
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    start_time = time.time()
    
    # 定义各个检索任务
    def task_long_term_memory():
        try:
            return retrieve_long_term_memory(req.user_id, req.role_id, rewritten_query)
        except Exception as e:
            logger.debug(f"长期记忆检索失败: {e}")
            return []
    
    def task_sqlite_knowledge():
        try:
            return retrieve_sqlite_role_knowledge(req.role_id, rewritten_query)
        except Exception as e:
            logger.debug(f"SQLite知识检索失败: {e}")
            return []
    
    def task_hybrid_search():
        try:
            return hybrid_search(rewritten_query, role_id=req.role_id)
        except Exception as e:
            logger.debug(f"Milvus混合检索失败: {e}")
            return []
    
    def task_stock_data():
        try:
            return retrieve_stock_data(rewritten_query)
        except Exception as e:
            logger.debug(f"股票数据检索失败: {e}")
            return []
    
    # 并行执行所有检索任务
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(task_long_term_memory): "long_term",
            executor.submit(task_sqlite_knowledge): "sqlite",
            executor.submit(task_hybrid_search): "hybrid",
            executor.submit(task_stock_data): "stock",
        }
        
        for future in as_completed(futures, timeout=15):  # 15秒超时
            task_name = futures[future]
            try:
                results[task_name] = future.result()
            except Exception as e:
                logger.warning(f"检索任务 {task_name} 超时或失败: {e}")
                results[task_name] = []
    
    # 获取结果
    long_term_memories = results.get("long_term", [])
    sqlite_knowledge = results.get("sqlite", [])
    retrieved_docs = results.get("hybrid", [])
    stock_info = results.get("stock", [])
    
    # 重排文档（基于检索结果，必须在检索之后）
    reranked_docs = rerank_documents(rewritten_query, retrieved_docs)
    
    retrieval_time = time.time() - start_time
    logger.info(f"多源检索完成，耗时: {retrieval_time:.2f}秒")
    
    # 构建上下文
    context_parts = []
    # 添加文件内容（放在最前面，优先参考）
    if file_content:
        context_parts.append(f"【用户上传文件内容】\n文件名: {req.file.name}\n\n{file_content}")
    if sqlite_knowledge:
        context_parts.append("【角色知识库】\n" + "\n\n".join(sqlite_knowledge))
    if reranked_docs:
        context_parts.append("【金融知识参考】\n" + "\n\n".join([doc["text"] for doc in reranked_docs]))
    if stock_info:
        stock_texts = [f"{s['stock_name']}({s['stock_code']}): 最新价{s['latest_price']}元, 涨跌幅{s['change_pct']}%, 市盈率{s['pe_ratio']}" for s in stock_info]
        context_parts.append("【股票行情数据】\n" + "\n\n".join(stock_texts))
    if long_term_memories:
        context_parts.append("【历史对话记忆】\n" + "\n\n".join(long_term_memories))
    context = "\n\n".join(context_parts) if context_parts else "无相关参考。"
    
    # 生成回答：优先用数据库直答兜底，避免空泛寒暄
    db_grounded_answer = build_database_grounded_answer(
        req.role_id,
        req.message,
        sqlite_knowledge,
        stock_info=stock_info,
    ) if sqlite_knowledge else ""

    if req.stream:
        try:
            # 使用线程池包装，添加超时控制
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            
            def generate_with_timeout():
                try:
                    return generate_response(
                        req.role_id, req.user_id, req.message, context, history, 
                        stream=True, llm_provider=req.llm_provider
                    )
                except Exception as e:
                    logger.warning(f"流式生成异常: {e}")
                    raise
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(generate_with_timeout)
                try:
                    # 30秒超时，避免长时间阻塞
                    response = future.result(timeout=30)
                    return response
                except TimeoutError:
                    logger.warning("流式生成超时，回退数据库直答")
                    raise Exception("流式生成超时")
        except Exception as e:
            logger.warning(f"流式模型回答失败，回退数据库直答: {e}")
            answer = db_grounded_answer or "我先按当前角色已有知识给你一个直接回答，但你可以再补充更具体一点的问题。"
            add_to_history(req.user_id, req.role_id, req.message, answer)
            try:
                store_long_term_memory(req.user_id, req.role_id, req.message, answer)
            except Exception:
                pass
            return StreamingResponse(iter([answer]), media_type="text/plain; charset=utf-8")
    
    try:
        answer = generate_response(
            req.role_id, req.user_id, req.message, context, history, 
            stream=False, llm_provider=req.llm_provider
        )
    except Exception as e:
        logger.warning(f"模型回答失败，回退数据库直答: {e}")
        answer = db_grounded_answer or "我先按当前角色已有知识给你一个直接回答，但你可以再补充更具体一点的问题。"

    generic_markers = [
        "请问您有什么",
        "有什么想聊的吗",
        "我可以提供一些",
        "很高兴能和你聊天",
        "您好！我是",
        "你好！我是",
    ]
    if db_grounded_answer and any(marker in answer for marker in generic_markers):
        answer = db_grounded_answer
    
    # 存储记忆
    add_to_history(req.user_id, req.role_id, req.message, answer)
    store_long_term_memory(req.user_id, req.role_id, req.message, answer)
    
    sources = []
    if reranked_docs:
        sources.extend([{"source": doc["source"], "summary": doc.get("summary", "")} for doc in reranked_docs])
    if stock_info:
        sources.extend([{"source": "东方财富网", "summary": f"{s['stock_name']}({s['stock_code']})"} for s in stock_info])
    
    # 设置缓存（无文件时启用）
    if not req.file:
        set_cached_answer(req.user_id, req.role_id, req.message, answer)
    
    return ChatResponse(answer=answer, sources=sources)

@app.get("/chat_history/{user_id}/{role_id}")
async def get_history(user_id: int, role_id: int):
    """获取指定用户和角色的聊天历史"""
    history = get_chat_history(user_id, role_id)
    return {"history": history}

@app.post("/clear_history/{user_id}/{role_id}")
async def clear_chat_history(user_id: int, role_id: int):
    """清空指定用户和角色的聊天历史（短期记忆）"""
    clear_history(user_id, role_id)
    return {"status": "cleared"}

@app.get("/roles", response_model=List[RoleResponse])
async def get_all_roles():
    """获取所有可用角色的列表"""
    roles = get_roles()
    return roles

@app.post("/roles", response_model=RoleResponse)
async def create_role(req: RoleCreateRequest):
    """创建自定义角色接口"""
    role_name = req.role_name.strip().lower()
    if not re.match(r"^[a-z0-9_]{2,50}$", role_name):
        raise HTTPException(status_code=400, detail="role_name 仅支持小写字母、数字、下划线，长度 2-50")
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM roles WHERE role_name=%s", (role_name,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="role_name 已存在")
        payload = req.dict() if hasattr(req, "dict") else req.model_dump()
        prompt_template = req.prompt_template or _build_custom_role_prompt(payload)
        cursor.execute(
            """
            INSERT INTO roles
            (role_name, display_name, description, personality, tone, background_story, hobbies, prompt_template, creator_user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                role_name, req.display_name, req.description, req.personality, req.tone,
                req.background_story, req.hobbies, prompt_template, req.creator_user_id
            ),
        )
        conn.commit()
        role_id = cursor.lastrowid
        cursor.execute(
            """
            SELECT id, role_name, display_name, description, personality, tone, background_story, hobbies, prompt_template
            FROM roles WHERE id=%s
            """,
            (role_id,),
        )
        row = cursor.fetchone()
        return RoleResponse(
            id=row[0], role_name=row[1], display_name=row[2], description=row[3], personality=row[4],
            tone=row[5], background_story=row[6], hobbies=row[7], prompt_template=row[8]
        )
    finally:
        conn.close()

@app.put("/roles/{role_id}", response_model=RoleResponse)
async def update_role(role_id: int, req: RoleUpdateRequest):
    """更新角色人设"""
    payload = req.dict(exclude_none=True) if hasattr(req, "dict") else req.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=400, detail="至少提供一个可更新字段")
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, role_name, display_name, description, personality, tone, background_story, hobbies, prompt_template
            FROM roles WHERE id=%s
            """,
            (role_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="角色不存在")
        role = {
            "id": row[0], "role_name": row[1], "display_name": row[2], "description": row[3],
            "personality": row[4], "tone": row[5], "background_story": row[6], "hobbies": row[7], "prompt_template": row[8]
        }
        role.update(payload)
        if "prompt_template" not in payload:
            role["prompt_template"] = _build_custom_role_prompt(role)
        cursor.execute(
            """
            UPDATE roles
            SET display_name=%s, description=%s, personality=%s, tone=%s, background_story=%s, hobbies=%s, prompt_template=%s
            WHERE id=%s
            """,
            (
                role["display_name"], role["description"], role["personality"], role["tone"],
                role["background_story"], role["hobbies"], role["prompt_template"], role_id
            ),
        )
        conn.commit()
        return RoleResponse(
            id=role["id"], role_name=role["role_name"], display_name=role["display_name"],
            description=role["description"], personality=role["personality"], tone=role["tone"],
            background_story=role["background_story"], hobbies=role["hobbies"], prompt_template=role["prompt_template"]
        )
    finally:
        conn.close()

@app.post("/upload_document")
async def upload_document(
    file: UploadFile = File(...),
    source: str = Form(...),
    doc_id: Optional[str] = Form(None),
    role_id: int = Form(1),
):
    """上传文档到知识库"""
    # 检查文件类型
    if not (file.filename.endswith(".txt") or file.filename.endswith(".pdf")):
        raise HTTPException(status_code=400, detail="仅支持 TXT 或 PDF 文件")
    
    # 读取文件内容
    content = await file.read()
    
    try:
        if file.filename.endswith(".pdf"):
            # PDF 文件处理
            doc = fitz.open("pdf", content)
            texts = []
            for page in doc:
                texts.append(page.get_text())
            text = "\n".join(texts)
        else:
            # TXT 文件处理
            text = content.decode("utf-8")
        
        # 分块
        chunks = chunk_text(text)
        
        # 添加到知识库
        add_documents_to_milvus(chunks, source, doc_id, role_id=int(role_id))
        
        return {"status": "success", "chunks_added": len(chunks), "filename": file.filename}
    except Exception as e:
        logger.error(f"上传文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传文档失败: {str(e)}")

@app.post("/query_knowledge")
async def query_knowledge(req: KnowledgeQueryRequest):
    """查询指定角色的知识库（使用混合检索）"""
    if not milvus_ready():
        return {"results": []}
    results = hybrid_search(req.query, role_id=int(req.role_id), top_k=req.top_k)
    return {"results": results}

@app.delete("/knowledge/{doc_id}")
async def delete_knowledge(doc_id: str):
    """删除知识库中指定ID的文档"""
    # 兼容：在所有角色知识库集合中尝试删除同 doc_id（避免找不到具体 role）
    for r in get_roles() or [{"id": 1}]:
        try:
            col = ensure_knowledge_collection(int(r["id"]))
            col.delete(f"doc_id == '{doc_id}'")
            col.flush()
            rebuild_bm25_model_for_collection(col.name)
        except Exception:
            continue
    return {"status": "success", "doc_id": doc_id}

@app.post("/sync_stock_data", response_model=StockSyncResponse)
async def sync_stock_data():
    """从东方财富同步股票数据到Milvus"""
    if not milvus_ready():
        return {"status": "skipped", "count": len(_load_local_stock_cache())}
    stock_list = fetch_stock_data_from_eastmoney()
    if stock_list:
        store_stock_data_to_milvus(stock_list)
        return {"status": "success", "count": len(stock_list)}
    return {"status": "failed", "count": 0}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "financial-rag-system"}

@app.get("/knowledge_stats", response_model=KnowledgeStatsResponse)
async def get_knowledge_stats():
    """获取知识库统计"""
    try:
        # 汇总所有角色知识库集合的数量
        knowledge_count = 0
        stock_count = len(_load_local_stock_cache())
        memory_count = 0
        if milvus_ready():
            for r in get_roles() or [{"id": 1}]:
                try:
                    knowledge_count += ensure_knowledge_collection(int(r["id"])).num_entities
                except Exception:
                    continue
            stock_collection = Collection(STOCK_COLLECTION_NAME)
            stock_count = stock_collection.num_entities
            memory_collection = Collection("long_term_memory")
            memory_count = memory_collection.num_entities
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge_docs")
        doc_count = cursor.fetchone()[0]
        conn.close()
        
        bm25_ready = milvus_ready() and any(_bm25_models.values())
        
        return KnowledgeStatsResponse(
            knowledge_count=knowledge_count,
            stock_count=stock_count,
            memory_count=memory_count,
            doc_count=doc_count,
            bm25_ready=bm25_ready
        )
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 请求缓存辅助函数 ====================
def get_cache_key(user_id: int, role_id: int, message: str) -> str:
    """生成缓存键"""
    return hashlib.md5(f"{user_id}:{role_id}:{message}".encode()).hexdigest()

def get_cached_answer(user_id: int, role_id: int, message: str) -> Optional[str]:
    """获取缓存的回答"""
    key = get_cache_key(user_id, role_id, message)
    cached = _request_cache.get(key)
    if cached and time.time() - cached["timestamp"] < _CACHE_TTL:
        logger.debug(f"命中缓存: {key[:8]}")
        return cached["answer"]
    return None

def set_cached_answer(user_id: int, role_id: int, message: str, answer: str):
    """设置缓存的回答"""
    # 清理过期缓存
    current_time = time.time()
    keys_to_remove = [k for k, v in _request_cache.items() if current_time - v["timestamp"] >= _CACHE_TTL]
    for k in keys_to_remove:
        del _request_cache[k]
    
    # 如果缓存已满，删除最旧的
    if len(_request_cache) >= _CACHE_MAX_SIZE:
        oldest_key = min(_request_cache.keys(), key=lambda k: _request_cache[k]["timestamp"])
        del _request_cache[oldest_key]
    
    key = get_cache_key(user_id, role_id, message)
    _request_cache[key] = {"answer": answer, "timestamp": current_time}
    logger.debug(f"缓存已设置: {key[:8]}")

# ========== 启动应用 ==========
if __name__ == "__main__":
    import uvicorn
    server_host = os.getenv("SERVER_HOST", "127.0.0.1")
    server_port = resolve_server_port(default_port=8001, host=server_host)
    logger.info(f"启动服务: {server_host}:{server_port}")
    uvicorn.run(app, host=server_host, port=server_port)
