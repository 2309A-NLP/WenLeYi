# -*- coding: utf-8 -*-
"""
工具函数模块：通用工具函数、配置常量、向量模型管理、角色提示词模板
职责：
- 提供通用工具函数（时间格式化、数据校验、加密解密、格式转换等）
- 管理向量模型（BGE-M3、BGE-Reranker）
- 定义角色提示词模板和相关配置
- 请求缓存管理
"""

import os
import logging
import hashlib
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 请求缓存（提升性能）====================
_request_cache: Dict[str, dict] = {}
_CACHE_TTL = 300
_CACHE_MAX_SIZE = 100

# ==================== 加载环境变量 ====================
from dotenv import load_dotenv
load_dotenv()

# ==================== 配置常量 ====================

# 大模型配置
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
DOUBAO_SECRET_KEY = os.getenv("DOUBAO_SECRET_KEY")
QIANWEN_API_KEY = os.getenv("QIANWEN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 本地模型路径配置
LOCAL_BGE_M3_PATH = os.getenv("BGE_M3_PATH", r"D:\模型\bge_m3_model")
LOCAL_BGE_RERANKER_PATH = os.getenv("BGE_RERANKER_PATH", r"D:\模型\bge-reranker-base")

# MySQL 数据库配置
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "roleplay"),
    "charset": "utf8mb4",
    "autocommit": False,
}

# Redis 配置
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0)),
    "decode_responses": True
}

# Milvus 向量数据库配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.18.128")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_USE_SSL = os.getenv("MILVUS_USE_SSL", "false").lower() == "true"
MILVUS_TIMEOUT = int(os.getenv("MILVUS_TIMEOUT", 30))
MILVUS_ENABLE_LOCAL = os.getenv("MILVUS_ENABLE_LOCAL", "false").lower() == "true"

# RAG 检索参数配置（优化版）
COLLECTION_PREFIX = "financial_knowledge_role"
EMBEDDING_DIM = 1024
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVE = 5
TOP_K_RERANK = 2
MEMORY_LENGTH = 3
TOP_K_LONG_TERM = 2
TOP_K_STOCK = 1
MAX_CONTEXT_LENGTH = 1500

# ==================== 角色提示词模板定义 ====================
ROLE_PROMPT_TEMPLATES = {
    "financial_advisor": """
你是一位专业的金融理财师，精通各类金融产品和投资策略。

要求：
1. 分析用户的问题，提供专业、客观的金融建议
2. 结合提供的参考信息进行回答，简洁明了
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
        ("孤独感", "感到孤独时，可以尝试主动联系朋友、参加兴趣小组，培养一个爱好，或者做一些帮助他人的事情来获得连接感。"),
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
# ==================== 角色与数据集映射配置 ====================
def _get_role_data_mapping() -> Dict[str, List[str]]:
    """获取角色与数据集文件的映射关系"""
    return {
        # 医生角色 - 医疗数据
        "doctor": [
            "medical dataset/medical.json",
            "Flash_distill ls/健康饮食指导_merged.json",
            "Flash_distill ls/日常保健知识_merged.json",
        ],
        
        # 心理医生角色 - 心理咨询数据
        "psychologist": [
            "PsyDTCorpus xlys/PsyDTCorpus_test_single_turn_split.json",
            "PsyDTCorpus xlys/PsyDTCorpus_train_mulit_turn_packing.json",
            "Flash_distill ls/心理健康维护_merged.json",
        ],
        
        # 教师角色 - 学习资料数据
        "teacher": [
            "Flash_distill ls/Python编程基础_merged.json",
            "Flash_distill ls/线性代数_merged.json",
            "Flash_distill ls/微积分入门_merged.json",
            "Flash_distill ls/概率统计_merged.json",
            "Flash_distill ls/机器学习入门_merged.json",
            "Flash_distill ls/数学建模_merged.json",
            "Flash_distill ls/数论基础_merged.json",
            "Flash_distill ls/几何基础_merged.json",
            "Flash_distill ls/三角函数_merged.json",
            "Flash_distill ls/学习方法指导_merged.json",
            "Flash_distill ls/常见错误排查_merged.json",
            "Flash_distill ls/数学竞赛_merged.json",
        ],
        
        # 律师角色 - 法律数据
        "lawyer": [
            "legal dataset  fl/corpus.jsonl",
            "legal dataset  fl/queries.json",
        ],
        
        # 科学家角色 - 科学知识数据
        "scientist": [
            "Flash_distill ls/技术概念解释_merged.json",
            "Flash_distill ls/科学知识问答_merged.json",
            "Flash_distill ls/数据分析_merged.json",
        ],
        
        # 英语学习助手角色 - 学习资料数据
        "english_tutor": [
            "Flash_distill ls/学习方法指导_merged.json",
        ],
        
        # 股票分析师角色 - 股票数据
        "stock_analyst": [
            "stockdata gp/stock_data/stock_list.csv",
            "stockdata gp/stock_data/hs_index_list_nev.csv",
            "stockdata gp/stock_data/hs_index_realtime.csv",
            "stockdata gp/stock_data/index_industry_concept_tree.csv",
            "stockdata gp/stock_data/index_related_to_stock.csv",
            "stockdata gp/stock_data/new_stock_calendar.csv",
            "stockdata gp/stock_data/real_time_trading_data_with.csv",
            "stockdata gp/stock_data/stock_related_to_index.csv",
            "eastmoney_stock_list.json",
        ],
        
        # 投资顾问角色 - 金融问答数据
        "investment_advisor": [
            "financial_advisor_qa.csv",
        ],
        
        # 财务规划师角色 - 理财规划数据
        "financial_planner": [
            "financial_advisor_qa.csv",
            "Flash_distill ls/理财规划建议_merged.json",
        ],
        
        # 虚拟朋友角色 - 通用对话数据
        "virtual_friend": [
            "Flash_distill ls/日常问题解决_merged.json",
            "Flash_distill ls/社交礼仪指导_merged.json",
            "Flash_distill ls/烹饪技巧分享_merged.json",
            "Flash_distill ls/家庭收纳整理_merged.json",
            "Flash_distill ls/时间管理技巧_merged.json",
            "Flash_distill ls/节能环保生活_merged.json",
            "Flash_distill ls/安全常识普及_merged.json",
        ],
        
        # 程序员/开发者角色 - 代码优化数据
        "programmer": [
            "Flash_distill ls/代码优化建议_merged.json",
            "Flash_distill ls/项目开发指导_merged.json",
            "Flash_distill ls/工具使用技巧_merged.json",
        ],
        
        # 职业规划师角色
        "career_advisor": [
            "Flash_distill ls/职业规划_merged.json",
        ],
        
        # 地理/生活助手 - 城市信息数据
        "geography_assistant": [
            "Flash_distill ls/上海_merged.json",
            "Flash_distill ls/北京_merged.json",
            "Flash_distill ls/浙江_merged.json",
            "Flash_distill ls/江苏_merged.json",
            "Flash_distill ls/安徽_merged.json",
            "Flash_distill ls/福建_merged.json",
            "Flash_distill ls/河南_merged.json",
            "Flash_distill ls/江西_merged.json",
            "Flash_distill ls/山西_merged.json",
            "Flash_distill ls/辽宁_merged.json",
            "Flash_distill ls/吉林_merged.json",
            "Flash_distill ls/黑龙江_merged.json",
            "Flash_distill ls/重庆_merged.json",
            "Flash_distill ls/天津_merged.json",
        ],
    }


# ==================== 全局变量和缓存 ====================
# 稠密向量嵌入模型（BGE-m3）的单例实例，用于将文本转换为向量
_embedding_model = None
# 重排序模型（BGE-reranker）的单例实例，用于对检索结果进行精细化排序
_reranker_model = None
# BM25 稀疏检索模型的缓存字典，键为集合名称，值为 BM25EmbeddingFunction 实例
_bm25_models: Dict[str, Any] = {}
# Redis 客户端实例，用于分布式缓存（可选，失败时降级为内存缓存）
_redis_client = None
# 内存中的短期对话历史缓存，键格式为 f"{user_id}:{role_id}"，值为消息列表
_in_memory_history: Dict[str, List[Dict[str, str]]] = {}
# Milvus 向量数据库是否可用的标志
_milvus_available = False
# 本地缓存的股票数据列表，用于 Milvus 不可用时的降级展示
_local_stock_cache: Optional[List[Dict[str, Any]]] = None
# Milvus 连接别名，用于标识一个特定的 Milvus 连接
_MILVUS_CONNECTION_ALIAS = "default"
# Milvus 中股票数据集合的名称常量
STOCK_COLLECTION_NAME = "stock_data"

# LLM客户端缓存，键为大模型提供商（如 "deepseek"），值为 OpenAI 客户端实例
_llm_clients = {}
# 豆包（百度文心）的 Token 信息缓存，包含 token 字符串和过期时间戳
_doubao_token_info = {"token": None, "expire_time": 0}


# ==================== 工具函数 ====================

def hash_password(password: str) -> str:
    """密码哈希"""
    # 使用 SHA256 对密码进行哈希，返回十六进制字符串（注：生产环境建议加盐）
    return hashlib.sha256(password.encode()).hexdigest()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """文本分块"""
    chunks = []  # 存储分块后的文本片段
    start = 0  # 当前分块的起始索引
    text_length = len(text)  # 文本总长度

    while start < text_length:
        # 计算当前分块的结束索引（不超过文本末尾）
        end = min(start + chunk_size, text_length)

        # 如果还没有到达末尾，尝试在分块边界附近寻找合适的断句符号（句号、感叹号、问号等）
        if end < text_length:
            for sep in ['。', '！', '？', '.', '!', '?', '\n']:
                # 在 [start, end) 范围内从右向左查找分隔符
                idx = text.rfind(sep, start, end)
                # 如果找到且位置超过 start + overlap，则在该分隔符后断开
                if idx != -1 and idx > start + chunk_overlap:
                    end = idx + 1  # 包含分隔符
                    break

        # 提取当前块并去除首尾空白
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # 下一个块的开始位置 = 当前结束位置 - 重叠长度，实现块之间的重叠
        start = end - chunk_overlap

    return chunks


def simple_chinese_tokenizer(text: str) -> list:
    """简单的中文分词器（避免依赖NLTK）"""
    # 将非中文、数字、字母、空格的字符替换为空格（去除标点符号）
    text = re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z\s]', ' ', text)

    tokens = []
    parts = text.split()  # 按空白字符分割

    for part in parts:
        i = 0
        while i < len(part):
            # 如果是中文字符，逐个字符作为一个 token
            if '\u4e00' <= part[i] <= '\u9fff':
                tokens.append(part[i])
                i += 1
            # 如果是字母或数字，连续取出一整个单词（字母数字串）
            elif part[i].isalnum():
                j = i
                while j < len(part) and part[j].isalnum():
                    j += 1
                tokens.append(part[i:j].lower())  # 转为小写
                i = j
            else:
                i += 1

    # 过滤掉长度为1的纯字母 token（例如单个字母）
    tokens = [t for t in tokens if not (len(t) == 1 and t.isalpha())]
    return tokens


def normalize_answer_text(text: str) -> str:
    """清洗模型回答，去掉 JSON 包装、markdown 符号和多余引号"""
    if text is None:
        return ""
    cleaned = str(text).strip()

    # 尝试解析 JSON 格式的回答，从中提取真正的文本内容（如 {"answer": "..."}）
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

    # 使用正则去除 JSON 样式的包裹：例如开头 {"response": " 和结尾 "}
    cleaned = re.sub(r'^\s*\{\s*"?(response|answer|content|text)"?\s*:\s*"?', "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'"\s*\}\s*$', "", cleaned)
    # 反转义内部的双引号
    cleaned = cleaned.replace('\\"', '"')
    # 移除 Markdown 粗体标记和标题标记
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("#", "")
    cleaned = cleaned.replace("*", "")
    # 最终去除开头结尾的空白、引号、换行、制表符
    return cleaned.strip(' "\n\t')


def _data_dir() -> str:
    """获取数据目录路径"""
    # 返回项目根目录下的 "data" 文件夹的绝对路径
    # 当前文件在 last_duan/utils.py，所以需要向上一级目录
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load_local_stock_cache() -> List[Dict[str, Any]]:
    """加载本地股票数据缓存"""
    global _local_stock_cache
    if _local_stock_cache is not None:
        return _local_stock_cache  # 已加载，直接返回

    # 定义两个可能的缓存文件路径，按优先级尝试
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
            # 如果文件内容是列表，直接缓存并返回
            if isinstance(data, list):
                _local_stock_cache = data
                return _local_stock_cache
            # 如果是字典，尝试将其值提取为列表（比如多级嵌套）
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

    # 未找到任何有效缓存，初始化为空列表
    _local_stock_cache = []
    return _local_stock_cache


def milvus_ready() -> bool:
    """检查 Milvus 是否已连接可用"""
    return _milvus_available


def set_milvus_available(available: bool):
    """设置 Milvus 连接状态"""
    global _milvus_available
    _milvus_available = available


def get_bm25_models() -> Dict[str, Any]:
    """获取 BM25 模型缓存"""
    return _bm25_models


def set_bm25_model(collection_name: str, model: Any):
    """设置 BM25 模型"""
    _bm25_models[collection_name] = model


def get_cache_key(user_id: int, role_id: int, message: str) -> str:
    """生成缓存键"""
    # 将用户ID、角色ID和消息内容拼接后计算 MD5，作为缓存键
    return hashlib.md5(f"{user_id}:{role_id}:{message}".encode()).hexdigest()


def get_cached_answer(user_id: int, role_id: int, message: str) -> Optional[str]:
    """获取缓存的回答"""
    key = get_cache_key(user_id, role_id, message)
    cached = _request_cache.get(key)  # _request_cache 是全局请求缓存字典（在本片段外定义）
    # 如果缓存存在且未过期（TTL 未超时），则返回缓存的回答
    if cached and time.time() - cached["timestamp"] < _CACHE_TTL:
        logger.debug(f"命中缓存: {key[:8]}")
        return cached["answer"]
    return None


def set_cached_answer(user_id: int, role_id: int, message: str, answer: str):
    """设置缓存的回答"""
    current_time = time.time()
    # 清理所有过期的缓存条目
    keys_to_remove = [k for k, v in _request_cache.items() if current_time - v["timestamp"] >= _CACHE_TTL]
    for k in keys_to_remove:
        del _request_cache[k]

    # 如果缓存条目数量超过最大容量，删除最老的条目
    if len(_request_cache) >= _CACHE_MAX_SIZE:
        oldest_key = min(_request_cache.keys(), key=lambda k: _request_cache[k]["timestamp"])
        del _request_cache[oldest_key]

    # 存储新缓存
    key = get_cache_key(user_id, role_id, message)
    _request_cache[key] = {"answer": answer, "timestamp": current_time}
    logger.debug(f"缓存已设置: {key[:8]}")


def _build_custom_role_prompt(role: Dict[str, Any]) -> str:
    """根据角色信息构建自定义提示词"""
    # 获取角色的显示名称，如果没有则使用内部名称，再没有就默认"角色"
    role_name = role.get("display_name") or role.get("role_name") or "角色"
    # 获取性格描述，默认为"自然、友好、专业"
    personality = (role.get("personality") or "自然、友好、专业").strip()
    # 获取语气风格，默认为"口语化、真诚"
    tone = (role.get("tone") or "口语化、真诚").strip()
    # 获取背景故事，默认为"无"
    background_story = (role.get("background_story") or "无").strip()
    # 获取爱好，默认为"无"
    hobbies = (role.get("hobbies") or "无").strip()
    # 构造提示词模板，包含角色设定和回答约束
    return f"""
你现在扮演"{role_name}"。

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


def rewrite_query(query: str, history: List[Dict[str, str]]) -> str:
    """Query 改写（结合历史对话，使用 LLM 进行智能改写）"""
    if not history:
        return query
    
    try:
        client = get_llm_client()
        model_name = get_llm_model_name()
        
        history_str = "\n".join([f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-5:]])
        
        prompt = f"""
        请将用户的当前问题结合历史对话进行改写，使其成为一个独立完整的问题。
        改写时需要：
        1. 将代词（如"它"、"这个"、"那个"）替换为具体指代对象
        2. 补充必要的上下文信息
        3. 保持问题的原意不变
        4. 输出改写后的问题即可，不要添加额外解释
        
        历史对话：
        {history_str}
        
        当前问题：{query}
        
        改写后的问题：
        """.strip()
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        
        rewritten = response.choices[0].message.content.strip()
        logger.info(f"Query改写: 原问题='{query[:30]}...' -> 改写后='{rewritten[:30]}...'")
        return rewritten if rewritten else query
        
    except Exception as e:
        logger.error(f"Query改写失败: {e}")
        return query


def is_probably_not_a_question(text: str) -> bool:
    """粗略判断用户是否尚未提出问题"""
    t = (text or "").strip()
    if not t:  # 空字符串视为未提出问题
        return True
    if "?" in t or "？" in t:  # 包含问号，认为是问题
        return False

    if len(t) >= 40:  # 长文本，可能是问题描述
        return False

    # 问候语的正则匹配
    greetings = r"^(你好|您好|嗨|hi|hello|在吗|在不在)\b"
    # 自我介绍的正则匹配（如“我叫张三”）
    intro = r"(我叫|我是|叫我|昵称是|名字是)\s*[\u4e00-\u9fffA-Za-z0-9·]{1,12}$"
    if re.search(greetings, t, flags=re.IGNORECASE):
        return True
    if re.search(intro, t):
        return True

    # 常见纯问候短句
    if t in {"你好", "您好", "嗨", "hi", "hello", "在吗", "在不在"}:
        return True

    # 包含这些动作词，更倾向于认为是问题
    action_words = {"回答", "帮忙", "请问", "请教", "想知道", "了解", "分析", "建议", "推荐", "查询"}
    for word in action_words:
        if word in t:
            return False

    return False


def polite_clarify_answer(text: str) -> str:
    """礼貌地追问用户具体问题"""
    # 尝试从文本中提取用户自称的名字
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
    # 未提取到名字时的通用追问
    return (
        "你好！我在。\n\n"
        "你想咨询哪个具体问题？你把问题直接发我就行（比如资产配置、指标解读、行业/公司客观信息梳理等）。"
    )


def _find_role_dataset_files(role_name: str) -> List[str]:
    """查找与角色名相关的数据集文件"""
    base = _data_dir()  # 数据目录
    role_lower = (role_name or "").lower()  # 角色名小写
    candidates: List[str] = []
    
    # 检查角色与数据集的映射
    role_data_mapping = _get_role_data_mapping()
    if role_name in role_data_mapping:
        mapped_files = role_data_mapping[role_name]
        for rel_path in mapped_files:
            full_path = os.path.join(base, rel_path)
            if os.path.exists(full_path):
                candidates.append(full_path)
        if candidates:
            return candidates
    
    try:
        # 遍历数据目录下的所有文件
        for fn in os.listdir(base):
            lower = fn.lower()
            # 只保留 .csv、.txt 或 .json 文件
            if not (lower.endswith(".csv") or lower.endswith(".txt") or lower.endswith(".json")):
                continue
            # 如果文件名包含角色名（不区分大小写），则加入候选列表
            if role_lower and role_lower in lower:
                candidates.append(os.path.join(base, fn))
    except Exception:
        candidates = []

    if candidates:
        return sorted(candidates)  # 返回排序后的候选文件列表

    # 如果没有匹配到角色专属文件，则尝试使用通用的 financial_qa.csv
    fallback = os.path.join(base, "financial_qa.csv")
    if os.path.exists(fallback):
        return [fallback]
    return []


def _load_texts_from_dataset(path: str) -> List[str]:
    """从 csv/txt/json 数据集中抽取可入库文本"""
    lower = path.lower()
    
    # 处理 TXT 文件：直接读取全部文本并分块
    if lower.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
        return chunk_text(txt) if txt else []

    # 处理 CSV 文件：期望有 question 和 ground_truth/answer 字段
    if lower.endswith(".csv"):
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
    
    # 处理 JSON 文件：支持多种格式
    if lower.endswith(".json"):
        return _load_json_dataset(path)
    
    return []


def _load_json_dataset(path: str) -> List[str]:
    """加载 JSON 格式的数据集"""
    try:
        import json
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        
        texts = []
        
        if isinstance(data, list):
            for item in data:
                # 格式1: {"question": "...", "answer": "..."} 或 {"question": "...", "ground_truth": "..."}
                if isinstance(item, dict):
                    q = (item.get("question") or item.get("q") or "").strip()
                    a = (item.get("answer") or item.get("ground_truth") or item.get("a") or "").strip()
                    if q and a:
                        texts.append(f"Q: {q}\nA: {a}")
                    elif q:
                        texts.append(q)
                    elif a:
                        texts.append(a)
                    
                    # 格式2: 医疗数据格式 {"disease": "...", "symptoms": "...", "drugs": "..."}
                    disease = item.get("disease")
                    symptoms = item.get("symptoms")
                    drugs = item.get("drugs")
                    department = item.get("department")
                    if disease:
                        parts = [f"疾病: {disease}"]
                        if symptoms:
                            parts.append(f"症状: {symptoms}")
                        if department:
                            parts.append(f"科室: {department}")
                        if drugs:
                            parts.append(f"常用药物: {drugs}")
                        texts.append("\n".join(parts))
                    
                    # 格式3: 对话格式 {"conversation": [...]}
                    conversation = item.get("conversation")
                    if isinstance(conversation, list):
                        dialog_text = ""
                        for msg in conversation:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role and content:
                                role_display = "用户" if role.lower() == "user" else "助手"
                                dialog_text += f"{role_display}: {content}\n"
                        if dialog_text:
                            texts.append(dialog_text.strip())
        
        elif isinstance(data, dict):
            # 处理单个对象
            conversation = data.get("conversation")
            if isinstance(conversation, list):
                dialog_text = ""
                for msg in conversation:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        role_display = "用户" if role.lower() == "user" else "助手"
                        dialog_text += f"{role_display}: {content}\n"
                if dialog_text:
                    texts.append(dialog_text.strip())
        
        return texts
    except Exception as e:
        logger.warning(f"加载 JSON 文件失败: {path}: {e}")
        return []


# ==================== 模型获取函数 ====================

def get_embedding_model() -> Any:
    """获取向量化模型（单例）"""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"加载 BGE-m3 模型: {LOCAL_BGE_M3_PATH}")
        try:
            from sentence_transformers import SentenceTransformer
            # 从本地路径加载 BGE-m3 模型
            _embedding_model = SentenceTransformer(LOCAL_BGE_M3_PATH)
            logger.info("✅ BGE-m3 模型加载成功")
        except Exception as e:
            logger.error(f"❌ BGE-m3 模型加载失败: {e}")
            raise  # 模型必须可用，否则抛出异常
    return _embedding_model


def get_reranker_model() -> Any:
    """获取重排序模型（单例）"""
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"加载 BGE-reranker 模型: {LOCAL_BGE_RERANKER_PATH}")
        try:
            from FlagEmbedding import FlagReranker
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"  # 优先使用 GPU
            logger.info(f"使用设备: {device}")
            try:
                # 尝试使用指定设备加载模型
                _reranker_model = FlagReranker(LOCAL_BGE_RERANKER_PATH, use_fp16=False, device=device)
            except Exception as e:
                logger.debug(f"标准加载失败，尝试备用方式: {e}")
                # 备用：强制使用 CPU 加载
                _reranker_model = FlagReranker(LOCAL_BGE_RERANKER_PATH, use_fp16=False, device="cpu")
            logger.info("✅ BGE-reranker 模型加载成功")
        except Exception as e:
            logger.warning(f"⚠️ BGE-reranker 模型加载失败: {e}")
            _reranker_model = None  # 置为 None，后续会降级
    return _reranker_model


def get_redis_client() -> Any:
    """获取 Redis 客户端（单例）"""
    global _redis_client
    if _redis_client is None:
        logger.info(f"连接 Redis: {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}")
        try:
            import redis
            _redis_client = redis.Redis(**REDIS_CONFIG)  # 根据配置创建连接
            _redis_client.ping()  # 测试连通性
            logger.info("✅ Redis 连接成功")
        except Exception as e:
            logger.warning(f"⚠️ Redis 连接失败: {e}")
            _redis_client = None  # 失败时设为 None，后续使用内存缓存
    return _redis_client


# ==================== 大模型调用函数 ====================

def get_llm_client(provider: str = None):
    """获取大模型客户端（带缓存）"""
    provider = provider or LLM_PROVIDER  # 如果未指定，使用全局默认提供商

    # 如果已缓存该提供商的客户端，直接返回
    if provider in _llm_clients:
        return _llm_clients[provider]

    import time

    if provider == "deepseek":
        from openai import OpenAI
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    elif provider == "doubao":
        # 豆包（百度文心）需要先获取 access_token
        now = time.time()
        # 检查 token 是否不存在或已过期
        if not _doubao_token_info["token"] or now >= _doubao_token_info["expire_time"]:
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
                # 设置过期时间为 29 天后（预留缓冲）
                _doubao_token_info["expire_time"] = now + (29 * 24 * 3600)
                logger.info("✅ 豆包Token获取成功")
            except Exception as e:
                logger.error(f"❌ 豆包Token获取失败: {e}")
                raise

        from openai import OpenAI
        client = OpenAI(api_key=_doubao_token_info["token"],
                        base_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro")
    elif provider == "qianwen":
        from openai import OpenAI
        client = OpenAI(api_key=QIANWEN_API_KEY, base_url="https://dashscope.io/api/compatible-mode/v1")
    elif provider == "chatgpt":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")
    else:
        raise ValueError(f"不支持的大模型提供商: {provider}")

    # 缓存客户端实例
    _llm_clients[provider] = client
    return client


def get_llm_model_name(provider: str = None) -> str:
    """获取大模型名称"""
    provider = provider or LLM_PROVIDER
    model_mapping = {
        "deepseek": "deepseek-chat",
        "doubao": "ernie-4.0-8k",
        "qianwen": "qwen-plus",
        "chatgpt": "gpt-4o-mini",
    }
    return model_mapping.get(provider, "deepseek-chat")