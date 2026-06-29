# 工单编号：人工智能NLP-Agent数字人项目-基金问答智能体任务
"""
配置文件 - 集中管理所有配置参数
"""
import os

# === 数据库配置 ===
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '博金杯比赛数据.db')

# === API配置 ===
# DeepSeek API
DEEPSEEK_API_KEY = 'sk-3464124a416d4807b956c5482bbe772d'
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
DEEPSEEK_MODEL = 'deepseek-chat'

# 小米API
XIAOMI_API_KEY = 'tp-cmvr1fh3pywarr7ywg3wxxj8hzivbqkwgj3wicy9zwnynqg7'
XIAOMI_BASE_URL = 'https://token-plan-cn.xiaomimimo.com/v1'
XIAOMI_MODEL = 'mimo-v2.5'

# === Milvus配置 ===
MILVUS_HOST = 'localhost'
MILVUS_PORT = 19530
MILVUS_COLLECTION = 'fund_fewshot'

# === Embedding配置 ===
EMBEDDING_MODEL_PATH = r'D:\桌面\模型\m3e-base'

# === SQL执行配置 ===
SQL_TIMEOUT = 60  # SQL执行超时秒数
MAX_RETRY = 2    # SQL错误时最大重试次数

# === 输出配置 ===
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
QUESTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'question.json')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'question_with_answer.jsonl')
ERROR_LOG = os.path.join(OUTPUT_DIR, 'error_log.jsonl')

# === Few-shot配置 ===
FEW_SHOT_COUNT = 5
