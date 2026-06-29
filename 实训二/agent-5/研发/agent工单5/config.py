# 工单编号：人工智能NLP-Agent数字人项目-招股书数据问答智能体任务
"""
配置文件 - 招股书数据问答智能体
所有可配置的参数集中在此文件管理
"""
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ==================== 路径配置 ====================
# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 招股书TXT文件目录（已解析好的文本文件）
TXT_DIR = os.path.join(BASE_DIR, "pdf_txt_file")

# 招股书PDF文件目录
PDF_DIR = os.path.join(BASE_DIR, "pdf")

# 问题文件路径（JSONL格式，每行一个JSON对象，已筛选只保留招股书问题）
QUESTION_FILE = os.path.join(BASE_DIR, "question_prospectus.json")

# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 索引数据目录（存放分块数据和Milvus相关信息）
DATA_DIR = os.path.join(BASE_DIR, "data")

# ==================== 模型配置 ====================
# m3e-base 向量化模型路径（中文embedding模型）
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", r"D:\桌面\模型\m3e-base")

# bge-reranker-base 重排序模型路径（用于检索结果重排序）
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH", r"D:\桌面\模型\bge-reranker-base")

# ==================== Milvus 向量数据库配置 ====================
MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", 19530))
# Milvus集合名称（用于存储招股书文本向量）
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "prospectus_chunks")

# DeepSeek API 配置（使用小米MiMo API）
DEEPSEEK_API_KEY = "sk-cvny3mtg8mbwivhwiv62ogaxofl1utpmimpg5wvs2zcgaczl"
DEEPSEEK_BASE_URL = "https://api.xiaomimimo.com/v1"
DEEPSEEK_MODEL = "mimo-v2.5"

# ==================== 检索配置 ====================
# 文本分块大小（字符数）
CHUNK_SIZE = 2048
# 分块重叠大小（字符数），保证上下文连贯
CHUNK_OVERLAP = 500
# 向量检索返回的候选数量
SEARCH_TOP_K = 10
# 重排序后保留的最终数量（送入LLM的文本块数）
FINAL_TOP_K = 5

# ==================== LLM 生成配置 ====================
# 温度参数，越低越确定性
TEMPERATURE = 0.1
# 最大生成token数
MAX_TOKENS = 2048
# API请求超时时间（秒）- 增大到120秒以应对长prompt和LLM慢响应
REQUEST_TIMEOUT = 120
