# -*- coding: utf-8 -*-
"""
工具模块：配置读取、LLM调用、向量模型加载、知识库搜索
这是整个系统的"工具箱"，提供各种基础功能
"""

import os
import logging
import time
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 加载环境变量（从.env文件读取API Key等配置）
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(_env_path)

# ==================== LLM（大语言模型）配置 ====================
# 支持DeepSeek和通义千问两种大模型

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")  # 默认用DeepSeek

# DeepSeek配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 通义千问配置（备用）
QIANWEN_API_KEY = os.getenv("QIANWEN_API_KEY", "")
QIANWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QIANWEN_MODEL = "qwen-plus"

# ==================== 向量模型配置 ====================
BGE_M3_PATH = os.getenv("BGE_M3_PATH", "")         # BGE嵌入模型路径
BGE_RERANKER_PATH = os.getenv("BGE_RERANKER_PATH", "")  # BGE重排序模型路径
EMBEDDING_DIM = 768  # m3e-base模型输出的向量维度

# ==================== MySQL 配置 ====================
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "financial_chat"),
    "charset": "utf8mb4",
}

# ==================== 服务参数 ====================
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))  # RAG服务端口
TOP_K_RETRIEVE = 5        # 检索返回的最相关条目数
MAX_CONTEXT_LENGTH = 4000  # 拼接给大模型的最大上下文长度


# ==================== LLM 客户端 ====================
_llm_client = None  # 单例模式，全局只创建一个客户端

def get_llm_client():
    """
    获取LLM客户端（支持DeepSeek/千问）
    采用单例模式：第一次调用时创建，后续直接复用
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    _llm_client = {
        "http_client": requests.Session(),  # 用requests.Session复用TCP连接
        "provider": LLM_PROVIDER,
    }

    if LLM_PROVIDER == "deepseek":
        _llm_client["api_key"] = DEEPSEEK_API_KEY
        _llm_client["base_url"] = DEEPSEEK_BASE_URL
        _llm_client["model"] = DEEPSEEK_MODEL
        logger.info(f"LLM: DeepSeek ({DEEPSEEK_MODEL})")
    elif LLM_PROVIDER == "qianwen":
        _llm_client["api_key"] = QIANWEN_API_KEY
        _llm_client["base_url"] = QIANWEN_BASE_URL
        _llm_client["model"] = QIANWEN_MODEL
        logger.info(f"LLM: 通义千问 ({QIANWEN_MODEL})")
    else:
        raise ValueError(f"不支持的LLM提供商: {LLM_PROVIDER}")

    return _llm_client


def call_llm(messages, stream=False, temperature=0.7, max_tokens=2048, max_retries=3, retry_delay=1):
    """
    【核心函数】调用大模型生成回答
    参数：
        messages: 对话消息列表，格式为 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        temperature: 温度参数，控制回答的随机性（0.7比较均衡）
        max_retries: 最大重试次数（网络不稳定时自动重试）
    返回：大模型生成的文本回答
    """
    client = get_llm_client()
    # 构造HTTP请求头（带API密钥认证）
    headers = {
        "Authorization": f"Bearer {client['api_key']}",
        "Content-Type": "application/json"
    }
    # 构造请求体（OpenAI兼容格式）
    payload = {
        "model": client["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client["http_client"].post(
                f"{client['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,  # 120秒超时
            )
            resp.raise_for_status()
            if stream:
                return resp  # 流式模式返回原始响应对象
            data = resp.json()
            # 从OpenAI兼容的响应格式中提取回答文本
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            last_error = f"请求超时（第{attempt+1}次尝试）"
            logger.warning(f"LLM调用超时，正在重试 ({attempt+1}/{max_retries})...")
        except requests.exceptions.RequestException as e:
            last_error = f"网络请求错误: {str(e)}"
            logger.warning(f"LLM调用失败，正在重试 ({attempt+1}/{max_retries}): {e}")
        except KeyError as e:
            last_error = f"响应格式错误: {str(e)}"
            logger.warning(f"LLM响应格式错误: {e}")
        except Exception as e:
            last_error = f"未知错误: {str(e)}"
            logger.error(f"LLM调用异常: {e}")

        # 递增延迟重试（1秒、2秒、3秒...）
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1))

    raise Exception(f"LLM调用失败（已重试{max_retries}次）: {last_error}")


# ==================== 向量模型（惰性加载） ====================
# 惰性加载：只在第一次使用时才加载模型，节省启动时间
_embedding_model = None
_reranker_model = None

def get_embedding_model():
    """加载BGE嵌入模型（把文本转成向量）"""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    if not BGE_M3_PATH or not os.path.exists(BGE_M3_PATH):
        logger.warning(f"嵌入模型路径不存在: {BGE_M3_PATH}，将使用简易嵌入")
        return None
    from sentence_transformers import SentenceTransformer
    _embedding_model = SentenceTransformer(BGE_M3_PATH)
    logger.info(f"嵌入模型已加载: {BGE_M3_PATH}")
    return _embedding_model


def get_reranker_model():
    """加载BGE重排序模型（对检索结果重新排序，提高准确率）"""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    if not BGE_RERANKER_PATH or not os.path.exists(BGE_RERANKER_PATH):
        logger.warning(f"重排序模型路径不存在: {BGE_RERANKER_PATH}")
        return None
    from sentence_transformers import CrossEncoder
    _reranker_model = CrossEncoder(BGE_RERANKER_PATH)
    logger.info(f"重排序模型已加载: {BGE_RERANKER_PATH}")
    return _reranker_model


def embed_text(text):
    """将单条文本向量化（调用嵌入模型）"""
    model = get_embedding_model()
    if model is None:
        return None
    return model.encode(text).tolist()


# ==================== 金融知识库搜索（jieba分词匹配） ====================
def search_knowledge_base(query, knowledge_items, top_k=5):
    """
    【核心函数】在本地知识库中检索最相关的问答对
    算法：jieba分词 + Jaccard相似度
    原理：
    1. 用jieba把用户问题和知识库每个问答都切成词
    2. 计算"词的交集"占"词的并集"的比例（Jaccard系数）
    3. 问题匹配权重0.7，答案匹配权重0.3（问题更重要）
    4. 按得分排序，返回top_k条
    """
    import jieba

    if not knowledge_items:
        return []

    # 用jieba搜索模式分词（会把"股票基金"切成"股票"、"基金"、"股票基金"）
    query_words = set(jieba.lcut_for_search(query))

    scored = []
    for item in knowledge_items:
        # 对知识库的每个问答对都分词
        q_words = set(jieba.lcut_for_search(item["q"]))  # 问题分词
        a_words = set(jieba.lcut_for_search(item["a"]))  # 答案分词

        # Jaccard相似度 = 交集大小 / 并集大小
        q_score = len(query_words & q_words) / max(len(query_words | q_words), 1)
        a_score = len(query_words & a_words) / max(len(query_words | a_words), 1)

        # 加权总分：问题匹配占70%，答案匹配占30%
        total_score = q_score * 0.7 + a_score * 0.3
        scored.append((total_score, item))

    # 按得分从高到低排序
    scored.sort(key=lambda x: x[0], reverse=True)
    # 返回得分大于0的top_k条
    return [item for score, item in scored[:top_k] if score > 0]
