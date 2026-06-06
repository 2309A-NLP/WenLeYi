"""RAG 系统配置中心 -- 从 .env 文件读取所有参数，提供默认值。

功能说明:
    1. 集中管理 RAG 系统全部可配置项
    2. 支持 .env 文件和环境变量两种配置方式
    3. 提供 log_config() 方法在启动时打印所有配置值（带 [CONFIG] 前缀）
"""

# ========== 标准库导入 ==========
import os
from pathlib import Path
from dotenv import load_dotenv

# ========== 加载 .env 文件 ==========
# 从项目根目录下读取 .env，覆盖系统环境变量中未设置的项
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


class Config:
    """集中管理 RAG 系统所有参数。

    所有配置均通过 .env 文件或环境变量设置，
    未设置时使用类属性中的默认值。
    """

    # ========== 基础路径 ==========
    # 项目根目录，所有其他路径均相对于此目录
    BASE_DIR = Path(__file__).parent.parent.resolve()

    # ========== 目录配置 ==========
    # 文档存储目录：用户上传的原始文档存放位置
    DOCS_DIR = str(BASE_DIR / "documents")
    # 向量库存储目录：FAISS 索引等持久化数据
    VECTOR_STORE_DIR = str(BASE_DIR / "vector_store")
    # 上传临时目录：文件上传过程中的临时存放
    UPLOAD_DIR = str(BASE_DIR / "uploads")

    # ========== 分块参数 ==========
    # 每个文本块的最大字符数，过大会影响检索精度
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
    # 相邻块之间的重叠字符数，保证语义连续性
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    # ========== 检索参数 ==========
    # 检索模式：hybrid（混合）/ vector（纯向量）/ bm25（关键词）
    SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid")
    # 最终返回给用户的 Top K 个结果
    TOP_K = int(os.getenv("TOP_K", "5"))
    # BM25 检索返回的候选数量
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))
    # FAISS 向量检索返回的候选数量
    FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "10"))
    # RRF 融合算法的 k 参数，值越大排名越平滑
    RRF_K = int(os.getenv("RRF_K", "60"))
    # 相似度阈值，低于此值的结果会被过滤掉
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    # ========== Reranker（重排序） ==========
    # 是否启用 Reranker 对检索结果进行二次排序
    ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "1") == "1"
    # 本地 Reranker 模型路径（为空则使用 API 模式）
    RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "")
    # Reranker 返回的 Top K 结果数
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

    # ========== Reranker API（云端模式） ==========
    # 当本地模型路径为空时，使用 API 调用远程 Reranker
    RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "")
    RERANKER_API_BASE = os.getenv("RERANKER_API_BASE", "https://api.siliconflow.cn/v1")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    # ========== Embedding（向量化模型） ==========
    # 本地 Embedding 模型路径，用于将文本转为向量
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    MILVUS_INSERT_BATCH_SIZE = int(os.getenv("MILVUS_INSERT_BATCH_SIZE", "200"))
    DOC_PROCESS_WORKERS = int(os.getenv("DOC_PROCESS_WORKERS", "0"))
    RETRIEVER_WORKERS = int(os.getenv("RETRIEVER_WORKERS", "3"))

    # ========== LLM（大语言模型） ==========
    # API 密钥（sk- 开头），调用大模型时的鉴权凭证
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    # API 基础 URL，指向大模型服务端点
    LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    # 模型名称，如 mimo-v2.5 / gpt-3.5-turbo 等
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    # API 调用超时时间（秒）
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

    # ========== 视觉模型（图片解析） ==========
    # 视觉模型 API 密钥，用于解析文档中的图片内容
    VISION_API_KEY = os.getenv("VISION_API_KEY", "")
    # 视觉模型 API 基础 URL
    VISION_API_BASE = os.getenv("VISION_API_BASE", "")
    # 视觉模型名称
    VISION_MODEL = os.getenv("VISION_MODEL", "")

    # ========== Milvus（向量数据库） ==========
    # Milvus 服务地址
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    # Milvus 服务端口
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    # Milvus 集合（表）名称
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks")
    # 向量存储后端选择：faiss（本地）/ milvus（分布式）
    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")

    # ========== 加权融合权重 ==========
    # 混合检索中各路结果的权重比例，总和不一定要为 1
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
    VEC_WEIGHT = float(os.getenv("VEC_WEIGHT", "0.6"))
    GRAPH_WEIGHT = float(os.getenv("GRAPH_WEIGHT", "0.3"))

    # ========== 多轮对话 ==========
    # 是否启用多轮对话记忆
    ENABLE_MULTI_TURN = os.getenv("ENABLE_MULTI_TURN", "0") == "1"
    # 多轮对话保留的历史轮次上限
    MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))

    # ========== Flask Web 服务 ==========
    # 监听地址和端口
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    # 调试模式开关
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

    # ========== HTTPS / SSL ==========
    # 是否启用 HTTPS 加密
    ENABLE_SSL = os.getenv("ENABLE_SSL", "0") == "1"
    # SSL 证书和私钥文件路径
    SSL_CERTFILE = os.getenv("SSL_CERTFILE", "")
    SSL_KEYFILE = os.getenv("SSL_KEYFILE", "")

    # ========== 历史问题检索 ==========
    # 历史问答记录目录
    QA_HISTORY_DIR = os.getenv("QA_HISTORY_DIR", "")
    # 是否启用 QA 缓存（关闭可避免相似问题返回旧答案）
    ENABLE_QUESTION_CACHE = os.getenv("ENABLE_QUESTION_CACHE", "1") == "1"
    # 历史问题相似度阈值
    QA_SIMILARITY_THRESHOLD = float(os.getenv("QA_SIMILARITY_THRESHOLD", "0.85"))

    # ========== Neo4j 图谱数据库 ==========
    # 是否启用 Neo4j 知识图谱
    ENABLE_NEO4J = os.getenv("ENABLE_NEO4J", "0") == "1"
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_TOP_K = int(os.getenv("NEO4J_TOP_K", "5"))

    # ========== Redis 短期记忆 ==========
    # 是否启用 Redis 缓存（短期对话记忆）
    ENABLE_REDIS = os.getenv("ENABLE_REDIS", "0") == "1"
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_MEMORY_TTL_SECONDS = int(os.getenv("REDIS_MEMORY_TTL_SECONDS", "604800"))

    # ========== MySQL 持久化存储 ==========
    # 是否启用 MySQL 持久化
    ENABLE_MYSQL = os.getenv("ENABLE_MYSQL", "0") == "1"
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_db")

    # ========== JWT 鉴权 ==========
    # 是否启用 JWT 登录鉴权
    ENABLE_JWT = os.getenv("ENABLE_JWT", "0") == "1"
    # JWT 签名密钥
    JWT_SECRET = os.getenv("JWT_SECRET", "default_secret")
    # JWT 令牌有效期（小时）
    JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
    JWT_USERNAME = os.getenv("JWT_USERNAME", "admin")
    JWT_PASSWORD = os.getenv("JWT_PASSWORD", "")

    # ========== 内容过滤 / 屏蔽词 ==========
    # 是否启用敏感词过滤
    ENABLE_FILTER = os.getenv("ENABLE_FILTER", "0") == "1"
    # 屏蔽词列表（逗号分隔）
    FILTER_WORDS = os.getenv("FILTER_WORDS", "").split(",") if os.getenv("FILTER_WORDS") else []
    FILTER_RESPONSE = os.getenv("FILTER_RESPONSE", "问题包含屏蔽词，已拦截。")

    # ========== 固定响应文案 ==========
    # 未检索到相关信息时的回复
    NO_INFO_RESPONSE = os.getenv("NO_INFO_RESPONSE", "未找到相关信息。")
    # 问题与文档无关时的回复
    IRRELEVANT_RESPONSE = os.getenv("IRRELEVANT_RESPONSE", "抱歉，我只能回答与当前文档相关的问题。")

    # ------------------------------------------------------------------ #
    #                       辅助方法                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mask_key(key: str) -> str:
        """遮蔽 API 密钥：只显示前 6 位和后 4 位，中间用星号替代。

        示例: sk-cj2...j6c6 -> sk-cj2***j6c6
        """
        if not key:
            return "(empty)"
        if len(key) <= 10:
            return key[:3] + "***" + key[-2:]
        return key[:6] + "***" + key[-4:]

    # ------------------------------------------------------------------ #
    #                    启动日志打印方法                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def log_config(cls) -> None:
        """在系统启动时打印所有配置项，方便排查问题。

        格式: [CONFIG] 配置项名称 = 配置值
        注意: API Key 类字段会自动遮蔽，仅显示首尾部分字符。
        """
        print("=" * 60)
        print("[CONFIG] RAG 系统配置启动日志")
        print("=" * 60)

        # ---------- 目录配置 ----------
        print("[CONFIG] --- 目录配置 ---")
        print(f"[CONFIG] BASE_DIR          = {cls.BASE_DIR}")
        print(f"[CONFIG] DOCS_DIR          = {cls.DOCS_DIR}")
        print(f"[CONFIG] VECTOR_STORE_DIR  = {cls.VECTOR_STORE_DIR}")
        print(f"[CONFIG] UPLOAD_DIR        = {cls.UPLOAD_DIR}")

        # ---------- 分块参数 ----------
        print("[CONFIG] --- 分块参数 ---")
        print(f"[CONFIG] CHUNK_SIZE        = {cls.CHUNK_SIZE}")
        print(f"[CONFIG] CHUNK_OVERLAP     = {cls.CHUNK_OVERLAP}")

        # ---------- 检索参数 ----------
        print("[CONFIG] --- 检索参数 ---")
        print(f"[CONFIG] SEARCH_MODE       = {cls.SEARCH_MODE}")
        print(f"[CONFIG] TOP_K             = {cls.TOP_K}")
        print(f"[CONFIG] BM25_TOP_K        = {cls.BM25_TOP_K}")
        print(f"[CONFIG] FAISS_TOP_K       = {cls.FAISS_TOP_K}")
        print(f"[CONFIG] RRF_K             = {cls.RRF_K}")
        print(f"[CONFIG] SIMILARITY_THRESHOLD = {cls.SIMILARITY_THRESHOLD}")

        # ---------- 加权融合权重 ----------
        print("[CONFIG] --- 加权融合权重 ---")
        print(f"[CONFIG] BM25_WEIGHT       = {cls.BM25_WEIGHT}")
        print(f"[CONFIG] VEC_WEIGHT        = {cls.VEC_WEIGHT}")
        print(f"[CONFIG] GRAPH_WEIGHT      = {cls.GRAPH_WEIGHT}")

        # ---------- Reranker 配置 ----------
        print("[CONFIG] --- Reranker 配置 ---")
        print(f"[CONFIG] ENABLE_RERANKER   = {cls.ENABLE_RERANKER}")
        print(f"[CONFIG] RERANKER_MODEL_PATH = {cls.RERANKER_MODEL_PATH or '(empty)'}")
        print(f"[CONFIG] RERANK_TOP_K      = {cls.RERANK_TOP_K}")
        print(f"[CONFIG] RERANKER_API_KEY  = {cls._mask_key(cls.RERANKER_API_KEY)}")
        print(f"[CONFIG] RERANKER_API_BASE = {cls.RERANKER_API_BASE}")
        print(f"[CONFIG] RERANKER_MODEL    = {cls.RERANKER_MODEL}")

        # ---------- Embedding 模型 ----------
        print("[CONFIG] --- Embedding 模型 ---")
        print(f"[CONFIG] EMBEDDING_MODEL_PATH = {cls.EMBEDDING_MODEL_PATH or '(empty)'}")
        print(f"[CONFIG] EMBEDDING_BATCH_SIZE = {cls.EMBEDDING_BATCH_SIZE}")
        print(f"[CONFIG] MILVUS_INSERT_BATCH_SIZE = {cls.MILVUS_INSERT_BATCH_SIZE}")
        print(f"[CONFIG] DOC_PROCESS_WORKERS = {cls.DOC_PROCESS_WORKERS}")
        print(f"[CONFIG] RETRIEVER_WORKERS = {cls.RETRIEVER_WORKERS}")

        # ---------- LLM 大模型 ----------
        print("[CONFIG] --- LLM 大模型 ---")
        print(f"[CONFIG] LLM_API_KEY       = {cls._mask_key(cls.LLM_API_KEY)}")
        print(f"[CONFIG] LLM_API_BASE      = {cls.LLM_API_BASE}")
        print(f"[CONFIG] LLM_MODEL         = {cls.LLM_MODEL}")
        print(f"[CONFIG] LLM_TIMEOUT       = {cls.LLM_TIMEOUT}s")

        # ---------- 视觉模型 ----------
        print("[CONFIG] --- 视觉模型（图片解析） ---")
        print(f"[CONFIG] VISION_API_KEY    = {cls._mask_key(cls.VISION_API_KEY)}")
        print(f"[CONFIG] VISION_API_BASE   = {cls.VISION_API_BASE or '(empty)'}")
        print(f"[CONFIG] VISION_MODEL      = {cls.VISION_MODEL or '(empty)'}")

        # ---------- Milvus 向量库 ----------
        print("[CONFIG] --- Milvus / 向量库 ---")
        print(f"[CONFIG] VECTOR_BACKEND    = {cls.VECTOR_BACKEND}")
        print(f"[CONFIG] MILVUS_HOST       = {cls.MILVUS_HOST}")
        print(f"[CONFIG] MILVUS_PORT       = {cls.MILVUS_PORT}")
        print(f"[CONFIG] MILVUS_COLLECTION = {cls.MILVUS_COLLECTION}")

        # ---------- 多轮对话 ----------
        print("[CONFIG] --- 多轮对话 ---")
        print(f"[CONFIG] ENABLE_MULTI_TURN = {cls.ENABLE_MULTI_TURN}")
        print(f"[CONFIG] MAX_HISTORY_TURNS = {cls.MAX_HISTORY_TURNS}")

        # ---------- Flask 服务 ----------
        print("[CONFIG] --- Flask 服务 ---")
        print(f"[CONFIG] HOST              = {cls.HOST}")
        print(f"[CONFIG] PORT              = {cls.PORT}")
        print(f"[CONFIG] DEBUG             = {cls.DEBUG}")

        # ---------- HTTPS / SSL ----------
        print("[CONFIG] --- HTTPS / SSL ---")
        print(f"[CONFIG] ENABLE_SSL        = {cls.ENABLE_SSL}")
        print(f"[CONFIG] SSL_CERTFILE      = {cls.SSL_CERTFILE or '(empty)'}")
        print(f"[CONFIG] SSL_KEYFILE       = {cls.SSL_KEYFILE or '(empty)'}")

        # ---------- 历史问题检索 ----------
        print("[CONFIG] --- 历史问题检索 ---")
        print(f"[CONFIG] QA_HISTORY_DIR    = {cls.QA_HISTORY_DIR or '(empty)'}")
        print(f"[CONFIG] ENABLE_QUESTION_CACHE = {cls.ENABLE_QUESTION_CACHE}")
        print(f"[CONFIG] QA_SIMILARITY_THRESHOLD = {cls.QA_SIMILARITY_THRESHOLD}")

        # ---------- Neo4j 图谱 ----------
        print("[CONFIG] --- Neo4j 图谱 ---")
        print(f"[CONFIG] ENABLE_NEO4J      = {cls.ENABLE_NEO4J}")
        print(f"[CONFIG] NEO4J_URI         = {cls.NEO4J_URI}")
        print(f"[CONFIG] NEO4J_USER        = {cls.NEO4J_USER}")
        print(f"[CONFIG] NEO4J_PASSWORD    = {cls._mask_key(cls.NEO4J_PASSWORD)}")
        print(f"[CONFIG] NEO4J_TOP_K       = {cls.NEO4J_TOP_K}")

        # ---------- Redis ----------
        print("[CONFIG] --- Redis 短期记忆 ---")
        print(f"[CONFIG] ENABLE_REDIS      = {cls.ENABLE_REDIS}")
        print(f"[CONFIG] REDIS_HOST        = {cls.REDIS_HOST}")
        print(f"[CONFIG] REDIS_PORT        = {cls.REDIS_PORT}")
        print(f"[CONFIG] REDIS_DB          = {cls.REDIS_DB}")
        print(f"[CONFIG] REDIS_MEMORY_TTL_SECONDS = {cls.REDIS_MEMORY_TTL_SECONDS}")

        # ---------- MySQL ----------
        print("[CONFIG] --- MySQL 持久化 ---")
        print(f"[CONFIG] ENABLE_MYSQL      = {cls.ENABLE_MYSQL}")
        print(f"[CONFIG] MYSQL_HOST        = {cls.MYSQL_HOST}")
        print(f"[CONFIG] MYSQL_PORT        = {cls.MYSQL_PORT}")
        print(f"[CONFIG] MYSQL_USER        = {cls.MYSQL_USER}")
        print(f"[CONFIG] MYSQL_PASSWORD    = {cls._mask_key(cls.MYSQL_PASSWORD)}")
        print(f"[CONFIG] MYSQL_DATABASE    = {cls.MYSQL_DATABASE}")

        # ---------- JWT 鉴权 ----------
        print("[CONFIG] --- JWT 鉴权 ---")
        print(f"[CONFIG] ENABLE_JWT        = {cls.ENABLE_JWT}")
        print(f"[CONFIG] JWT_SECRET        = {cls._mask_key(cls.JWT_SECRET)}")
        print(f"[CONFIG] JWT_EXPIRE_HOURS  = {cls.JWT_EXPIRE_HOURS}")
        print(f"[CONFIG] JWT_USERNAME      = {cls.JWT_USERNAME}")
        print(f"[CONFIG] JWT_PASSWORD      = {cls._mask_key(cls.JWT_PASSWORD)}")

        # ---------- 屏蔽词过滤 ----------
        print("[CONFIG] --- 屏蔽词过滤 ---")
        print(f"[CONFIG] ENABLE_FILTER     = {cls.ENABLE_FILTER}")
        print(f"[CONFIG] FILTER_WORDS      = {cls.FILTER_WORDS if cls.FILTER_WORDS else '(none)'}")
        print(f"[CONFIG] FILTER_RESPONSE   = {cls.FILTER_RESPONSE}")

        # ---------- 固定回复文案 ----------
        print("[CONFIG] --- 固定回复文案 ---")
        print(f"[CONFIG] NO_INFO_RESPONSE  = {cls.NO_INFO_RESPONSE}")
        print(f"[CONFIG] IRRELEVANT_RESPONSE = {cls.IRRELEVANT_RESPONSE}")

        print("=" * 60)
        print("[CONFIG] 配置日志打印完毕")
        print("=" * 60)
