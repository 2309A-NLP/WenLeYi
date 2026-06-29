# -*- coding: utf-8 -*-
"""
金融对话系统 - 服务入口（FastAPI应用）
这是整个RAG后端的启动文件，负责：
1. 初始化FastAPI应用
2. 加载知识库到内存
3. 初始化数据库和向量模型
4. 注册API路由
5. 启动HTTP服务器
"""

import logging
import os
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# 加载环境变量（从.env文件读取API Key、数据库密码等）
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# 配置日志格式：时间 [级别] 模块名: 消息
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# 自定义日志过滤器：屏蔽不重要的日志，让终端更清爽
class QuietFilter(logging.Filter):
    """过滤掉milvus连接失败、DB查询失败等已知的无害警告"""
    def filter(self, record):
        msg = record.getMessage()
        if 'milvus' in msg.lower() or 'pymilvus' in msg.lower():
            return False  # 过滤milvus相关日志
        if 'DB查询失败' in msg or '数据库不可用' in msg:
            return False  # 过滤数据库不可用日志
        if '向量搜索失败' in msg:
            return False  # 过滤向量搜索失败日志
        return True

# 应用过滤器到根日志记录器
quiet_filter = QuietFilter()
logging.getLogger().addFilter(quiet_filter)

# 降低第三方库的日志级别，减少干扰
logging.getLogger("jieba").setLevel(logging.WARNING)
logging.getLogger("pymilvus").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="pymilvus")

SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))  # 默认端口8000

# ==================== 前端页面加载 ====================
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")

def _get_html():
    """读取前端HTML文件内容"""
    try:
        with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<html><body><h1>页面加载失败</h1><p>{e}</p></body></html>"


# ==================== 应用生命周期管理 ====================
# 在应用启动时执行初始化，关闭时执行清理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI生命周期：启动时加载资源，关闭时释放"""
    logger.info("=" * 50)
    logger.info("金融对话系统 正在启动...")
    logger.info("=" * 50)

    # 1. 加载知识库（5447条金融问答对）到内存
    from knowledge_base import load_knowledge
    kb = load_knowledge()
    logger.info(f"知识库已加载: {kb['total']} 条问答, {len(kb['categories'])} 个分类")

    # 2. 初始化MySQL数据库（如果可用）
    try:
        from database import init_database
        init_database()
        logger.info("MySQL 数据库初始化成功")
    except Exception as e:
        logger.warning(f"数据库不可用（无状态模式）: {e}")

    # 3. 预加载嵌入模型（用于向量搜索）
    try:
        from milvus_search import _load_model
        _load_model()
        logger.info("Milvus嵌入模型已预加载")
    except Exception as e:
        logger.warning(f"Milvus嵌入模型加载失败（将降级到jieba搜索）: {e}")

    # 4. 注册API路由（所有/api/v1/开头的接口）
    from api import router
    app.include_router(router, prefix="/api/v1")
    logger.info("API 路由已注册")
    logger.info(f"前端页面: http://127.0.0.1:{SERVER_PORT}")
    logger.info("=" * 50)

    yield  # 应用运行中...

    logger.info("金融对话系统已停止")


# ==================== 创建FastAPI应用 ====================
app = FastAPI(
    title="金融对话系统",
    description="基于RAG的金融领域智能问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS中间件：允许前端跨域访问后端API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],       # 允许所有HTTP方法
    allow_headers=["*"],       # 允许所有请求头
)


@app.get("/")
def index():
    """根路径：返回前端页面"""
    return HTMLResponse(_get_html())


# ==================== 启动入口 ====================
if __name__ == "__main__":
    # host="0.0.0.0" 表示允许外部访问（AutoDL需要）
    uvicorn.run("main:app", host="0.0.0.0", port=SERVER_PORT, reload=False)
