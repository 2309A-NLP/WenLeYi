# -*- coding: utf-8 -*-
"""
项目入口文件：负责创建应用实例、注册路由/蓝图、配置中间件、启动服务
职责：
- 创建 FastAPI 应用实例
- 配置中间件（CORS、日志等）
- 初始化数据库连接
- 注册 API 路由
- 启动服务
"""

# 导入标准库模块
import logging  # 日志记录模块
import os  # 操作系统接口，用于读取环境变量和路径操作
import sys  # 系统相关参数和函数，用于修改模块搜索路径

# 导入第三方库
from dotenv import load_dotenv  # 从 .env 文件加载环境变量
from fastapi import FastAPI  # FastAPI 框架核心类
from fastapi.middleware.cors import CORSMiddleware  # 跨域资源共享中间件
from fastapi.middleware.gzip import GZipMiddleware  # GZip 压缩中间件

# 处理相对导入问题，支持直接运行 python main.py
# 判断当前脚本是否作为主程序运行（而不是被导入）
if __name__ == "__main__":
    # 获取当前文件（main.py）所在的目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取当前目录的父目录（即项目根目录）
    parent_dir = os.path.dirname(current_dir)
    # 如果父目录不在模块搜索路径中，则将其插入到最前面，以便导入项目内的其他模块（如 database, utils, api）
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # 构建环境变量文件的绝对路径：父目录下的 peizhi_huanjing/.env
    env_path = os.path.join(parent_dir, 'peizhi_huanjing', '.env')
    # 加载该 .env 文件中的环境变量
    load_dotenv(env_path)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# ==================== 配置常量 ====================
# 从环境变量获取服务器监听端口，默认为 8000
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
# 从环境变量获取调试模式标志，默认为 False（字符串 "true" 不区分大小写转为布尔值）
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"


# ==================== 初始化函数 ====================
def init_services():
    """初始化所有服务"""
    logger.info("🚀 开始初始化服务...")

    # 初始化 MySQL 数据库（用于存储用户、角色、对话历史、角色知识等）
    logger.info("📊 初始化 MySQL 数据库...")
    try:
        import database  # 动态导入 database 模块（因为之前已修改 sys.path）
        database.init_database()  # 调用 database 模块中的数据库初始化函数（创建表等）
        logger.info("✅ MySQL 数据库初始化成功")
    except Exception as e:
        logger.error(f"❌ MySQL 数据库初始化失败: {e}")
        raise  # 数据库初始化失败是致命错误，抛出异常终止启动

    # 初始化 Milvus 向量数据库（可选，失败不影响整体启动，因为可以降级到本地缓存）
    logger.info("📚 初始化 Milvus 向量数据库...")
    try:
        import database
        database.init_milvus()  # 初始化 Milvus 连接并创建所需集合
        logger.info("✅ Milvus 向量数据库初始化成功")
    except Exception as e:
        #  Milvus 连接失败仅记录警告，不阻断服务启动（将使用本地缓存或降级检索）
        logger.warning(f"⚠️ Milvus 向量数据库初始化失败（将使用本地缓存模式）: {e}")
        logger.info("🔍 诊断 Milvus 连接问题...")
        try:
            import database
            # 调用诊断函数，获取连接失败的可能原因和建议
            diagnosis = database.diagnose_milvus_connection()
            for error in diagnosis["errors"]:
                logger.info(f"   {error}")
            for suggestion in diagnosis["suggestions"]:
                logger.info(f"   💡 {suggestion}")
        except Exception as de:
            logger.warning(f"⚠️ 诊断失败: {de}")

    # 初始化角色知识库（如果 Milvus 可用）
    try:
        import database
        # 检查 Milvus 是否就绪（刚刚尝试初始化后，可能全局标志已设）
        if database.milvus_ready():
            logger.info("🔄 初始化角色知识库...")
            try:
                # 获取所有角色列表
                roles = database.get_roles()
                # 对每个角色，如果其知识库集合为空，则自动从 data/ 目录导入种子数据（如金融问答对）
                for role in roles:
                    database.seed_role_knowledge_if_empty(int(role["id"]), role["role_name"])
                logger.info("✅ 角色知识库初始化完成")
            except Exception as e:
                logger.warning(f"⚠️ 角色知识库初始化失败: {e}")
    except Exception as e:
        logger.warning(f"⚠️ 检查 Milvus 状态失败: {e}")

    # 加载向量化模型（BGE-m3），用于将文本转换为稠密向量。提前加载以预热，避免首次请求延迟
    logger.info("🧠 加载向量化模型...")
    try:
        import utils
        utils.get_embedding_model()  # 该函数会加载模型并缓存，可能耗时
        logger.info("✅ BGE-m3 向量化模型加载成功")
    except Exception as e:
        logger.error(f"❌ 向量化模型加载失败: {e}")  # 向量化模型必须可用，否则系统无法工作，但这里只记录错误不抛出？原代码未 raise，维持原样

    # 加载重排序模型（BGE-reranker），用于对检索结果进行精细化排序（可选）
    try:
        import utils
        utils.get_reranker_model()
        logger.info("✅ BGE-reranker 重排序模型加载成功")
    except Exception as e:
        # 重排序模型不是必须的，加载失败仅警告
        logger.warning(f"⚠️ 重排序模型加载失败: {e}")

    # 加载 Redis 客户端（用于缓存 LLM 回答和检索结果，提升性能）
    logger.info("🔌 加载 Redis 客户端...")
    try:
        import utils
        redis_client = utils.get_redis_client()  # 尝试连接 Redis
        if redis_client:
            logger.info("✅ Redis 客户端加载成功")
        else:
            logger.warning("⚠️ Redis 不可用，将使用内存缓存")
    except Exception as e:
        logger.warning(f"⚠️ Redis 客户端加载失败: {e}")

    logger.info("🎉 服务初始化完成")


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    logger.info("🏗️ 创建 FastAPI 应用...")

    # 实例化 FastAPI 应用，配置标题、描述、版本、调试模式
    app = FastAPI(
        title="金融领域 RAG 智能对话系统",
        description="基于 Retrieval-Augmented Generation 的金融领域智能对话系统，支持多角色、多模态输入和长期记忆",
        version="1.0.0",
        debug=DEBUG_MODE,
    )

    # 配置 CORS 中间件：允许跨域请求（开发环境通常允许所有来源，生产环境应限制）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有源
        allow_credentials=True,  # 允许携带凭证（Cookie、Authorization 头）
        allow_methods=["*"],  # 允许所有 HTTP 方法
        allow_headers=["*"],  # 允许所有请求头
    )

    # 配置 GZip 压缩中间件：对大于 1000 字节的响应体进行 GZip 压缩，减少网络传输
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 注册 API 路由（将所有的端点定义绑定到 app 上）
    logger.info("📡 注册 API 路由...")
    import api
    api.register_api_routes(app)  # 调用 api 模块中的路由注册函数，传入 app 实例

    logger.info("✅ FastAPI 应用创建完成")
    return app


# ==================== 启动服务 ====================
app = None  # 全局变量，用于存储应用实例（在模块级别先声明）


def main():
    """主函数：初始化服务并启动应用"""
    global app  # 声明修改全局变量 app

    try:
        # 1. 初始化所有服务（数据库、模型、缓存等）
        init_services()
        # 2. 创建 FastAPI 应用并注册路由
        app = create_app()

        logger.info(f"🚀 服务启动准备完成，端口: {SERVER_PORT}")

        # 3. 使用 Uvicorn 启动 ASGI 服务器
        import uvicorn
        import logging
        
        # 创建自定义日志过滤器，过滤掉 socket.io 请求
        class SocketIOFilter(logging.Filter):
            def filter(self, record):
                # 检查日志记录中是否包含 socket.io
                message = record.getMessage()
                if "/socket.io/" in message:
                    return False
                return True
        
        # 获取 uvicorn access logger 并添加过滤器
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        uvicorn_access_logger.addFilter(SocketIOFilter())
        
        uvicorn.run(
            app,
            host="0.0.0.0",  # 监听所有网络接口
            port=SERVER_PORT,  # 从环境变量获取的端口
            reload=DEBUG_MODE,  # 调试模式下自动重载代码变更
            log_level="info",  # 日志级别
            timeout_keep_alive=120,  # keep-alive 超时时间（秒）
        )

    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}", exc_info=True)
        raise


# 当直接运行此脚本时（不是作为模块导入），执行 main() 函数
if __name__ == "__main__":
    main()

# ==================== ASGI 入口（供 Gunicorn 等使用） ====================
# 这里再次创建 app 实例，以便像 Gunicorn + UvicornWorker 这样的生产服务器能够导入 app 变量
# 注意：该 app 实例没有经过 init_services() 初始化，通常生产环境中初始化工作会在启动脚本中单独完成；
# 但这种写法可能导致服务未初始化的问题。原代码保持原样，我们仅添加注释，不做修改。
app = create_app()
