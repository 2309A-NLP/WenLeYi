# -*- coding: utf-8 -*-
"""
API接口模块：所有路由接口、请求处理、业务逻辑。

本文件实现的核心功能：
1. 用户认证（注册、登录）
2. 角色管理（获取内置角色、创建/更新自定义角色）
3. 核心对话（普通对话 + 流式对话，支持文件上传）
4. 多源并行检索（长期记忆、SQLite角色知识库、Milvus混合检索、股票数据）
5. 检索结果重排序（BGE-Reranker）
6. 上下文构建与大模型调用（支持多种LLM提供商）
7. 短期记忆（Redis）与长期记忆（Milvus）的读写
8. 知识库文档上传（TXT/PDF）与删除
9. 股票数据手动同步
10. 系统健康检查与知识库统计

本文件作为FastAPI路由注册中心，依赖 database.py（数据层）和 utils.py（工具层）。
"""

import logging
import time
import io
import base64
import os
import sys
import threading
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from fastapi import HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 添加当前目录到路径，支持直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入 utils 模块（工具函数、配置、模型加载）
try:
    from utils import (
        logger as utils_logger, TOP_K_RERANK, MAX_CONTEXT_LENGTH,
        get_cached_answer, set_cached_answer, rewrite_query,
        is_probably_not_a_question, polite_clarify_answer,
        get_llm_client, get_llm_model_name, normalize_answer_text,
        chunk_text
    )
except ImportError:
    from .utils import (
        logger as utils_logger, TOP_K_RERANK, MAX_CONTEXT_LENGTH,
        get_cached_answer, set_cached_answer, rewrite_query,
        is_probably_not_a_question, polite_clarify_answer,
        get_llm_client, get_llm_model_name, normalize_answer_text,
        chunk_text
    )

# 导入 database 模块（数据库操作：MySQL、Milvus、Redis）
try:
    from database import (
        register_user, login_user, get_user_info, get_roles, get_role_prompt,
        get_role_name, retrieve_sqlite_role_knowledge, build_database_grounded_answer,
        get_chat_history, add_to_history, clear_history,
        hybrid_search, rerank_documents,
        retrieve_long_term_memory, store_long_term_memory,
        retrieve_stock_data, add_documents_to_milvus,
        sync_stock_data_to_milvus, milvus_ready, init_database, init_milvus,
        diagnose_milvus_connection, set_milvus_available, seed_role_knowledge_if_empty,
        get_db_connection, STOCK_COLLECTION_NAME, _load_local_stock_cache
    )
except ImportError:
    from .database import (
        register_user, login_user, get_user_info, get_roles, get_role_prompt,
        get_role_name, retrieve_sqlite_role_knowledge, build_database_grounded_answer,
        get_chat_history, add_to_history, clear_history,
        hybrid_search, rerank_documents,
        retrieve_long_term_memory, store_long_term_memory,
        retrieve_stock_data, add_documents_to_milvus,
        sync_stock_data_to_milvus, milvus_ready, init_database, init_milvus,
        diagnose_milvus_connection, set_milvus_available, seed_role_knowledge_if_empty,
        get_db_connection, STOCK_COLLECTION_NAME, _load_local_stock_cache
    )

logger = logging.getLogger(__name__)


# ==================== Pydantic 模型（请求/响应数据结构） ====================

class FileData(BaseModel):
    """文件数据模型，用于接收前端上传的 Base64 编码文件"""
    name: str          # 文件名
    type: str          # 文件类型（如 "text/plain", "application/pdf"）
    data: str          # Base64 编码的文件内容


class ChatRequest(BaseModel):
    """聊天请求模型，包含用户消息、角色信息、流式选项等"""
    user_id: int                          # 用户ID
    role_id: int                          # 角色ID（对话的 AI 角色）
    message: str                          # 用户发送的文本消息
    stream: bool = False                  # 是否启用流式响应（默认为 False）
    llm_provider: Optional[str] = None    # 可选，指定使用的 LLM 提供商（如 "openai", "qwen"）
    file: Optional[FileData] = None       # 可选，上传的文件数据（如图片、文档）


class ChatResponse(BaseModel):
    """聊天响应模型，用于非流式模式返回完整回答"""
    answer: str                                   # 助手生成的回答文本
    sources: Optional[List[Dict[str, str]]] = None  # 可选，引用来源列表（每个来源包含 text, source 等）


class UserLogin(BaseModel):
    """用户登录请求模型"""
    username: str   # 用户名
    password: str   # 密码


class UserRegister(BaseModel):
    """用户注册请求模型"""
    username: str          # 用户名（必填）
    password: str          # 密码（必填）
    email: Optional[str] = None   # 可选，电子邮箱
    phone: Optional[str] = None   # 可选，手机号码


class UserResponse(BaseModel):
    """用户信息响应模型，用于返回用户基本信息（不含密码）"""
    user_id: int                  # 用户ID
    username: str                 # 用户名
    email: Optional[str] = None   # 电子邮箱（可能为空）
    phone: Optional[str] = None   # 手机号码（可能为空）


class DocumentUploadRequest(BaseModel):
    """文档上传请求模型（用于将文本导入知识库）"""
    source: str                     # 文档来源标识（如 "user_upload", "url"）
    doc_id: Optional[str] = None    # 可选，文档唯一标识（不提供时自动生成）


class RoleResponse(BaseModel):
    """角色信息响应模型，返回角色的完整配置"""
    id: int                                      # 角色ID
    role_name: str                               # 角色内部名称（用于代码逻辑）
    display_name: Optional[str] = None           # 显示的友好名称（前端展示用）
    description: Optional[str] = None            # 角色描述（简短介绍）
    personality: Optional[str] = None            # 角色性格设定
    tone: Optional[str] = None                   # 说话语气风格（如 "专业", "幽默"）
    background_story: Optional[str] = None       # 角色背景故事
    hobbies: Optional[str] = None                # 角色兴趣爱好
    prompt_template: Optional[str] = None        # 系统提示词模板（用于 LLM）


class RoleCreateRequest(BaseModel):
    """角色创建请求模型"""
    role_name: str                           # 角色名称（唯一标识）
    display_name: str                        # 显示名称
    description: Optional[str] = None        # 描述
    personality: str                         # 性格（必填）
    tone: str                                # 语气（必填）
    background_story: str                    # 背景故事（必填）
    hobbies: str                             # 兴趣爱好（必填）
    prompt_template: Optional[str] = None    # 提示词模板（可选）
    creator_user_id: Optional[int] = None    # 创建者用户ID（可选，用于权限控制）


class RoleUpdateRequest(BaseModel):
    """角色信息更新请求模型（所有字段可选）"""
    display_name: Optional[str] = None        # 显示名称
    description: Optional[str] = None         # 描述
    personality: Optional[str] = None         # 性格
    tone: Optional[str] = None                # 语气
    background_story: Optional[str] = None    # 背景故事
    hobbies: Optional[str] = None             # 兴趣爱好
    prompt_template: Optional[str] = None     # 提示词模板


class KnowledgeQueryRequest(BaseModel):
    """知识库查询请求模型"""
    query: str           # 查询文本
    top_k: int = 5       # 返回最相关的前 K 条知识（默认 5）
    role_id: int = 1     # 角色ID（默认为 1，即通用角色）


class StockSyncResponse(BaseModel):
    """股票数据同步响应模型"""
    status: str                 # 同步状态（如 "success", "failed"）
    count: Optional[int] = None # 同步的股票数量（可选）


class KnowledgeStatsResponse(BaseModel):
    """知识库统计信息响应模型"""
    knowledge_count: int   # 角色知识库中的文档数量
    stock_count: int       # 股票数据集合中的条目数量
    memory_count: int      # 长期记忆集合中的记录数量
    doc_count: int         # 文档总数（可能针对某个角色）
    bm25_ready: bool       # BM25 稀疏检索模型是否已就绪（可用于当前集合）

# ==================== 大模型调用相关 ====================
def build_llm_messages(
        role_id: int,
        message: str,
        context: str,
        history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """根据角色设置和检索上下文构建发送给LLM的消息列表"""
    # 获取角色的系统提示词模板（如果未配置则为空字符串）
    prompt_template = get_role_prompt(role_id) or ""
    # 获取角色的内部名称（例如 "doctor", "stock_analyst"）
    role_name = get_role_name(role_id) or "assistant"

    # 将内部角色名称映射为中文显示名称（用于系统提示中的角色描述）
    role_display = {
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
    }.get(role_name, role_name)  # 如果未在映射表中找到，则使用原始英文名称

    # 构建系统提示词（system prompt），定义角色行为准则和输出规则
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
6. 除非用户明确要求，不要空泛地说'如有需要我可以继续帮助你'。
""".strip()  # 去除首尾空白

    # 初始化消息列表，第一条为系统消息（角色设定和规则）
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # 如果提供了检索上下文且不为空，则添加第二条系统消息，将数据库检索结果注入
    if context and context.strip():
        messages.append({"role": "system", "content": f"以下是可参考的数据库与检索结果，请优先吸收后再回答：\n{context}"})

    # 添加最近8条对话历史作为上下文（保持多轮对话连贯性）
    # 遍历 history 列表的最后8项（如果不足8项则全部使用）
    for item in history[-8:]:
        # 只添加角色为 "user" 或 "assistant" 且内容非空的消息
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})

    # 最后添加当前用户消息
    messages.append({"role": "user", "content": message})
    return messages


# 回答缓存（内存级，TTL 300秒）
_answer_cache = {}                      # 缓存字典，键为缓存键（如消息哈希），值为 {"result": 回答内容, "time": 时间戳}
_answer_cache_lock = threading.Lock()   # 线程锁，用于保护缓存字典的并发访问
_DISABLE_CACHE = os.getenv("DISABLE_CACHE", "false").lower() == "true"  # 从环境变量读取是否禁用缓存（字符串 "true" 转为布尔 True）


def generate_response(role_id: int, user_id: int, message: str, context: str, history: List[Dict[str, str]],
                      stream: bool = False, llm_provider: str = None):
    """
    生成大模型回答（支持流式与非流式）
    - 优先检查缓存（非流式且未禁用缓存时）
    - 构建消息列表并调用LLM API
    - 流式输出时实时返回并最后存储记忆；非流式时直接返回完整答案
    """
    # 检查是否禁用了缓存（通过环境变量 DISABLE_CACHE 控制）
    if not _DISABLE_CACHE:
        # 构造缓存键：角色ID + 消息前100字符 + 上下文前50字符（避免过长）
        cache_key = f"{role_id}:{message[:100]}:{context[:50]}"
        # 使用线程锁保护缓存字典的并发读写
        with _answer_cache_lock:
            # 检查缓存中是否存在该键
            if cache_key in _answer_cache:
                cache_entry = _answer_cache[cache_key]  # 获取缓存条目（包含结果和时间戳）
                # 判断缓存是否在有效期内（300秒）
                if time.time() - cache_entry["time"] < 300:
                    logger.info(f"命中回答缓存: {message[:30]}...")
                    cached_answer = cache_entry["result"]  # 获取缓存的回答文本
                    # 如果请求是流式输出，但缓存命中的是完整答案，为了保持接口一致性，将结果包装成流式生成器
                    if stream:
                        # 定义一个异步生成器，只 yield 一次完整的缓存答案
                        async def cached_stream():
                            yield cached_answer
                        # 返回 StreamingResponse，媒体类型为纯文本
                        return StreamingResponse(cached_stream(), media_type="text/plain; charset=utf-8")
                    # 非流式请求直接返回缓存答案
                    return cached_answer

    # 构建发送给大模型的消息列表（包含系统提示、检索上下文、对话历史、当前用户消息）
    messages = build_llm_messages(role_id, message, context, history)
    # 获取大模型客户端实例（根据 llm_provider 选择，如 openai、qwen 等）
    client = get_llm_client(llm_provider)
    # 获取对应提供商使用的模型名称（如 "gpt-4o-mini", "qwen-turbo"）
    model_name = get_llm_model_name(llm_provider)

    # 判断是否为流式请求
    if stream:
        # ---------- 流式请求处理 ----------
        # 调用大模型 API，开启流式输出（stream=True）
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=2048,        # 最大输出 token 数
            temperature=0.2,        # 低温度，使输出更确定性
            stream=True,            # 启用流式
            timeout=120             # 超时时间（秒）
        )

        # 定义异步生成器函数，用于逐 token 返回并最后执行后续存储操作
        async def generate_and_store():
            """异步生成器：边接收流式token边返回，结束后存储记忆并更新缓存"""
            full_text_parts: List[str] = []  # 收集所有 token 片段，用于最终拼接完整回答
            try:
                logger.info("开始流式输出...")
                # 遍历响应中的每个消息块（chunk）
                for chunk in response:
                    # 提取当前 chunk 中的增量内容（delta.content）
                    delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                    if delta:
                        logger.debug(f"流式输出: {delta[:20]}...")
                        full_text_parts.append(delta)  # 收集片段
                        yield delta                    # 立即将片段返回给客户端
                logger.info(f"流式输出完成，总长度: {len(''.join(full_text_parts))}")
            finally:
                # 无论流式输出是否异常，最后都要进行记忆存储和缓存更新
                try:
                    # 将收集到的所有片段拼接成完整回答，并规范化文本格式
                    full_text = normalize_answer_text("".join(full_text_parts))
                    if full_text:
                        # 存储短期记忆（对话历史，通常放在内存或数据库中，用于多轮对话上下文）
                        add_to_history(user_id, role_id, message, full_text)
                        # 存储长期记忆（向量化后存入 Milvus，用于长期知识积累）
                        store_long_term_memory(user_id, role_id, message, full_text)
                        # 如果未禁用缓存，将本次回答存入缓存
                        if not _DISABLE_CACHE:
                            with _answer_cache_lock:
                                # 重新生成缓存键（与前面的键相同）
                                cache_key_local = f"{role_id}:{message[:100]}:{context[:50]}"
                                # 存储回答和当前时间戳
                                _answer_cache[cache_key_local] = {"result": full_text, "time": time.time()}
                                # 限制缓存大小不超过 200 条，超过时删除最早的条目
                                if len(_answer_cache) > 200:
                                    oldest_key = min(_answer_cache.keys(), key=lambda k: _answer_cache[k]["time"])
                                    del _answer_cache[oldest_key]
                except Exception as e:
                    logger.warning(f"流式回答落库失败: {e}")

        # 返回 StreamingResponse，数据源为上面定义的异步生成器，媒体类型为纯文本（UTF-8 编码）
        return StreamingResponse(generate_and_store(), media_type="text/plain; charset=utf-8")

    else:
        # ---------- 非流式请求处理 ----------
        # 调用大模型 API，非流式（一次完成）
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            timeout=120
        )
        # 提取模型返回的文本内容，去除首尾空白，并进行规范化（例如去除多余换行、统一标点等）
        answer = normalize_answer_text(response.choices[0].message.content.strip())

        # 如果未禁用缓存，将本次回答存入缓存
        if not _DISABLE_CACHE:
            with _answer_cache_lock:
                cache_key = f"{role_id}:{message[:100]}:{context[:50]}"  # 生成缓存键
                _answer_cache[cache_key] = {"result": answer, "time": time.time()}  # 存储回答和时间戳
                # 限制缓存大小不超过 200 条，超出则删除最早的一条
                if len(_answer_cache) > 200:
                    oldest_key = min(_answer_cache.keys(), key=lambda k: _answer_cache[k]["time"])
                    del _answer_cache[oldest_key]

        # 直接返回完整的回答字符串
        return answer


# ==================== 文件解析函数 ====================
def parse_file_content(file_data: FileData) -> str:
    """
    解析前端传来的 Base64 编码文件内容，提取文本信息。
    支持格式：txt, md, json, pdf, doc/docx, 图片（OCR）, 语音（ASR）, 视频（帧提取+OCR）
    """
    try:
        # 提取 Base64 数据（可能带有 data:xxx;base64, 前缀）
        # 判断文件数据的字符串是否以 "data:" 开头（Data URI 格式）
        if file_data.data.startswith('data:'):
            # 按逗号分割 Data URI，前半部分是元数据（如 "data:text/plain;base64"），后半部分是纯 Base64 数据
            data_uri_parts = file_data.data.split(',')
            # 确保分割后至少有 2 部分，取第二部分作为 Base64 数据
            if len(data_uri_parts) > 1:
                base64_data = data_uri_parts[1]
            else:
                # 如果只有一个部分（没有逗号），则整个字符串就是 Base64 数据
                base64_data = file_data.data
        else:
            # 如果不是 Data URI 格式，则整个字符串就是 Base64 数据
            base64_data = file_data.data

        # 将 Base64 字符串解码为原始字节数据
        file_bytes = base64.b64decode(base64_data)

        # 获取文件名的全小写形式，便于后缀名匹配
        file_name = file_data.name.lower()
        # 获取文件 MIME 类型的全小写形式
        file_type = file_data.type.lower()

        # 纯文本类：.txt, .md, .json 文件
        if file_name.endswith('.txt') or file_name.endswith('.md') or file_name.endswith('.json'):
            # 尝试用 UTF-8 解码字节数据，忽略解码错误（替换非法字符）
            return file_bytes.decode('utf-8', errors='ignore')

        # PDF 文件处理
        elif file_name.endswith('.pdf'):
            try:
                # 动态导入 PyPDF2 库（用于读取 PDF）
                from PyPDF2 import PdfReader
                # 使用字节流创建 PdfReader 对象
                pdf_reader = PdfReader(io.BytesIO(file_bytes))
                text = ""  # 初始化空字符串，用于收集所有页面的文本
                # 遍历 PDF 的每一页
                for page in pdf_reader.pages:
                    # 提取当前页的文本
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text  # 将有效文本追加到结果中
                # 如果提取到内容则返回，否则返回默认提示
                return text if text else "无法提取PDF内容"
            except Exception as e:
                # 记录警告日志并返回错误信息
                logger.warning(f"PDF解析失败: {e}")
                return f"PDF文件解析失败: {str(e)}"

        # Word 文档处理（.doc 和 .docx）
        elif file_name.endswith('.doc') or file_name.endswith('.docx'):
            try:
                # 动态导入 python-docx 库
                from docx import Document
                # 使用字节流打开 Word 文档
                doc = Document(io.BytesIO(file_bytes))
                # 提取所有段落的文本，用换行符连接
                text = "\n".join([para.text for para in doc.paragraphs])
                return text if text else "无法提取Word内容"
            except Exception as e:
                logger.warning(f"Word解析失败: {e}")
                return f"Word文件解析失败: {str(e)}"

        # 图片文件处理（OCR 识别）
        elif 'image' in file_type or file_name.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            try:
                # 导入 PIL 库中的 Image 模块
                from PIL import Image
                # 使用字节流打开图片
                image = Image.open(io.BytesIO(file_bytes))
                # 记录图片尺寸和颜色模式，便于调试
                logger.info(f"图片尺寸: {image.size}, 模式: {image.mode}")

                # 调用内部 OCR 函数进行文字识别（根据环境选择合适的 OCR 引擎）
                text = _perform_ocr(file_bytes, image)
                return text

            except ImportError as e:
                # 如果缺少 OCR 依赖库，给出安装提示
                logger.warning(f"图片识别依赖未安装: {e}")
                return "图片识别功能未启用，请安装以下依赖之一：\n1. pip install pytesseract Pillow\n2. pip install easyocr\n3. pip install paddlepaddle paddleocr"
            except Exception as e:
                logger.error(f"图片识别异常: {e}")
                return f"图片识别失败: {str(e)}"

        # 语音文件处理（ASR 自动语音识别）
        elif 'audio' in file_type or file_name.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
            try:
                logger.info(f"开始语音识别: {file_name}, 大小: {len(file_bytes)} bytes")
                # 调用内部语音识别函数，传入原始字节数据和文件名（用于推断格式）
                text = _perform_speech_recognition(file_bytes, file_name)
                return text
            except Exception as e:
                logger.error(f"语音识别异常: {e}")
                return f"语音识别失败: {str(e)}"

        # 视频文件处理（提取关键帧 + OCR 识别）
        elif 'video' in file_type or file_name.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            try:
                logger.info(f"开始视频处理: {file_name}, 大小: {len(file_bytes)} bytes")
                # 调用内部视频处理函数，提取帧并进行 OCR
                text = _process_video(file_bytes, file_name)
                return text
            except Exception as e:
                logger.error(f"视频处理异常: {e}")
                return f"视频处理失败: {str(e)}"

        else:
            # 不支持的格式，返回错误提示
            return f"不支持的文件类型: {file_name}"

    except Exception as e:
        # 最外层异常捕获：任何未预料到的错误都记录并返回通用错误信息
        logger.error(f"文件解析异常: {e}")
        return f"文件解析错误: {str(e)}"


def _perform_ocr(file_bytes: bytes, image=None) -> str:
    """执行 OCR 识别"""
    text = None
    
    # 尝试 Tesseract OCR
    try:
        import pytesseract
        from PIL import Image
        
        if image is None:
            image = Image.open(io.BytesIO(file_bytes))
            
        # Windows 下自动搜索 tesseract.exe 路径
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
        img_gray = image.convert('L')
        img_binary = img_gray.point(lambda x: 0 if x < 128 else 255, '1')
        text = pytesseract.image_to_string(img_binary, lang='chi_sim+eng')
        logger.info(f"Tesseract OCR识别完成，长度: {len(text)}字符")
    except ImportError:
        logger.warning("Tesseract未安装，尝试其他OCR方案")
    except Exception as e:
        logger.warning(f"Tesseract OCR失败: {e}")

    # 若 Tesseract 无结果，尝试 EasyOCR
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

    # 若仍无结果，返回提示信息
    if text and text.strip():
        return text
    else:
        if image:
            return f"【图片识别结果】\n图片信息: {image.size}, {image.mode}\n\n⚠️ 图片中未识别到文字内容。"
        return "⚠️ 图片中未识别到文字内容。"


def _perform_speech_recognition(file_bytes: bytes, file_name: str) -> str:
    """执行语音识别（ASR）"""
    # 方法1: 使用 SpeechRecognition 库
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        
        # 将字节数据写入临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{file_name.split('.')[-1]}", delete=False) as f:
            f.write(file_bytes)
            temp_path = f.name
        
        try:
            with sr.AudioFile(temp_path) as source:
                audio_data = recognizer.record(source)
            
            # 尝试使用 Google 语音识别
            try:
                text = recognizer.recognize_google(audio_data, language='zh-CN')
                logger.info(f"Google ASR识别完成，长度: {len(text)}字符")
                return f"【语音识别结果】\n{text}"
            except sr.UnknownValueError:
                logger.warning("Google ASR无法识别语音")
            except sr.RequestError as e:
                logger.warning(f"Google ASR请求失败: {e}")
                
            # 尝试使用 CMU Sphinx（离线）
            try:
                text = recognizer.recognize_sphinx(audio_data, language='zh-CN')
                logger.info(f"Sphinx ASR识别完成，长度: {len(text)}字符")
                return f"【语音识别结果（离线）】\n{text}"
            except sr.UnknownValueError:
                logger.warning("Sphinx ASR无法识别语音")
            except sr.RequestError as e:
                logger.warning(f"Sphinx ASR请求失败: {e}")
                
        finally:
            import os
            os.unlink(temp_path)
            
        return "⚠️ 语音识别失败：无法识别语音内容或服务不可用。"
        
    except ImportError:
        logger.warning("SpeechRecognition未安装")
        
    # 方法2: 使用 Whisper（OpenAI）
    try:
        import whisper
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{file_name.split('.')[-1]}", delete=False) as f:
            f.write(file_bytes)
            temp_path = f.name
        
        try:
            model = whisper.load_model("base")
            result = model.transcribe(temp_path, language='zh')
            text = result["text"]
            logger.info(f"Whisper ASR识别完成，长度: {len(text)}字符")
            return f"【语音识别结果】\n{text}"
        finally:
            import os
            os.unlink(temp_path)
            
    except ImportError:
        logger.warning("Whisper未安装")
    except Exception as e:
        logger.warning(f"Whisper ASR失败: {e}")
    
    return "语音识别功能未启用，请安装以下依赖之一：\n1. pip install SpeechRecognition pyaudio\n2. pip install openai-whisper"


def _process_video(file_bytes: bytes, file_name: str) -> str:
    """处理视频文件（提取帧并进行OCR）"""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix=f".{file_name.split('.')[-1]}", delete=False) as f:
        f.write(file_bytes)
        temp_path = f.name
    
    try:
        # 尝试使用 OpenCV 提取帧
        try:
            import cv2
            
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return "无法打开视频文件"
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"视频信息: {fps:.1f} FPS, {frame_count} 帧, 时长: {duration:.1f}秒")
            
            # 提取关键帧（每5秒取一帧）
            frame_interval = int(fps * 5) if fps > 0 else 30
            ocr_results = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # 将帧转换为 PIL Image
                    from PIL import Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    # 执行 OCR
                    text = _perform_ocr(cv2.imencode('.png', frame)[1].tobytes(), image)
                    if text and text.strip() and "未识别到文字" not in text:
                        ocr_results.append(f"【第 {frame_idx // int(fps) + 1} 秒】\n{text}")
                
                frame_idx += 1
            
            cap.release()
            
            if ocr_results:
                return f"【视频分析结果】\n视频信息: {fps:.1f} FPS, {frame_count} 帧, 时长: {duration:.1f}秒\n\n识别到的文字内容:\n" + "\n\n".join(ocr_results)
            else:
                return f"【视频分析结果】\n视频信息: {fps:.1f} FPS, {frame_count} 帧, 时长: {duration:.1f}秒\n\n⚠️ 视频中未识别到文字内容。"
            
        except ImportError:
            logger.warning("OpenCV未安装")
            return "视频处理功能未启用，请安装依赖：pip install opencv-python"
        except Exception as e:
            logger.error(f"视频帧提取失败: {e}")
            return f"视频帧提取失败: {str(e)}"
            
    finally:
        os.unlink(temp_path)


# ==================== API 接口实现 ====================
def register_api_routes(app):
    """注册所有API路由到 FastAPI 应用"""

    # ---------- 用户认证 ----------
    @app.post("/register", response_model=UserResponse)
    async def register(user: UserRegister):
        """用户注册接口"""
        # 调用 register_user 函数注册新用户，传入用户名、密码、邮箱、手机号
        # 返回 True 表示注册成功（用户名未被占用），False 表示用户名已存在
        success = register_user(user.username, user.password, user.email, user.phone)
        if not success:
            # 如果用户名已存在，返回 HTTP 400 错误
            raise HTTPException(status_code=400, detail="用户名已存在")

        # 获取数据库连接
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            # 查询刚注册的用户信息
            cursor.execute("SELECT id, username, email, phone FROM users WHERE username=%s", (user.username,))
            result = cursor.fetchone()
            if result:
                # 构造并返回用户响应对象
                return UserResponse(
                    user_id=result[0],
                    username=result[1],
                    email=result[2],
                    phone=result[3]
                )
            # 理论上应存在，若不存在则返回服务器错误
            raise HTTPException(status_code=500, detail="注册失败")
        finally:
            # 确保关闭数据库连接
            conn.close()

    @app.post("/login", response_model=UserResponse)
    async def login(user: UserLogin):
        """用户登录接口"""
        # 验证用户名和密码，成功返回用户ID，失败返回 None
        user_id = login_user(user.username, user.password)
        if user_id is None:
            # 认证失败，返回 401 未授权
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        # 获取用户详细信息（字典形式）
        user_info = get_user_info(user_id)
        if user_info:
            # 将字典解包并构造响应对象返回
            return UserResponse(**user_info)
        # 如果获取失败，返回服务器错误
        raise HTTPException(status_code=500, detail="获取用户信息失败")

    # ---------- 对话接口 ----------
    @app.post("/api/chat_stream")
    async def chat_stream_api(req: ChatRequest):
        """
        流式对话接口（支持文件上传）
        1. 解析文件内容（如有）
        2. 获取短期记忆
        3. Query 改写
        4. 并行执行四路检索（长期记忆、SQLite知识、Milvus混合检索、股票数据）
        5. 重排序检索结果
        6. 构建上下文
        7. 调用 generate_response() 流式返回
        """
        # 记录请求基本信息
        logger.info(f"收到流式聊天请求: user_id={req.user_id}, role_id={req.role_id}, has_file={bool(req.file)}")

        # 解析文件内容（默认空字符串）
        file_content = ""
        if req.file:
            # 记录文件元信息
            logger.info(
                f"收到文件: {req.file.name}, 类型: {req.file.type}, 数据长度: {len(req.file.data) if req.file.data else 0}")
            # 调用解析函数提取文本
            file_content = parse_file_content(req.file)
            logger.info(f"文件内容提取完成，长度: {len(file_content)}字符")

        # 获取用户与角色的对话历史（短期记忆）
        history = get_chat_history(req.user_id, req.role_id)
        # 结合历史对用户问题进行重写，以便更好地检索
        rewritten_query = rewrite_query(req.message, history)
        logger.info(f"原问题: {req.message[:50]}... -> 改写后: {rewritten_query[:50]}...")

        # ============================================================================
        # 🔥 并行检索模块（核心优化）
        # 采用 ThreadPoolExecutor 实现四路并行检索，显著提升检索效率
        # 四个数据源同时检索，总耗时取决于最慢的一个任务
        # ============================================================================

        # -------------------------- 任务定义 --------------------------
        # 每个任务独立封装，内部捕获异常，保证单个任务失败不影响其他任务

        def task_long_term_memory():
            """
            任务1: 检索长期记忆
            从 Milvus 的 long_term_memory 集合中检索与当前问题相关的历史对话
            使用向量化相似性匹配，返回用户之前的对话记录作为上下文
            """
            try:
                # 参数: 用户ID, 角色ID, 查询语句
                return retrieve_long_term_memory(req.user_id, req.role_id, rewritten_query)
            except Exception as e:
                logger.debug(f"长期记忆检索失败: {e}")
                return []  # 失败时返回空列表，不影响其他任务

        def task_sqlite_knowledge():
            """
            任务2: 检索 SQLite 角色知识库
            查询角色专属的本地知识库（MySQL/SQLite兜底）
            当 Milvus 不可用时提供备选知识来源
            """
            try:
                # 参数: 角色ID, 查询语句
                return retrieve_sqlite_role_knowledge(req.role_id, rewritten_query)
            except Exception as e:
                logger.debug(f"SQLite知识检索失败: {e}")
                return []

        def task_hybrid_search():
            """
            任务3: Milvus 混合检索（核心检索）
            同时进行稠密向量检索(IVF_FLAT)和稀疏向量检索(BM25)
            通过 RRF (Reciprocal Rank Fusion) 融合结果
            这是 RAG 系统的核心检索能力
            """
            try:
                # 参数: 查询语句, 角色ID
                return hybrid_search(rewritten_query, req.role_id)
            except Exception as e:
                logger.debug(f"Milvus混合检索失败: {e}")
                return []

        def task_stock_data():
            """
            任务4: 检索股票数据
            从 Milvus 的 stock_data 集合中检索相关股票信息
            包括股票代码、名称、行情数据等
            """
            try:
                # 参数: 查询语句
                return retrieve_stock_data(rewritten_query)
            except Exception as e:
                logger.debug(f"股票数据检索失败: {e}")
                return []

        # -------------------------- 线程池执行 --------------------------
        # 创建结果字典，用于存储各任务的返回结果
        results = {}

        # 创建线程池，最大工作线程数为4（与任务数匹配）
        # ThreadPoolExecutor 会自动管理线程的创建和复用
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有任务到线程池，返回 Future 对象
            # Future 是一个代表异步操作结果的对象
            futures = {
                executor.submit(task_long_term_memory): "long_term",  # 任务1
                executor.submit(task_sqlite_knowledge): "sqlite",     # 任务2
                executor.submit(task_hybrid_search): "hybrid",       # 任务3（核心）
                executor.submit(task_stock_data): "stock",           # 任务4
            }

            try:
                # 遍历已完成的任务（as_completed 会按完成顺序返回）
                # 设置总超时时间为45秒，防止某个慢任务阻塞整个流程
                for future in as_completed(futures, timeout=45):
                    # 获取任务名称（从 Future 到名称的映射）
                    task_name = futures[future]
                    try:
                        # 获取任务结果（会阻塞直到该任务完成）
                        results[task_name] = future.result()
                    except Exception as e:
                        # 单个任务执行过程中发生异常
                        logger.warning(f"检索任务 {task_name} 执行异常: {e}")
                        results[task_name] = []  # 降级为空结果

            except TimeoutError:
                # 总体超时处理：部分任务在45秒内未完成
                logger.warning(f"部分检索任务超时（45秒），已完成的将直接使用，未完成的使用空结果")
                # 处理未完成的任务，设置为空列表
                for future_obj in futures:
                    if future_obj not in results and not future_obj.done():
                        results[futures[future_obj]] = []

        # -------------------------- 结果提取 --------------------------
        # 从结果字典中提取各路检索结果，没有的默认为空列表
        long_term_memories = results.get("long_term", [])    # 长期记忆结果
        sqlite_knowledge = results.get("sqlite", [])        # SQLite知识库结果
        retrieved_docs = results.get("hybrid", [])          # 混合检索结果（核心）
        stock_info = results.get("stock", [])               # 股票数据结果

        # 对混合检索的结果进行重排序（使用 BGE-reranker 等模型）
        reranked_docs = rerank_documents(rewritten_query, retrieved_docs)

        # 构建上下文：按优先级拼接不同的信息源
        context_parts = []
        if file_content:
            # 文件内容优先级最高
            context_parts.append(f"【用户上传文件内容】\n文件名: {req.file.name}\n\n{file_content}")
        if sqlite_knowledge:
            context_parts.append("【角色知识库】\n" + "\n\n".join(sqlite_knowledge))
        if reranked_docs:
            context_parts.append("【金融知识参考】\n" + "\n\n".join([doc["text"] for doc in reranked_docs]))
        if stock_info:
            # 格式化股票信息为易读文本
            stock_texts = [
                f"{s['stock_name']}({s['stock_code']}): 最新价{s['latest_price']}元, 涨跌幅{s['change_pct']}%, 市盈率{s['pe_ratio']}"
                for s in stock_info]
            context_parts.append("【股票行情数据】\n" + "\n\n".join(stock_texts))
        if long_term_memories:
            context_parts.append("【历史对话记忆】\n" + "\n\n".join(long_term_memories))
        # 合并所有部分，若无则给出默认提示
        context = "\n\n".join(context_parts) if context_parts else "无相关参考。"

        # 调用 generate_response 生成流式回答
        try:
            response = generate_response(
                req.role_id, req.user_id, req.message, context, history,
                stream=True, llm_provider=req.llm_provider
            )
            return response
        except Exception as e:
            # 如果生成失败，返回一个简单的错误流，并存储历史
            logger.warning(f"流式生成失败: {e}")
            answer = "很抱歉，我暂时无法处理您的请求。"
            add_to_history(req.user_id, req.role_id, req.message, answer)
            return StreamingResponse(iter([answer]), media_type="text/plain; charset=utf-8")

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """
        核心对话接口（非流式，或流式但统一返回 ChatResponse）
        处理逻辑与 /api/chat_stream 类似，但最终返回完整 JSON
        """
        # 获取对话历史
        history = get_chat_history(req.user_id, req.role_id)

        # 缓存检查（仅当没有文件上传时启用）
        if not req.file:
            cached_answer = get_cached_answer(req.user_id, req.role_id, req.message)
            if cached_answer:
                logger.info(f"缓存命中，直接返回")
                # 将缓存答案存储到短期历史中
                add_to_history(req.user_id, req.role_id, req.message, cached_answer)
                return ChatResponse(answer=cached_answer, sources=[])

        # 解析文件内容（如有）
        file_content = ""
        if req.file:
            logger.info(
                f"收到文件: {req.file.name}, 类型: {req.file.type}, 数据长度: {len(req.file.data) if req.file.data else 0}")
            file_content = parse_file_content(req.file)
            logger.info(f"文件内容提取完成，长度: {len(file_content)}字符")

        # 判断是否为有效问题（过滤寒暄/无意义信息）
        if is_probably_not_a_question(req.message):
            answer = polite_clarify_answer(req.message)
            add_to_history(req.user_id, req.role_id, req.message, answer)
            return ChatResponse(answer=answer, sources=[])

        # 重写查询
        rewritten_query = rewrite_query(req.message, history)
        logger.info(f"原始查询: {req.message} -> 改写后: {rewritten_query}")

        start_time = time.time()

        # 并行检索（与流式接口相同）
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

        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(task_long_term_memory): "long_term",
                executor.submit(task_sqlite_knowledge): "sqlite",
                executor.submit(task_hybrid_search): "hybrid",
                executor.submit(task_stock_data): "stock",
            }
            # 注意这里 timeout=15 秒，与流式接口不同
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

        # 重排序
        reranked_docs = rerank_documents(rewritten_query, retrieved_docs)

        retrieval_time = time.time() - start_time
        logger.info(f"多源检索完成，耗时: {retrieval_time:.2f}秒")

        # 构建上下文，注意限制长度
        context_parts = []
        if file_content:
            # 文件内容只取前500字符，防止过长
            context_parts.append(f"【用户上传文件内容】\n文件名: {req.file.name}\n\n{file_content[:500]}")
        if sqlite_knowledge:
            context_parts.append("【角色知识库】\n" + "\n\n".join(sqlite_knowledge))
        if reranked_docs:
            # 每个文档文本只取前400字符
            doc_texts = [doc["text"][:400] for doc in reranked_docs]
            context_parts.append("【金融知识参考】\n" + "\n\n".join(doc_texts))
        if stock_info:
            stock_texts = [
                f"{s['stock_name']}({s['stock_code']}): 最新价{s['latest_price']}元, 涨跌幅{s['change_pct']}%, 市盈率{s['pe_ratio']}"
                for s in stock_info]
            context_parts.append("【股票行情数据】\n" + "\n\n".join(stock_texts))
        if long_term_memories:
            # 长期记忆只取前2条
            context_parts.append("【历史对话记忆】\n" + "\n\n".join(long_term_memories[:2]))

        context = "\n\n".join(context_parts) if context_parts else "无相关参考。"
        # 如果上下文超过全局最大长度，截断并提示
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[:MAX_CONTEXT_LENGTH] + "\n\n【上下文已截断】"

        # 基于数据库直答的兜底回答（当 LLM 不可用时使用）
        db_grounded_answer = build_database_grounded_answer(
            req.role_id,
            req.message,
            sqlite_knowledge,
            stock_info=stock_info,
        ) if sqlite_knowledge else ""

        # 如果请求为流式，则进入流式分支（带超时控制）
        if req.stream:
            try:
                # 定义生成器函数
                def generate_with_timeout():
                    try:
                        return generate_response(
                            req.role_id, req.user_id, req.message, context, history,
                            stream=True, llm_provider=req.llm_provider
                        )
                    except Exception as e:
                        logger.warning(f"流式生成异常: {e}")
                        raise

                # 使用单线程池设置30秒超时
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(generate_with_timeout)
                    try:
                        response = future.result(timeout=30)
                        return response
                    except TimeoutError:
                        logger.warning("流式生成超时，回退数据库直答")
                        raise Exception("流式生成超时")
            except Exception as e:
                # 流式失败时降级为数据库直答
                logger.warning(f"流式模型回答失败，回退数据库直答: {e}")
                answer = db_grounded_answer or "我先按当前角色已有知识给你一个直接回答，但你可以再补充更具体一点的问题。"
                add_to_history(req.user_id, req.role_id, req.message, answer)
                try:
                    store_long_term_memory(req.user_id, req.role_id, req.message, answer)
                except Exception:
                    pass
                return StreamingResponse(iter([answer]), media_type="text/plain; charset=utf-8")

        # 非流式调用：生成完整回答
        try:
            answer = generate_response(
                req.role_id, req.user_id, req.message, context, history,
                stream=False, llm_provider=req.llm_provider
            )
        except Exception as e:
            logger.warning(f"模型回答失败，回退数据库直答: {e}")
            answer = db_grounded_answer or "我先按当前角色已有知识给你一个直接回答，但你可以再补充更具体一点的问题。"

        # 如果模型返回了通用开场白（如“您好！我是...”），且数据库直答存在，则替换为数据库直答
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

        # 存储记忆（短期和长期）
        add_to_history(req.user_id, req.role_id, req.message, answer)
        store_long_term_memory(req.user_id, req.role_id, req.message, answer)

        # 构建来源信息（用于前端展示引用）
        sources = []
        if reranked_docs:
            sources.extend([{"source": doc["source"], "summary": doc.get("summary", "")} for doc in reranked_docs])
        if stock_info:
            sources.extend(
                [{"source": "东方财富网", "summary": f"{s['stock_name']}({s['stock_code']})"} for s in stock_info])

        # 缓存答案（仅当无文件时）
        if not req.file:
            set_cached_answer(req.user_id, req.role_id, req.message, answer)

        return ChatResponse(answer=answer, sources=sources)

    # ---------- 历史管理 ----------
    @app.get("/chat_history/{user_id}/{role_id}")
    async def get_history(user_id: int, role_id: int):
        """获取指定用户和角色的聊天历史（短期记忆）"""
        history = get_chat_history(user_id, role_id)
        return {"history": history}

    @app.post("/clear_history/{user_id}/{role_id}")
    async def clear_chat_history(user_id: int, role_id: int):
        """清空指定用户和角色的聊天历史（短期记忆）"""
        clear_history(user_id, role_id)
        return {"status": "cleared"}

    # ---------- 角色管理 ----------
    @app.get("/roles", response_model=List[RoleResponse])
    async def get_all_roles():
        """获取所有可用角色列表"""
        roles = get_roles()
        return roles

    @app.post("/roles", response_model=RoleResponse)
    async def create_role(req: RoleCreateRequest):
        """创建自定义角色"""
        import re
        # 对 role_name 清洗和校验：只允许小写字母、数字、下划线，长度2-50
        role_name = req.role_name.strip().lower()
        if not re.match(r"^[a-z0-9_]{2,50}$", role_name):
            raise HTTPException(status_code=400, detail="role_name 仅支持小写字母、数字、下划线，长度 2-50")
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            # 检查是否已存在同名角色
            cursor.execute("SELECT id FROM roles WHERE role_name=%s", (role_name,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="role_name 已存在")
            # 获取请求的字典形式（兼容 Pydantic v1/v2）
            payload = req.dict() if hasattr(req, "dict") else req.model_dump()
            # 如果未提供 prompt_template，则根据角色属性自动构建
            try:
                from utils import _build_custom_role_prompt
            except ImportError:
                from .utils import _build_custom_role_prompt
            prompt_template = req.prompt_template or _build_custom_role_prompt(payload)
            # 插入数据库
            cursor.execute(
                """
                INSERT INTO roles
                (role_name, display_name, description, personality, tone, background_story, hobbies, prompt_template,
                 creator_user_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    role_name, req.display_name, req.description, req.personality, req.tone,
                    req.background_story, req.hobbies, prompt_template, req.creator_user_id
                ),
            )
            conn.commit()
            role_id = cursor.lastrowid
            # 查询刚插入的角色信息用于返回
            cursor.execute(
                """
                SELECT id,
                       role_name,
                       display_name,
                       description,
                       personality,
                       tone,
                       background_story,
                       hobbies,
                       prompt_template
                FROM roles
                WHERE id = %s
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
        """更新角色人设（性格、语气、背景、爱好等）"""
        # 获取需要更新的字段（排除 None 值）
        payload = req.dict(exclude_none=True) if hasattr(req, "dict") else req.model_dump(exclude_none=True)
        if not payload:
            raise HTTPException(status_code=400, detail="至少提供一个可更新字段")
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            # 先查询现有角色
            cursor.execute(
                """
                SELECT id,
                       role_name,
                       display_name,
                       description,
                       personality,
                       tone,
                       background_story,
                       hobbies,
                       prompt_template
                FROM roles
                WHERE id = %s
                """,
                (role_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="角色不存在")
            # 将现有数据转为字典
            role = {
                "id": row[0], "role_name": row[1], "display_name": row[2], "description": row[3],
                "personality": row[4], "tone": row[5], "background_story": row[6], "hobbies": row[7],
                "prompt_template": row[8]
            }
            # 更新字段
            role.update(payload)
            # 如果请求中没有显式提供 prompt_template，则根据更新后的角色属性自动重新生成
            if "prompt_template" not in payload:
                try:
                    from utils import _build_custom_role_prompt
                except ImportError:
                    from .utils import _build_custom_role_prompt
                role["prompt_template"] = _build_custom_role_prompt(role)
            # 执行更新
            cursor.execute(
                """
                UPDATE roles
                SET display_name=%s,
                    description=%s,
                    personality=%s,
                    tone=%s,
                    background_story=%s,
                    hobbies=%s,
                    prompt_template=%s
                WHERE id = %s
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
                background_story=role["background_story"], hobbies=role["hobbies"],
                prompt_template=role["prompt_template"]
            )
        finally:
            conn.close()

    # ---------- 知识库管理 ----------
    @app.post("/upload_document")
    async def upload_document(
            file: UploadFile = File(...),
            source: str = Form(...),
            doc_id: Optional[str] = Form(None),
            role_id: int = Form(1),
    ):
        """上传文档到知识库（仅支持 TXT 或 PDF）"""
        # 校验文件格式
        if not (file.filename.endswith(".txt") or file.filename.endswith(".pdf")):
            raise HTTPException(status_code=400, detail="仅支持 TXT 或 PDF 文件")

        content = await file.read()  # 读取文件二进制内容

        try:
            import fitz  # PyMuPDF，用于解析 PDF
            if file.filename.endswith(".pdf"):
                # 使用 PyMuPDF 解析 PDF
                doc = fitz.open("pdf", content)
                texts = []
                for page in doc:
                    texts.append(page.get_text())
                text = "\n".join(texts)
            else:
                # TXT 文件直接 UTF-8 解码
                text = content.decode("utf-8")

            # 对长文本进行分块（chunk）
            chunks = chunk_text(text)  # 假设 chunk_text 函数已定义
            # 将分块后的文本向量化并存储到 Milvus（角色知识库）
            add_documents_to_milvus(chunks, source, doc_id, role_id=int(role_id))
            return {"status": "success", "chunks_added": len(chunks), "filename": file.filename}
        except Exception as e:
            logger.error(f"上传文档失败: {e}")
            raise HTTPException(status_code=500, detail=f"上传文档失败: {str(e)}")

    @app.post("/query_knowledge")
    async def query_knowledge(req: KnowledgeQueryRequest):
        """直接查询指定角色的知识库（混合检索）"""
        if not milvus_ready():
            return {"results": []}
        results = hybrid_search(req.query, role_id=int(req.role_id), top_k=req.top_k)
        return {"results": results}

    @app.delete("/knowledge/{doc_id}")
    async def delete_knowledge(doc_id: str):
        """删除知识库中指定ID的文档（在所有角色的集合中尝试删除）"""
        from pymilvus import Collection
        try:
            from database import ensure_knowledge_collection, rebuild_bm25_model_for_collection
        except ImportError:
            from .database import ensure_knowledge_collection, rebuild_bm25_model_for_collection

        # 遍历所有角色（如果无法获取角色列表，则至少处理 role_id=1）
        for r in get_roles() or [{"id": 1}]:
            try:
                col = ensure_knowledge_collection(int(r["id"]))
                # 删除 doc_id 字段匹配的实体
                col.delete(f"doc_id == '{doc_id}'")
                col.flush()  # 强制持久化
                # 删除后需要重建该集合的 BM25 模型（因为文档变更）
                rebuild_bm25_model_for_collection(col.name)
            except Exception:
                continue
        return {"status": "success", "doc_id": doc_id}

    # ---------- 股票数据 ----------
    @app.post("/sync_stock_data", response_model=StockSyncResponse)
    async def sync_stock_data():
        """手动触发从东方财富同步股票数据到 Milvus"""
        if not milvus_ready():
            # 如果 Milvus 不可用，返回本地缓存的数量
            return {"status": "skipped", "count": len(_load_local_stock_cache())}
        stock_list = []
        try:
            try:
                from database import fetch_stock_data_from_eastmoney, store_stock_data_to_milvus
            except ImportError:
                from .database import fetch_stock_data_from_eastmoney, store_stock_data_to_milvus
            # 从东方财富 API 获取股票数据
            stock_list = fetch_stock_data_from_eastmoney()
            if stock_list:
                # 存储到 Milvus
                store_stock_data_to_milvus(stock_list)
                return {"status": "success", "count": len(stock_list)}
        except Exception as e:
            logger.error(f"同步股票数据失败: {e}")
        return {"status": "failed", "count": 0}

    # ---------- 系统接口 ----------
    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {"status": "ok", "service": "financial-rag-system"}

    @app.get("/knowledge_stats", response_model=KnowledgeStatsResponse)
    async def get_knowledge_stats():
        """获取知识库统计信息（知识条目数、股票数、记忆数、文档数、BM25就绪状态）"""
        try:
            knowledge_count = 0
            # 从本地缓存读取股票数量（如果没有 Milvus 则使用缓存）
            stock_count = len(_load_local_stock_cache())
            memory_count = 0
            if milvus_ready():
                from pymilvus import Collection
                # 累加所有角色的知识库条目数
                for r in get_roles() or [{"id": 1}]:
                    try:
                        try:
                            from database import ensure_knowledge_collection
                        except ImportError:
                            from .database import ensure_knowledge_collection
                        knowledge_count += ensure_knowledge_collection(int(r["id"])).num_entities
                    except Exception:
                        continue
                # 获取股票集合的实体数
                stock_collection = Collection(STOCK_COLLECTION_NAME)
                stock_count = stock_collection.num_entities
                # 获取长期记忆集合的实体数
                memory_collection = Collection("long_term_memory")
                memory_count = memory_collection.num_entities

            # 从 SQLite 获取知识文档总数（另一个来源）
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge_docs")
            doc_count = cursor.fetchone()[0]
            conn.close()

            # 检查 BM25 模型是否已构建（全局字典非空）
            try:
                from utils import get_bm25_models
            except ImportError:
                from .utils import get_bm25_models
            bm25_ready = milvus_ready() and any(get_bm25_models().values())

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

    # ---------- 空路由（处理未匹配的请求）----------
    @app.get("/socket.io/{path:path}")
    async def handle_socket_io(path: str):
        """
        处理 Milvus 客户端可能发起的 socket.io 请求
        Milvus Python SDK 在某些版本会尝试通过 socket.io 连接，这里返回空响应避免 404 错误
        """
        logger.debug(f"收到 socket.io 请求: /socket.io/{path}")
        return {"status": "ok"}

    return app