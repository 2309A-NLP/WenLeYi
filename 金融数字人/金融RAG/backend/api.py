# -*- coding: utf-8 -*-
"""
API接口模块：聊天、知识问答、股票查询
这是RAG系统的核心入口，负责接收用户问题、检索知识库、调用大模型、返回答案
"""

import hashlib
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

# 导入工具函数：LLM调用、知识库搜索、服务器端口配置
from utils import call_llm, search_knowledge_base, SERVER_PORT
# 导入知识库管理：加载所有问答对、获取统计信息
from knowledge_base import get_all_items, load_knowledge, get_statistics
from utils import logger as ut_logger

# 条件导入数据库模块（如果MySQL不可用，可以降级运行，不影响核心功能）
try:
    from database import (
        register_user, login_user, save_chat, get_recent_history, init_database
    )
    DB_AVAILABLE = True
except Exception as e:
    ut_logger.warning(f"数据库不可用（将使用无状态模式）: {e}")
    DB_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()  # FastAPI路由器，所有API接口都注册在这个路由器上

# 对话历史缓存（内存中，重启即清）
_chat_history_store: dict = {}


def _get_history(user_id, limit=4):
    """获取最近N条对话历史，用于让大模型"记住"上下文"""
    if user_id in _chat_history_store:
        return _chat_history_store[user_id][-limit:]
    return []


def _add_to_history(user_id, role, message):
    """添加一条对话历史到内存缓存"""
    if user_id not in _chat_history_store:
        _chat_history_store[user_id] = []
    _chat_history_store[user_id].append({"role": role, "content": message})
    # 只保留最近20条，防止内存溢出
    if len(_chat_history_store[user_id]) > 20:
        _chat_history_store[user_id] = _chat_history_store[user_id][-20:]


# ==================== 数据模型定义（Pydantic） ====================

class ChatRequest(BaseModel):
    """聊天请求的数据结构"""
    message: str              # 用户的问题文本
    user_id: Optional[int] = 0  # 用户ID，默认0表示匿名用户
    stream: bool = False      # 是否流式返回（暂未使用）
    use_knowledge: bool = True  # 是否启用知识库检索

class ChatResponse(BaseModel):
    """聊天响应的数据结构"""
    answer: str                        # 大模型生成的回答
    sources: Optional[List[Dict[str, str]]] = None  # 检索到的知识来源

class RegisterRequest(BaseModel):
    """用户注册请求"""
    username: str
    password: str

class LoginRequest(BaseModel):
    """用户登录请求"""
    username: str
    password: str

class AuthResponse(BaseModel):
    """认证响应"""
    success: bool
    message: str
    user_id: Optional[int] = None


# ==================== 金融领域系统提示词 ====================
# 这段提示词告诉大模型：你是金融助手，要专业、合规、友好地回答问题

FINANCIAL_SYSTEM_PROMPT = """你是一位专业的金融助手，回答以下问题：

专业知识：准确回答股票、基金、债券、保险等金融领域的问题
合规性：不推荐具体个股、不承诺收益，涉及投资建议时添加免责声明
严谨态度：数据引用需准确，不确定时如实说明
友好表达：用通俗易懂的语言解释专业概念

注意：
不做具体的短线买卖建议
说出"投资有风险，决策需谨慎"之类的免责说明
涉及法律法规时引用准确

输出要求：回复必须使用纯文本格式，不要使用任何 Markdown 符号（不要用 **、##、-、*、> 等符号）。用【】替代 Markdown 标题，用数字加顿号替代无序列表。

如果用户问非金融的问题，礼貌地引导回金融话题。"""


# ==================== API 接口定义 ====================

@router.get("/health")
def health_check():
    """健康检查接口：前端用来检测后端是否正常运行"""
    kb = load_knowledge()
    return {
        "status": "ok",
        "knowledge_items": kb["total"],      # 知识库总条目数
        "categories": kb["categories"],       # 知识库分类列表
        "server_port": SERVER_PORT,           # 服务端口
    }


@router.post("/register")
def register(req: RegisterRequest):
    """用户注册接口"""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="数据库不可用")
    # 用SHA256对密码进行哈希加密，不明文存储
    pw_hash = hashlib.sha256(req.password.encode()).hexdigest()
    uid = register_user(req.username, pw_hash)
    if uid is None:
        return AuthResponse(success=False, message="用户名已存在")
    return AuthResponse(success=True, message="注册成功", user_id=uid)


@router.post("/login")
def login(req: LoginRequest):
    """用户登录接口"""
    if not DB_AVAILABLE:
        # 数据库不可用时，直接返回成功（无状态模式）
        return AuthResponse(success=True, message="数据库不可用，跳过登录", user_id=0)
    pw_hash = hashlib.sha256(req.password.encode()).hexdigest()
    user = login_user(req.username, pw_hash)
    if user is None:
        return AuthResponse(success=False, message="用户名或密码错误")
    return AuthResponse(success=True, message="登录成功", user_id=user[0])


@router.post("/chat")
def chat(req: ChatRequest):
    """
    【核心接口】金融知识问答
    整个RAG流程在这里完成：
    1. 混合检索（向量搜索 + jieba分词搜索，结果融合）
    2. 如果知识库没找到，尝试查数据库
    3. 构造prompt，调用大模型生成回答
    4. 保存对话历史
    """
    # ========== 第1步：混合检索 ==========
    sources = []       # 检索到的知识来源列表
    db_result = None   # 数据库查询结果

    if req.use_knowledge:
        all_results = {}  # 用字典去重：question -> {answer, category, score, source}

        # 1a. 向量搜索：用Embedding模型把问题转成向量，在Milvus中找最相似的问答对
        try:
            from milvus_search import vector_search
            vec_matches = vector_search(req.message, top_k=10, score_threshold=0.5)
            if vec_matches:
                for m in vec_matches:
                    q = m["question"]
                    # 如果同一问题已有更高分的结果，跳过
                    if q not in all_results or m.get("score", 0) > all_results[q].get("score", 0):
                        all_results[q] = {
                            "question": q,
                            "answer": m["answer"],
                            "category": m.get("category", ""),
                            "score": m.get("score", 0) * 1.0,  # 向量相似度得分
                        }
                logger.info(f"向量搜索命中 {len(vec_matches)} 条")
        except Exception as ve:
            logger.warning(f"向量搜索失败: {ve}")

        # 1b. jieba分词搜索：用分词匹配的方式在知识库中搜索（与向量搜索并行）
        try:
            kb_items = get_all_items()  # 获取全部5447条知识
            jieba_matches = search_knowledge_base(req.message, kb_items, top_k=10)
            if jieba_matches:
                for m in jieba_matches:
                    q = m["q"]
                    if q in all_results:
                        # 融合：如果向量搜索也命中了，给它加0.3分（双重验证更可靠）
                        all_results[q]["score"] += 0.3
                        all_results[q]["source_jieba"] = True
                    else:
                        # 只有jieba命中，基础分0.5
                        all_results[q] = {
                            "question": q,
                            "answer": m["a"],
                            "category": m.get("category", ""),
                            "score": 0.5,
                            "source_jieba": True,
                        }
                logger.info(f"jieba搜索命中 {len(jieba_matches)} 条")
        except Exception as je:
            logger.warning(f"jieba搜索失败: {je}")

        # 1c. 融合排序：按总分从高到低排序，取前5条
        if all_results:
            sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
            sources = sorted_results[:5]
            logger.info(f"混合检索共 {len(all_results)} 条，选取 top{len(sources)}")

    # ========== 第2步：如果知识库没找到，试试查数据库 ==========
    # 检测问题是否包含"基金"、"股票"、"持仓"等关键词，如果是则尝试用SQL查数据库
    if not sources or any(kw in req.message for kw in ["基金", "股票", "持仓", "净值", "行情", "代码", "重仓", "查询", "查", "份额", "费率", "规模", "行业"]):
        try:
            from db_query import query_db, get_llm_generated_sql, format_db_results
            # 让LLM根据用户问题生成SQL查询语句
            sql, err = get_llm_generated_sql(req.message, None)
            if sql:
                rows, qerr = query_db(sql)
                if rows:
                    db_result = format_db_results(rows, req.message)
                    sources.append({"question": req.message[:50], "answer": db_result[:100], "category": "数据库查询"})
        except Exception as e:
            logger.warning(f"DB查询失败: {e}")

    # ========== 第3步：构造提示词，调用大模型 ==========
    # 系统提示词：告诉大模型的角色和行为规范
    messages = [{"role": "system", "content": FINANCIAL_SYSTEM_PROMPT}]

    # 加上历史对话（让大模型有上下文记忆，支持多轮对话）
    history = _get_history(req.user_id, limit=4)
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})

    # 把检索到的知识片段拼接成上下文
    context_parts = []
    if sources:
        for src in sources:
            if src.get("category") == "数据库查询":
                context_parts.append(f"【数据库查询结果】{src['answer']}")
            else:
                context_parts.append(f"【{src['category']}】Q: {src['question']}\n   A: {src['answer']}")
        # 拼接格式："以下是与用户问题相关的信息：...请基于以上信息回答"
        context = "以下是与用户问题相关的信息：\n\n" + "\n".join(context_parts)
        context += "\n\n请基于以上信息回答用户的问题。"
        messages.append({"role": "user", "content": f"{context}\n\n用户问题：{req.message}"})
    else:
        # 没有检索到相关知识，直接让大模型回答
        messages.append({"role": "user", "content": f"请回答：{req.message}"})

    # 调用大模型（DeepSeek API），生成最终回答
    try:
        answer = call_llm(messages)
    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        # 降级策略：LLM失败时，直接返回数据库结果或知识库答案
        if db_result:
            answer = db_result
        elif sources:
            answer = sources[0]["answer"]
        else:
            answer = "抱歉，暂时无法回答这个问题。"

    # ========== 第4步：清理返回数据 ==========
    clean_sources = None
    if sources:
        clean_sources = []
        for src in sources:
            clean = {}
            for k, v in src.items():
                if k not in ("score", "source_jieba"):  # 去掉内部评分字段
                    clean[k] = str(v) if not isinstance(v, str) else v
            clean_sources.append(clean)

    # ========== 第5步：保存对话历史 ==========
    # 同时保存到内存缓存和MySQL数据库（如果可用）
    _add_to_history(req.user_id, "user", req.message)
    _add_to_history(req.user_id, "assistant", answer)
    if DB_AVAILABLE and req.user_id:
        save_chat(req.user_id, "user", req.message)
        save_chat(req.user_id, "assistant", answer)

    return ChatResponse(answer=answer, sources=clean_sources)


@router.get("/knowledge/stats")
def knowledge_stats():
    """知识库统计接口：返回各分类的知识条目数量"""
    return get_statistics()


@router.get("/knowledge/{category}")
def get_category_knowledge(category: str):
    """获取某个分类下的所有知识条目"""
    from knowledge_base import get_by_category
    items = get_by_category(category)
    if not items:
        raise HTTPException(status_code=404, detail=f"分类 '{category}' 不存在")
    return {"category": category, "count": len(items), "items": items}
