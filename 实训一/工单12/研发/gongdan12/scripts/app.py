"""Flask 启动入口 -- HTTPS + SSE 流式 + 文件上传 + 历史检索。"""
import os
import sys
import io
import json
import ssl
import ipaddress
import time

# =====================================================================
# 强制 UTF-8 编码（修复 Windows GBK 乱码问题）
# =====================================================================
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 将项目根目录加入 sys.path，确保所有模块可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================================
# 导入 Flask 及核心组件
# =====================================================================
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
from core.config import Config
from core.vector_store import VectorStore, QuestionStore
from core.document_processor import process_documents
from core.retriever import HybridRetriever
from core.llm import LLMClient
from core.reranker import Reranker
from core.rag import RAGPipeline
from core.memory_store import RedisMemoryStore
from core.auth import JWTAuth
from core.content_filter import ContentFilter
from core.graph_builder import GraphBuilder
from core.graph_store import Neo4jGraphStore
from core.mysql_store import MySQLStore

print("[APP] ================================")
print("[APP] RAG 服务初始化开始")
print("[APP] ================================")

# =====================================================================
# Flask 应用基础配置
# =====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB 文件上传上限

# =====================================================================
# 加载全局配置
# =====================================================================
print("[APP] 正在加载 Config ...")
config = Config()
print("[APP] Config 加载完成，开始输出配置详情:")
config.log_config()

# =====================================================================
# 初始化鉴权、屏蔽词过滤、持久化组件（可选）
# =====================================================================
auth = JWTAuth(
    enabled=config.ENABLE_JWT,
    secret=config.JWT_SECRET,
    expire_hours=config.JWT_EXPIRE_HOURS,
    username=getattr(config, "JWT_USERNAME", "admin"),
    password=getattr(config, "JWT_PASSWORD", ""),
)
content_filter = ContentFilter(
    enabled=config.ENABLE_FILTER,
    words=config.FILTER_WORDS,
    response=getattr(config, "FILTER_RESPONSE", "问题包含屏蔽词，已拦截。"),
)
print(f"[APP] JWT 鉴权 enabled={config.ENABLE_JWT}")
print(f"[APP] 屏蔽词过滤 enabled={config.ENABLE_FILTER}, words={len(config.FILTER_WORDS)}")

# =====================================================================
# 初始化向量存储 (VectorStore)
# =====================================================================
print("[APP] 正在初始化 VectorStore ...")
vector_store = VectorStore(
    milvus_host=config.MILVUS_HOST,
    milvus_port=config.MILVUS_PORT,
    collection_name=config.MILVUS_COLLECTION,
    embedding_model_path=config.EMBEDDING_MODEL_PATH,
    embedding_batch_size=config.EMBEDDING_BATCH_SIZE,
    insert_batch_size=config.MILVUS_INSERT_BATCH_SIZE,
)
print(f"[APP] VectorStore 初始化完成, is_ready={vector_store.is_ready}")

# 根据索引是否就绪决定加载策略
if not vector_store.is_ready:
    print("[APP] 索引尚未就绪，请先运行: python scripts/build_index.py")
    chunks = []
    print("[APP] chunks 列表为空 (索引未就绪)")
else:
    print("[APP] 索引已就绪，正在加载索引数据 ...")
    # FAISS 用 _load_index(), Milvus 用 load_chunks()
    if hasattr(vector_store, "_load_index"):
        vector_store._load_index()
        print("[APP] 已通过 _load_index() 加载 FAISS 索引")
    elif hasattr(vector_store, "load_chunks"):
        vector_store.load_chunks()
        print("[APP] 已通过 load_chunks() 加载 Milvus 索引")
    chunks = vector_store._chunks if hasattr(vector_store, "_chunks") else []
    print(f"[APP] chunks 加载完成, 共 {len(chunks)} 个文档块")

# =====================================================================
# 初始化历史问题存储 (QuestionStore) -- 使用 Milvus
# =====================================================================
print("[APP] 正在初始化 QuestionStore ...")
question_store = QuestionStore(
    milvus_host=config.MILVUS_HOST,
    milvus_port=config.MILVUS_PORT,
    collection_name="qa_history",
    embedding_model_path=config.EMBEDDING_MODEL_PATH,
)
print("[APP] QuestionStore 初始化完成")

# =====================================================================
# 初始化 Redis 短期记忆（可选）
# =====================================================================
memory_store = None
if config.ENABLE_REDIS:
    print("[APP] 正在初始化 Redis 短期记忆 ...")
    memory_store = RedisMemoryStore(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        ttl_seconds=getattr(config, "REDIS_MEMORY_TTL_SECONDS", 604800),
    )
    print(f"[APP] Redis 短期记忆初始化完成, ready={memory_store.is_ready}")
else:
    print("[APP] Redis 短期记忆未启用，使用进程内多轮历史")

# =====================================================================
# 初始化 MySQL 持久化（可选）
# =====================================================================
mysql_store = None
if config.ENABLE_MYSQL:
    print("[APP] 正在初始化 MySQL 持久化 ...")
    mysql_store = MySQLStore(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        database=config.MYSQL_DATABASE,
    )
    print(f"[APP] MySQL 持久化初始化完成, ready={mysql_store.is_ready}")
else:
    print("[APP] MySQL 持久化未启用")

# =====================================================================
# 初始化 Neo4j 图谱增强（可选）
# =====================================================================
graph_store = None
graph_builder = None
if config.ENABLE_NEO4J:
    print("[APP] 正在初始化 Neo4j 图谱增强 ...")
    graph_store = Neo4jGraphStore(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
    )
    if graph_store and not graph_store.is_ready:
        graph_store = None
    if graph_store:
        graph_builder = GraphBuilder(graph_store)
    print(f"[APP] Neo4j 图谱增强初始化完成, ready={bool(graph_store)}")
else:
    print("[APP] Neo4j 图谱增强未启用")

# =====================================================================
# 初始化混合检索器 (HybridRetriever)，内含 BM25 模块
# =====================================================================
print("[APP] 正在初始化 HybridRetriever (含 BM25) ...")
retriever = HybridRetriever(vector_store, chunks, config, graph_store=graph_store)
print("[APP] HybridRetriever / BM25 初始化完成")

# =====================================================================
# 初始化 LLM 客户端
# =====================================================================
print("[APP] 正在初始化 LLM 客户端 ...")
llm = LLMClient(config.LLM_API_KEY, config.LLM_API_BASE, config.LLM_MODEL, config.LLM_TIMEOUT)
print(f"[APP] LLM 初始化完成, api_base={config.LLM_API_BASE}, model={config.LLM_MODEL}")

# =====================================================================
# 初始化重排器 (Reranker) -- 根据配置决定是否启用
# =====================================================================
reranker = None
if config.ENABLE_RERANKER and config.RERANKER_MODEL_PATH:
    print("[APP] 正在初始化 Reranker ...")
    reranker = Reranker(config.RERANKER_MODEL_PATH)
    print(f"[APP] Reranker 初始化完成, enabled={config.ENABLE_RERANKER}, path={config.RERANKER_MODEL_PATH}")
else:
    print(f"[APP] Reranker 未启用, enabled={config.ENABLE_RERANKER}, path={config.RERANKER_MODEL_PATH}")

# =====================================================================
# 预加载模型（消除首次查询延迟）
# =====================================================================
import time as _time
print("[APP] 正在预加载 Embedding 模型 ...")
_warmup_start = _time.time()
try:
    vector_store._get_embedder()
    print(f"[APP] Embedding 模型预加载完成, 耗时={_time.time() - _warmup_start:.1f}s")
except Exception as e:
    print(f"[APP] Embedding 模型预加载失败: {e}")

if reranker:
    print("[APP] 正在预加载 Reranker 模型 ...")
    _warmup_start = _time.time()
    try:
        reranker._get_model()
        print(f"[APP] Reranker 模型预加载完成, 耗时={_time.time() - _warmup_start:.1f}s")
    except Exception as e:
        print(f"[APP] Reranker 模型预加载失败: {e}")

# =====================================================================
# 初始化视觉 LLM（图片解析用）
# =====================================================================
vision_llm = None
print("[APP] 正在检查视觉模型 (Vision LLM) 配置 ...")
if config.VISION_API_KEY and config.VISION_API_BASE:
    print("[APP] 检测到视觉模型配置，正在初始化 Vision LLM ...")
    vision_llm = LLMClient(
        config.VISION_API_KEY,
        config.VISION_API_BASE,
        config.VISION_MODEL or config.LLM_MODEL,
        60,
    )
    print(f"[APP] Vision LLM 初始化完成, api_base={config.VISION_API_BASE}, model={config.VISION_MODEL or config.LLM_MODEL}")
else:
    print("[APP] 未配置视觉模型 (VISION_API_KEY 或 VISION_API_BASE 为空)，跳过 Vision LLM 初始化")

# =====================================================================
# 初始化 RAG Pipeline
# =====================================================================
print("[APP] 正在初始化 RAG Pipeline ...")
rag = RAGPipeline(retriever, llm, config, reranker, question_store=question_store, memory_store=memory_store)
print("[APP] RAG Pipeline 初始化完成")

print("[APP] ================================")
print("[APP] 所有组件初始化完成，服务就绪")
print("[APP] ================================")


# =====================================================================
# 路由定义
# =====================================================================


def _check_blocked_question(question: str):
    blocked, matched_words = content_filter.check(question)
    if blocked:
        print(f"[APP] 屏蔽词命中: {matched_words}")
    return blocked, matched_words


def _blocked_json_response(matched_words):
    return jsonify({
        "answer": content_filter.response,
        "sources": [],
        "blocked": True,
        "matched_words": matched_words,
    })


def _blocked_stream_response(start_time, route_name: str, matched_words):
    def generate():
        yield f"data: {json.dumps({'text': content_filter.response}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        elapsed = time.time() - start_time
        print(f"[APP] [{route_name}] 屏蔽词拦截完成, matched={matched_words}, 耗时={elapsed:.2f}s")

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _save_persistent_chat(session_id: str, question: str, answer: str, sources=None):
    if mysql_store and getattr(mysql_store, "is_ready", False) and question and answer:
        mysql_store.save_chat(session_id, question, answer, sources or [])


@app.route("/")
def welcome():
    """欢迎引导页 -- 首次访问时展示引导，若模板不存在则回退到主聊天页。"""
    print("[APP] [GET /] 收到请求")
    try:
        result = render_template("welcome.html")
        print("[APP] [GET /] 返回 welcome.html")
        return result
    except Exception:
        print("[APP] [GET /] welcome.html 不存在，回退到 index.html")
        return render_template("index.html")


@app.route("/chat")
def chat_page():
    """主聊天页面。"""
    print("[APP] [GET /chat] 收到请求")
    try:
        result = render_template("index.html")
        print("[APP] [GET /chat] 返回 index.html")
        return result
    except Exception:
        print("[APP] [GET /chat] 模板不可用，返回 JSON 提示")
        return jsonify({"message": "RAG 服务运行中"})


@app.route("/api/login", methods=["POST"])
def api_login():
    """JWT 登录接口 -- ENABLE_JWT=1 时用于换取访问令牌。"""
    print("[APP] [POST /api/login] 收到请求")
    if not config.ENABLE_JWT:
        return jsonify({"enabled": False, "message": "JWT 鉴权未启用"})

    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not auth.validate_login(username, password):
        print(f"[APP] [POST /api/login] 登录失败: username={username}")
        return jsonify({"error": "用户名或密码错误"}), 401

    token = auth.create_token(username)
    print(f"[APP] [POST /api/login] 登录成功: username={username}")
    return jsonify({
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": config.JWT_EXPIRE_HOURS * 3600,
    })


@app.route("/api/query", methods=["POST"])
@auth.require_auth
def api_query():
    """文本查询接口 -- 接收用户问题，返回 RAG 回答及引用来源。"""
    start_time = time.time()
    print("[APP] [POST /api/query] 收到请求")
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    session_id = data.get("chat_id", "").strip() or request.remote_addr or "default"
    use_history = data.get("use_history", True)
    if not question:
        print("[APP] [POST /api/query] 问题为空，返回 400")
        return jsonify({"error": "问题不能为空"}), 400
    blocked, matched_words = _check_blocked_question(question)
    if blocked:
        return _blocked_json_response(matched_words)

    print(f"[APP] [POST /api/query] 问题: {question[:80]}, session_id={session_id}, use_history={use_history}")
    answer = rag.query(question, use_history=use_history, session_id=session_id)
    sources = rag.get_sources(question)
    _save_persistent_chat(session_id, question, answer, sources)
    elapsed = time.time() - start_time
    print(f"[APP] [POST /api/query] 完成, 耗时={elapsed:.2f}s, answer长度={len(answer)}")
    return jsonify({"answer": answer, "sources": sources})


@app.route("/api/memory/delete", methods=["POST"])
@auth.require_auth
def api_memory_delete():
    """Delete the cached answer for one exact question in the current chat."""
    print("[APP] [POST /api/memory/delete] 收到请求")
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    session_id = data.get("chat_id", "").strip() or request.remote_addr or "default"
    if not question:
        return jsonify({"error": "question 不能为空"}), 400

    stats = rag.forget_question(session_id, question)
    return jsonify({
        "message": "已删除该问题的缓存记录",
        "chat_id": session_id,
        "question": question,
        "deleted": stats,
    })


@app.route("/api/query-stream", methods=["GET"])
@auth.require_auth
def api_query_stream():
    """SSE 流式查询接口 -- 通过 Server-Sent Events 逐块返回回答。"""
    start_time = time.time()
    print("[APP] [GET /api/query-stream] 收到请求")
    question = request.args.get("q", "").strip()
    session_id = request.args.get("chat_id", "").strip() or request.remote_addr or "default"
    if not question:
        print("[APP] [GET /api/query-stream] 问题为空，返回 400")
        return jsonify({"error": "问题不能为空"}), 400
    blocked, matched_words = _check_blocked_question(question)
    if blocked:
        return _blocked_stream_response(start_time, "GET /api/query-stream", matched_words)

    print(f"[APP] [GET /api/query-stream] 问题: {question[:80]}, session_id={session_id}")

    def generate():
        chunk_count = 0
        answer_parts = []
        for chunk_text in rag.query(question, use_history=True, stream=True, session_id=session_id):
            chunk_count += 1
            answer_parts.append(chunk_text)
            yield f"data: {json.dumps({'text': chunk_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        answer = "".join(answer_parts)
        sources = rag.get_sources(question)
        _save_persistent_chat(session_id, question, answer, sources)
        elapsed = time.time() - start_time
        print(f"[APP] [GET /api/query-stream] 流式输出完成, 共 {chunk_count} 块, 耗时={elapsed:.2f}s")

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/query-stream-multi", methods=["GET"])
@auth.require_auth
def api_query_stream_multi():
    """SSE 流式多轮对话查询 -- 支持上下文历史的流式回答。"""
    start_time = time.time()
    print("[APP] [GET /api/query-stream-multi] 收到请求")
    question = request.args.get("q", "").strip()
    session_id = request.args.get("chat_id", "").strip() or request.remote_addr or "default"
    if not question:
        print("[APP] [GET /api/query-stream-multi] 问题为空，返回 400")
        return jsonify({"error": "问题不能为空"}), 400
    blocked, matched_words = _check_blocked_question(question)
    if blocked:
        return _blocked_stream_response(start_time, "GET /api/query-stream-multi", matched_words)

    print(f"[APP] [GET /api/query-stream-multi] 问题: {question[:80]}, session_id={session_id}")

    def generate():
        chunk_count = 0
        answer_parts = []
        for chunk_text in rag.query(question, use_history=True, stream=True, session_id=session_id):
            chunk_count += 1
            answer_parts.append(chunk_text)
            yield f"data: {json.dumps({'text': chunk_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        answer = "".join(answer_parts)
        sources = rag.get_sources(question)
        _save_persistent_chat(session_id, question, answer, sources)
        elapsed = time.time() - start_time
        print(f"[APP] [GET /api/query-stream-multi] 流式输出完成, 共 {chunk_count} 块, 耗时={elapsed:.2f}s")

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/query-image", methods=["POST"])
@auth.require_auth
def api_query_image():
    """图片+文字查询接口 -- 支持上传图片并结合文字提问。"""
    start_time = time.time()
    print("[APP] [POST /api/query-image] 收到请求")
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    image_b64 = data.get("image", "")
    session_id = data.get("chat_id", "").strip() or request.remote_addr or "default"

    if not question and not image_b64:
        print("[APP] [POST /api/query-image] 问题和图片均为空，返回 400")
        return jsonify({"error": "请提供问题或图片"}), 400
    if question:
        blocked, matched_words = _check_blocked_question(question)
        if blocked:
            return _blocked_json_response(matched_words)

    print(f"[APP] [POST /api/query-image] question={question[:60] if question else '(空)'}, has_image={bool(image_b64)}, session_id={session_id}")

    if image_b64:
        # 有图片时使用视觉 LLM（或普通 LLM 兜底）
        vision_prompt = question or "请详细描述这张图片的内容"
        vllm = vision_llm or llm
        using = "Vision LLM" if vision_llm else "fallback LLM"
        print(f"[APP] [POST /api/query-image] 使用 {using} 处理图片")
        history = []
        if config.ENABLE_MULTI_TURN:
            history = rag._load_conversation_history(session_id)
            print(f"[APP] [POST /api/query-image] 加载 Redis/内存历史 {len(history)} 条")
        answer = vllm.chat(
            "请根据以下图片回答问题：" + vision_prompt,
            image_base64=image_b64,
            history=history,
        )
        if config.ENABLE_MULTI_TURN:
            rag._append_conversation_history(session_id, question or "识别图片", answer)
    else:
        # 纯文字查询，走 RAG 流程
        print("[APP] [POST /api/query-image] 无图片，走 RAG 流程")
        answer = rag.query(question, use_history=True, session_id=session_id)

    sources = rag.get_sources(question) if question else []
    _save_persistent_chat(session_id, question or "识别图片", answer, sources)
    elapsed = time.time() - start_time
    print(f"[APP] [POST /api/query-image] 完成, 耗时={elapsed:.2f}s, answer长度={len(answer)}")
    return jsonify({"answer": answer, "sources": sources})


@app.route("/api/upload", methods=["POST"])
@auth.require_auth
def api_upload():
    """文件上传接口 -- 上传后自动增量更新索引。"""
    start_time = time.time()
    print("[APP] [POST /api/upload] 收到上传请求")

    if "files" not in request.files:
        print("[APP] [POST /api/upload] 请求中无 files 字段，返回 400")
        return jsonify({"error": "没有文件"}), 400

    files = request.files.getlist("files")
    saved = []
    for f in files:
        if f.filename:
            filename = secure_filename(f.filename)
            save_path = os.path.join(config.DOCS_DIR, filename)
            os.makedirs(config.DOCS_DIR, exist_ok=True)
            f.save(save_path)
            saved.append(filename)
            print(f"[APP] [POST /api/upload] 文件已保存: {filename}")

    if not saved:
        print("[APP] [POST /api/upload] 无有效文件，返回 400")
        return jsonify({"error": "没有有效文件"}), 400

    print(f"[APP] [POST /api/upload] 共保存 {len(saved)} 个文件，开始增量更新索引 ...")

    # 增量更新索引
    try:
        new_chunks = process_documents(
            config.DOCS_DIR,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP,
            max_workers=config.DOC_PROCESS_WORKERS,
        )
        print(f"[APP] [POST /api/upload] 文档处理完成, 新增 {len(new_chunks)} 个文档块")
        vector_store.build_index_incremental(new_chunks)
        if graph_builder and graph_store and graph_store.is_ready:
            graph_stats = graph_builder.build_from_chunks(new_chunks, clear=False)
            print(f"[APP] [POST /api/upload] Neo4j 图谱增量更新完成: {graph_stats}")
        else:
            print("[APP] [POST /api/upload] Neo4j 未启用或未就绪，跳过图谱增量更新")
        # 更新 retriever 的 BM25 模块
        retriever.bm25 = HybridRetriever(
            vector_store,
            vector_store._chunks,
            config,
            graph_store=graph_store,
        ).bm25 if vector_store._chunks else None
        elapsed = time.time() - start_time
        print(f"[APP] [POST /api/upload] 索引增量更新成功, 耗时={elapsed:.2f}s")
    except Exception as e:
        print(f"[APP] [POST /api/upload] 索引增量更新失败: {e}")
        return jsonify({"error": f"索引更新失败: {e}", "saved": saved}), 500

    elapsed = time.time() - start_time
    print(f"[APP] [POST /api/upload] 全部完成, 耗时={elapsed:.2f}s")
    return jsonify({"message": f"上传成功，已索引 {len(saved)} 个文件", "files": saved})


@app.route("/api/documents")
@auth.require_auth
def api_documents():
    """获取已上传的文档列表 -- 返回文件名和大小。"""
    print("[APP] [GET /api/documents] 收到请求")
    docs = []
    if os.path.exists(config.DOCS_DIR):
        for f in sorted(os.listdir(config.DOCS_DIR)):
            if f.lower().endswith((".pdf", ".docx", ".doc", ".txt")):
                size = os.path.getsize(os.path.join(config.DOCS_DIR, f))
                docs.append({"name": f, "size": f"{size / 1024:.1f}KB"})
    print(f"[APP] [GET /api/documents] 返回 {len(docs)} 个文档")
    return jsonify({"documents": docs})


@app.route("/api/suggestions")
@auth.require_auth
def api_suggestions():
    """返回快捷提问推荐列表。"""
    print("[APP] [GET /api/suggestions] 收到请求")
    lang = (request.args.get("lang") or "zh").lower()
    if lang.startswith("en"):
        suggestions = [
            "What is the main topic of this document?",
            "Which key data points are mentioned?",
            "Please summarize the core takeaways",
            "What important time points are included?",
            "What are the risk factors in this document?",
        ]
    else:
        suggestions = [
            "这份文档的主要内容是什么？",
            "文档中提到了哪些关键数据？",
            "请总结一下文档的核心要点",
            "文档中有哪些重要的时间节点？",
            "这份文档的风险因素有哪些？",
        ]
    print(f"[APP] [GET /api/suggestions] 返回 {len(suggestions)} 条推荐")
    return jsonify({"suggestions": suggestions, "lang": lang})


@app.route("/api/health")
def health():
    """健康检查接口 -- 返回服务状态和索引就绪情况。"""
    print("[APP] [GET /api/health] 收到请求")
    ready = vector_store.is_ready
    print(f"[APP] [GET /api/health] status=ok, index_ready={ready}")
    return jsonify({"status": "ok", "index_ready": ready})


# =====================================================================
# HTTPS 支持 -- 自动生成自签名证书
# =====================================================================

def create_ssl_cert():
    """自动生成自签名 SSL 证书（若已存在则复用）。"""
    cert_dir = os.path.join(BASE_DIR, "certs")
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("[APP] SSL 证书已存在，复用现有证书")
        return cert_file, key_file

    os.makedirs(cert_dir, exist_ok=True)
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime

        # 生成 RSA 私钥
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
        # 签发证书，有效期一年
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]), critical=False)
            .sign(key, hashes.SHA256())
        )

        # 写入私钥和证书文件
        with open(key_file, "wb") as f:
            f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        print(f"[APP] SSL 自签名证书已生成: {cert_dir}")
        return cert_file, key_file
    except ImportError:
        print("[APP] 需要安装 cryptography: pip install cryptography")
        return None, None


# =====================================================================
# 主入口 -- 根据 ENABLE_SSL 环境变量决定 HTTP/HTTPS 模式
# =====================================================================
if __name__ == "__main__":
    host = config.HOST
    port = config.PORT

    print("[APP] ================================")
    print(f"[APP] 即将启动服务, host={host}, port={port}")
    print("[APP] ================================")

    if os.getenv("ENABLE_SSL", "0") == "1":
        cert_file, key_file = create_ssl_cert()
        if cert_file and key_file:
            print(f"[APP] HTTPS 模式启动: https://{host}:{port}")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert_file, key_file)
            app.run(host=host, port=port, debug=config.DEBUG, ssl_context=context)
        else:
            print("[APP] SSL 证书生成失败，回退到 HTTP 模式")
            print(f"[APP] HTTP 模式启动: http://{host}:{port}")
            app.run(host=host, port=port, debug=config.DEBUG)
    else:
        print(f"[APP] HTTP 模式启动: http://{host}:{port}")
        app.run(host=host, port=port, debug=config.DEBUG)
