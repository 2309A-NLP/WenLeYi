# -*- coding: utf-8 -*-
"""
数据库操作模块：MySQL、Milvus、Redis 数据库连接和数据操作

本文件实现的核心功能：

【MySQL 操作】
1. 用户管理：注册、登录、获取用户信息
2. 角色管理：获取所有角色、获取角色提示词、获取角色名称
3. 角色本地知识库：检索 SQLite 角色知识表（兜底检索）
4. 兜底回答生成：基于本地命中内容直接组织自然语言回答

【Milvus 向量数据库操作】
1. 角色知识库集合：为每个角色创建/确保独立的 collection（financial_knowledge_role_{id}）
2. 混合检索：稠密向量（BGE-M3）+ 稀疏向量（BM25），RRF 融合
3. 重排序：调用 BGE-reranker 模型精排检索结果
4. 长期记忆：向量化的长期记忆存储与检索（按 user_id + role_id 过滤）
5. 股票数据：将东方财富股票数据向量化后存入 Milvus，并提供检索接口
6. 文档入库：文本分块、生成稠密/稀疏向量、插入 Milvus、自动重建 BM25 模型

【Redis 操作】
1. 短期记忆：聊天历史的存取与清空（基于 Redis 或内存降级）

【股票数据源】
1. 从东方财富（AKShare）实时获取全量股票数据
2. 本地 JSON 缓存降级检索（当 Milvus 不可用时）

【辅助功能】
- Milvus 连接诊断（网络检查、认证排查）
- 自动重试与降级机制（支持本地模式）
- 角色知识库自动种子入库（从 data/ 中的 CSV/TXT 读取）
"""

import os
import logging
import sys
import threading
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# 添加当前目录到路径，支持直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logger = logging.getLogger(__name__)

# 导入 utils 中的配置和函数
try:
    from utils import (
        MYSQL_CONFIG, MILVUS_HOST, MILVUS_PORT, MILVUS_USER, MILVUS_PASSWORD,
        MILVUS_USE_SSL, MILVUS_TIMEOUT, MILVUS_ENABLE_LOCAL, COLLECTION_PREFIX,
        EMBEDDING_DIM, TOP_K_RETRIEVE, TOP_K_LONG_TERM, TOP_K_STOCK, MEMORY_LENGTH,
        ROLE_PROMPT_TEMPLATES, ROLE_DISPLAY_NAMES, ROLE_DEFAULT_DESCRIPTIONS, ROLE_SQLITE_KNOWLEDGE,
        STOCK_COLLECTION_NAME, _MILVUS_CONNECTION_ALIAS,
        hash_password, get_embedding_model, get_reranker_model, get_redis_client, milvus_ready, set_milvus_available,
        get_bm25_models, set_bm25_model, _find_role_dataset_files, _load_texts_from_dataset,
        simple_chinese_tokenizer, _load_local_stock_cache
    )
except ImportError:
    from .utils import (
        MYSQL_CONFIG, MILVUS_HOST, MILVUS_PORT, MILVUS_USER, MILVUS_PASSWORD,
        MILVUS_USE_SSL, MILVUS_TIMEOUT, MILVUS_ENABLE_LOCAL, COLLECTION_PREFIX,
        EMBEDDING_DIM, TOP_K_RETRIEVE, TOP_K_LONG_TERM, TOP_K_STOCK, MEMORY_LENGTH,
        ROLE_PROMPT_TEMPLATES, ROLE_DISPLAY_NAMES, ROLE_DEFAULT_DESCRIPTIONS, ROLE_SQLITE_KNOWLEDGE,
        STOCK_COLLECTION_NAME, _MILVUS_CONNECTION_ALIAS,
        hash_password, get_embedding_model, get_reranker_model, get_redis_client, milvus_ready, set_milvus_available,
        get_bm25_models, set_bm25_model, _find_role_dataset_files, _load_texts_from_dataset,
        simple_chinese_tokenizer, _load_local_stock_cache
    )


# ==================== MySQL 数据库操作 ====================

def get_db_connection():
    """获取 MySQL 数据库连接（使用 pymysql，配置来自环境变量）"""
    try:
        import pymysql
        conn = pymysql.connect(**MYSQL_CONFIG)
        logger.debug(f"数据库连接成功")
        return conn
    except Exception as e:
        logger.error(f"数据库连接失败: {str(e)}")
        raise


def init_database():
    """
    初始化 MySQL 数据库和所有必需的表。

    创建的表：
    - users: 用户信息（id, username, password_hash, email, phone, created_at）
    - roles: 角色定义（id, role_name, display_name, description, personality, tone,
             background_story, hobbies, prompt_template, creator_user_id, created_at, updated_at）
    - knowledge_docs: 文档元信息（id, doc_id, title, source, content, chunk_count, created_at, updated_at）
    - role_knowledge: 角色本地知识库（role_id, title, content），用于兜底检索

    同时会：
    - 自动添加缺失的列（通过 ALTER TABLE）
    - 插入预定义的 11 个角色（金融理财师、医生、心理医生、虚拟朋友等）
    - 为每个角色填充 ROLE_SQLITE_KNOWLEDGE 中定义的本地知识对
    """
    try:
        import pymysql
        server_conn = pymysql.connect(
            host=MYSQL_CONFIG["host"],
            port=MYSQL_CONFIG["port"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            charset=MYSQL_CONFIG["charset"],
            autocommit=True,
        )
    except pymysql.err.OperationalError as e:
        if getattr(e, "args", None) and len(e.args) >= 2 and e.args[0] == 1045:
            logger.error(
                "MySQL 鉴权失败：请在 .env 中配置 MYSQL_HOST/MYSQL_PORT/MYSQL_USER/MYSQL_PASSWORD/MYSQL_DATABASE。"
            )
        raise
    try:
        with server_conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        server_conn.close()

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # 用户表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS users
                       (
                           id
                           BIGINT
                           PRIMARY
                           KEY
                           AUTO_INCREMENT,
                           username
                           VARCHAR
                       (
                           50
                       ) UNIQUE NOT NULL,
                           password_hash VARCHAR
                       (
                           64
                       ) NOT NULL,
                           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           email VARCHAR
                       (
                           100
                       ),
                           phone VARCHAR
                       (
                           20
                       )
                           ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                       """)

        # 角色表（包含扩展字段 personality, tone, background_story, hobbies 等）
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS roles
                       (
                           id BIGINT PRIMARY KEY AUTO_INCREMENT,
                           role_name VARCHAR
                       (
                           50
                       ) UNIQUE NOT NULL,
                           display_name VARCHAR
                       (
                           80
                       ),
                           description TEXT,
                           personality TEXT,
                           tone TEXT,
                           background_story TEXT,
                           hobbies TEXT,
                           prompt_template TEXT,
                           creator_user_id BIGINT NULL,
                           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                           ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                       """)
        # 兼容旧表结构：添加缺失的列
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'display_name'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN display_name VARCHAR(80) NULL AFTER role_name")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'personality'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN personality TEXT NULL AFTER description")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'tone'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN tone TEXT NULL AFTER personality")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'background_story'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN background_story TEXT NULL AFTER tone")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'hobbies'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN hobbies TEXT NULL AFTER background_story")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'creator_user_id'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE roles ADD COLUMN creator_user_id BIGINT NULL AFTER prompt_template")
        cursor.execute("SHOW COLUMNS FROM roles LIKE 'updated_at'")
        if not cursor.fetchone():
            cursor.execute(
                "ALTER TABLE roles ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")

        # 插入默认角色（如果不存在）
        for role_name, template in ROLE_PROMPT_TEMPLATES.items():
            cursor.execute("SELECT id FROM roles WHERE role_name=%s", (role_name,))
            if not cursor.fetchone():
                display_name = ROLE_DISPLAY_NAMES.get(role_name, role_name)
                description = ROLE_DEFAULT_DESCRIPTIONS.get(role_name, "")
                personality = f"你是{display_name}，擅长基于专业信息给出可执行建议。"
                tone = "专业、耐心、清晰"
                background_story = f"你长期从事{display_name}相关工作，重视事实、逻辑与风险提示。"
                hobbies = "阅读行业报告、案例复盘、知识分享"
                cursor.execute(
                    """
                    INSERT INTO roles
                    (role_name, display_name, description, personality, tone, background_story, hobbies,
                     prompt_template)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (role_name, display_name, description, personality, tone, background_story, hobbies, template)
                )

        # 知识文档元信息表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS knowledge_docs
                       (
                           id
                           BIGINT
                           PRIMARY
                           KEY
                           AUTO_INCREMENT,
                           doc_id
                           VARCHAR
                       (
                           100
                       ) UNIQUE NOT NULL,
                           title VARCHAR
                       (
                           255
                       ),
                           source VARCHAR
                       (
                           255
                       ),
                           content TEXT,
                           chunk_count INTEGER DEFAULT 0,
                           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                           ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                       """)

        # 角色本地知识表（用于 SQLite 风格的兜底检索）
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS role_knowledge
                       (
                           id
                           BIGINT
                           PRIMARY
                           KEY
                           AUTO_INCREMENT,
                           role_id
                           BIGINT
                           NOT
                           NULL,
                           title
                           VARCHAR
                       (
                           255
                       ) NOT NULL,
                           content TEXT NOT NULL,
                           UNIQUE
                       (
                           role_id,
                           title
                       ),
                           FOREIGN KEY
                       (
                           role_id
                       ) REFERENCES roles
                       (
                           id
                       ) ON DELETE CASCADE
                           ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                       """)
        cursor.execute("SHOW COLUMNS FROM role_knowledge LIKE 'title'")
        title_col = cursor.fetchone()
        if title_col:
            title_type = str(title_col[1]).lower()
            if "text" in title_type:
                # 将 TEXT 列改为 VARCHAR，以便支持 UNIQUE 索引
                cursor.execute("ALTER TABLE role_knowledge MODIFY COLUMN title VARCHAR(255) NOT NULL")

        # 同步/更新角色描述和提示词
        for role_name, template in ROLE_PROMPT_TEMPLATES.items():
            display_name = ROLE_DISPLAY_NAMES.get(role_name, role_name)
            description = ROLE_DEFAULT_DESCRIPTIONS.get(role_name, "")
            personality = f"你是{display_name}，擅长基于专业信息给出可执行建议。"
            tone = "专业、耐心、清晰"
            background_story = f"你长期从事{display_name}相关工作，重视事实、逻辑与风险提示。"
            hobbies = "阅读行业报告、案例复盘、知识分享"
            cursor.execute("SELECT id FROM roles WHERE role_name=%s", (role_name,))
            existing = cursor.fetchone()
            if existing:
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
                    WHERE role_name = %s
                    """,
                    (display_name, description, personality, tone, background_story, hobbies, template, role_name),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO roles
                    (role_name, display_name, description, personality, tone, background_story, hobbies,
                     prompt_template)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (role_name, display_name, description, personality, tone, background_story, hobbies, template)
                )

        # 为每个角色插入预定义的本地知识对（ROLE_SQLITE_KNOWLEDGE）
        cursor.execute("SELECT id, role_name FROM roles")
        for role_id, role_name in cursor.fetchall():
            for title, content in ROLE_SQLITE_KNOWLEDGE.get(role_name, []):
                cursor.execute(
                    "INSERT IGNORE INTO role_knowledge (role_id, title, content) VALUES (%s, %s, %s)",
                    (role_id, title, content),
                )

        conn.commit()
        logger.info("MySQL 数据库初始化完成")
    finally:
        conn.close()


def register_user(username: str, password: str, email: Optional[str] = None, phone: Optional[str] = None) -> bool:
    """注册新用户，返回是否成功（用户名唯一性校验）"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
        if cursor.fetchone():
            logger.info(f"用户注册失败：用户名已存在: {username}")
            return False

        password_hash = hash_password(password)
        logger.debug(f"注册用户 {username}，密码哈希: {password_hash[:16]}...")

        cursor.execute(
            "INSERT INTO users (username, password_hash, email, phone) VALUES (%s, %s, %s, %s)",
            (username, password_hash, email, phone)
        )
        conn.commit()
        logger.info(f"用户注册成功: {username}, 受影响行数: {cursor.rowcount}")

        cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
        result = cursor.fetchone()
        if result:
            logger.debug(f"注册验证成功，用户ID: {result[0]}")
        else:
            logger.error(f"注册验证失败：用户 {username} 注册后查询不到")

        return True
    finally:
        conn.close()


def login_user(username: str, password: str) -> Optional[int]:
    """用户登录验证，返回 user_id 或 None"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        password_hash = hash_password(password)
        logger.debug(f"登录用户 {username}，密码哈希: {password_hash[:16]}...")

        cursor.execute("SELECT id, password_hash FROM users WHERE username=%s", (username,))
        user_record = cursor.fetchone()

        if not user_record:
            logger.info(f"用户登录失败：用户不存在: {username}")
            return None

        db_user_id, db_password_hash = user_record
        logger.debug(f"数据库中用户ID: {db_user_id}, 存储的密码哈希: {db_password_hash[:16]}...")

        if password_hash == db_password_hash:
            logger.info(f"用户登录成功: {username}, 用户ID: {db_user_id}")
            return db_user_id
        else:
            logger.info(f"用户登录失败：密码错误: {username}")
            return None
    finally:
        conn.close()


def get_user_info(user_id: int) -> Optional[Dict[str, Any]]:
    """获取用户信息（不包含密码哈希）"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username, email, phone, created_at FROM users WHERE id=%s", (user_id,))
        result = cursor.fetchone()
        if result:
            return {
                "user_id": user_id,
                "username": result[0],
                "email": result[1],
                "phone": result[2],
                "created_at": result[3]
            }
        return None
    finally:
        conn.close()


def get_roles() -> List[Dict[str, Any]]:
    """获取所有角色列表（包含 id, role_name, display_name, description, personality, tone, background_story, hobbies, prompt_template）"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
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
                       ORDER BY id
                       """)
        results = cursor.fetchall()
        return [{
            "id": r[0],
            "role_name": r[1],
            "display_name": r[2] or r[1],
            "description": r[3],
            "personality": r[4],
            "tone": r[5],
            "background_story": r[6],
            "hobbies": r[7],
            "prompt_template": r[8],
        } for r in results]
    finally:
        conn.close()


def get_role_prompt(role_id: int) -> Optional[str]:
    """获取角色的提示词模板（优先使用数据库存储的，若无则动态构建）"""
    try:
        from utils import _build_custom_role_prompt
    except ImportError:
        from .utils import _build_custom_role_prompt
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT role_name, display_name, personality, tone, background_story, hobbies, prompt_template
            FROM roles
            WHERE id = %s
            """,
            (role_id,),
        )
        result = cursor.fetchone()
        if not result:
            return None
        role = {
            "role_name": result[0],
            "display_name": result[1],
            "personality": result[2],
            "tone": result[3],
            "background_story": result[4],
            "hobbies": result[5],
            "prompt_template": result[6],
        }
        return role["prompt_template"] or _build_custom_role_prompt(role)
    finally:
        conn.close()


def get_role_name(role_id: int) -> Optional[str]:
    """获取角色的英文名称（如 financial_advisor）"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT role_name FROM roles WHERE id=%s", (role_id,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def retrieve_sqlite_role_knowledge(role_id: int, query: str, top_k: int = 3) -> List[str]:
    """
    从 MySQL 的 role_knowledge 表中检索本地知识（作为兜底检索源）。
    使用简单的词频匹配打分，返回最相关的 top_k 条内容。
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT title, content FROM role_knowledge WHERE role_id=%s", (role_id,))
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    tokens = [t for t in simple_chinese_tokenizer(query) if t]
    scored: List[Any] = []
    for title, content in rows:
        haystack = f"{title}\n{content}".lower()
        score = 0
        for token in tokens:
            if token.lower() in haystack:
                score += 2 if len(token) > 1 else 1
        if score == 0:
            score = 1  # 保底分数，防止全部为0
        scored.append((score, f"{title}：{content}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:top_k]]


def build_database_grounded_answer(
        role_id: int,
        message: str,
        sqlite_knowledge: List[str],
        stock_info: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    当大模型不可用或超时时，直接基于本地检索命中内容组织一版自然的回答。
    根据不同角色（医生、心理医生、教师、律师、理财师等）采用不同的回答模板。
    """
    try:
        from utils import ROLE_DISPLAY_NAMES, normalize_answer_text
    except ImportError:
        from .utils import ROLE_DISPLAY_NAMES, normalize_answer_text

    role_name = get_role_name(role_id) or "assistant"
    role_display = ROLE_DISPLAY_NAMES.get(role_name, role_name)
    stock_info = stock_info or []

    snippets = []
    for item in sqlite_knowledge[:3]:
        snippets.append(item.split("：", 1)[-1].strip())

    stock_lines = []
    for s in stock_info[:2]:
        stock_lines.append(
            f"{s['stock_name']}({s['stock_code']})当前公开数据显示最新价约{s['latest_price']}元，涨跌幅{s['change_pct']}%，市盈率{s['pe_ratio']}。"
        )

    # 医生角色回答模板
    if role_name == "doctor":
        body = "先根据你提到的症状给你一个常见情况的判断：\n" + "\n".join(
            [f"{i + 1}. {x}" for i, x in enumerate(snippets[:3])])
        return normalize_answer_text(
            f"你这个问题我直接按常见情况跟你说。\n\n{body}\n\n"
            "如果只是普通不适，一般先休息、补水、清淡饮食，药物以对症处理为主，不要自己叠加吃多种复方药。"
            " 但如果你同时有高热不退、呼吸困难、胸痛、意识模糊，或者症状越来越重，就不要继续拖，尽快去医院。"
            " 如果你愿意，我也可以继续根据你的具体症状，比如有没有发烧、咳嗽、流鼻涕、咽痛，帮你再细分一下。"
        )

    # 心理医生角色回答模板
    if role_name == "psychologist":
        body = "\n".join([f"{i + 1}. {x}" for i, x in enumerate(snippets[:3])])
        return normalize_answer_text(
            f"我先接住你的感受。很多时候，一个人状态不好并不是因为你不够努力，而是已经累了、压住了太多情绪。\n\n"
            f"结合你现在的话题，我比较建议先做这几件事：\n{body}\n\n"
            "你不用一下子把自己调整到很好，先把状态从最难受拉回一点点就够了。"
            " 如果你愿意，可以直接告诉我是学习压力、关系问题，还是单纯觉得很烦很累，我会按那个方向继续陪你分析。"
        )

    # 虚拟朋友角色回答模板
    if role_name == "virtual_friend":
        core = snippets[0] if snippets else "我在，咱们就顺着你现在想聊的继续。"
        return normalize_answer_text(
            f"{core}\n\n"
            f"你刚刚说的是“{message}”，我更想先听听你现在是随口一问，还是今天真的发生了什么。"
            " 如果你想轻松聊，我就陪你闲聊；如果你今天状态不太好，你也可以直接跟我说最卡在心里的那一件事。"
        )

    # 教师角色回答模板
    if role_name == "teacher":
        body = "\n".join([f"{i + 1}. {x}" for i, x in enumerate(snippets[:3])])
        return normalize_answer_text(
            f"这个问题我给你按“先讲清楚，再举例子”的方式说。\n\n{body}\n\n"
            "如果你愿意，我还可以继续把它拆成更简单的版本，或者直接拿一道题/一句话来给你演示。"
        )

    # 律师角色回答模板
    if role_name == "lawyer":
        body = "\n".join([f"{i + 1}. {x}" for i, x in enumerate(snippets[:3])])
        return normalize_answer_text(
            f"这个问题先按一般处理思路说，不直接下案件结论。\n\n{body}\n\n"
            "如果你要我继续帮你判断，最好补充一下地区、合同/借条/聊天记录/付款记录这些关键信息，我可以按证据和处理步骤继续帮你梳理。"
        )

    # 金融相关角色（股票分析师、投资顾问、理财师等）回答模板
    if role_name in {"stock_analyst", "investment_advisor", "financial_advisor", "financial_planner"}:
        body = "\n".join([f"{i + 1}. {x}" for i, x in enumerate(snippets[:3])])
        extra = ("\n\n补充一点公开市场数据：\n" + "\n".join(stock_lines)) if stock_lines else ""
        return normalize_answer_text(
            f"这个问题可以直接从几个关键点来看：\n{body}{extra}\n\n"
            "如果你愿意，再补充你的目标、期限和风险承受能力，我可以把建议说得更具体一些。"
        )

    # 默认模板
    return normalize_answer_text(
        f"我先结合这个角色的知识库给你一个直接回答：\n\n" +
        "\n".join([f"{i + 1}. {x}" for i, x in enumerate(snippets[:3])]) +
        "\n\n如果你愿意，我可以继续顺着你这句话往下展开。"
    )


# ==================== Milvus 向量数据库操作 ====================

def knowledge_collection_name(role_id: int) -> str:
    """生成角色知识库的 collection 名称（格式：financial_knowledge_role_{role_id}）"""
    return f"{COLLECTION_PREFIX}_{int(role_id)}"


def _knowledge_collection_schema() -> Any:
    """定义角色知识库 collection 的 schema（包含稠密向量和稀疏向量）"""
    from pymilvus import CollectionSchema, FieldSchema, DataType
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="create_time", dtype=DataType.INT64),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
    ]
    return CollectionSchema(fields, "角色隔离金融知识库集合")


def ensure_knowledge_collection(role_id: int) -> Any:
    """确保某个 role_id 的知识库 collection 存在，若不存在则创建并建立索引，最后加载到内存"""
    from pymilvus import Collection, utility, MilvusException

    name = knowledge_collection_name(role_id)
    try:
        if not utility.has_collection(name):
            logger.info(f"📦 创建角色知识库集合: {name}")
            collection = Collection(name, _knowledge_collection_schema())

            logger.debug(f"🔧 创建稠密向量索引: {name}")
            collection.create_index(
                "dense_vector",
                {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}},
            )

            logger.debug(f"🔧 创建稀疏向量索引: {name}")
            collection.create_index(
                "sparse_vector",
                {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
            )

            logger.info(f"✅ 角色知识库集合 {name} 创建完成")
        else:
            logger.debug(f"📋 角色知识库集合 {name} 已存在")
            collection = Collection(name)

        _ensure_collection_loaded(collection, name)
        return collection

    except Exception as e:
        logger.error(f"❌ 确保知识库集合 {name} 失败: {e}")
        raise


def _ensure_collection_loaded(collection: Any, collection_name: str = ""):
    """确保集合已加载到内存（兼容 pymilvus 2.6.x API）"""
    try:
        from pymilvus import utility
        try:
            load_state = utility.load_state(collection_name or collection.name)
            if load_state != utility.LoadState.Loaded:
                logger.info(f"📥 加载集合: {collection_name or collection.name}")
                collection.load()
        except AttributeError:
            # 旧版本没有 load_state 方法，直接加载
            logger.debug(f"📥 加载集合: {collection_name or collection.name}")
            collection.load()
    except Exception as e:
        logger.warning(f"⚠️ 检查/加载集合失败: {e}")
        try:
            collection.load()
        except Exception:
            pass


def diagnose_milvus_connection() -> dict:
    """诊断 Milvus 连接问题，返回详细报告（网络检查、连接测试、版本信息）"""
    result = {
        "success": False,
        "host": MILVUS_HOST,
        "port": MILVUS_PORT,
        "ssl": MILVUS_USE_SSL,
        "timeout": MILVUS_TIMEOUT,
        "errors": [],
        "suggestions": [],
        "server_info": None
    }

    try:
        logger.info("🔍 开始诊断 Milvus 连接...")

        # 1. 检查网络连通性
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            conn_result = sock.connect_ex((MILVUS_HOST, MILVUS_PORT))
            if conn_result == 0:
                result["errors"].append(f"✅ 网络连接正常: {MILVUS_HOST}:{MILVUS_PORT}")
            else:
                result["errors"].append(f"❌ 网络连接失败: 端口 {MILVUS_PORT} 无法访问")
                result["suggestions"].append("请检查 Milvus 服务是否正在运行")
                result["suggestions"].append("请检查防火墙设置，确保端口 19530 已开放")
        finally:
            sock.close()

        # 2. 尝试连接 Milvus 并获取版本
        from pymilvus import connections, utility

        try:
            connections.connect(
                alias="diagnose",
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                timeout=MILVUS_TIMEOUT
            )

            version = utility.get_server_version()
            result["server_info"] = {"version": version}
            result["errors"].append(f"✅ Milvus 连接成功，版本: {version}")

            collections = utility.list_collections()
            result["errors"].append(f"📚 当前集合数量: {len(collections)}")

            connections.disconnect("diagnose")
            result["success"] = True

        except Exception as e:
            error_msg = str(e)
            result["errors"].append(f"❌ Milvus 连接失败: {error_msg}")

            if "connection refused" in error_msg.lower():
                result["suggestions"].append("Milvus 服务未启动或端口错误")
                result["suggestions"].append("使用命令启动: milvus start 或 docker-compose up -d")
            elif "timeout" in error_msg.lower():
                result["suggestions"].append("网络超时，请检查网络连接")
                result["suggestions"].append(f"尝试增加 MILVUS_TIMEOUT 环境变量（当前: {MILVUS_TIMEOUT}s）")
            else:
                result["suggestions"].append(f"错误详情: {error_msg}")

        return result

    except Exception as e:
        result["errors"].append(f"❌ 诊断过程出错: {str(e)}")
        return result


def init_milvus(max_retries: int = 3, retry_delay: int = 2):
    """
    初始化 Milvus 连接和集合。
    支持重试机制、多种连接方式（带/不带 alias），失败后可尝试本地模式。
    成功后自动创建角色知识库集合、长期记忆集合、股票数据集合。
    """
    # 导入 Milvus 连接相关模块：connections（管理连接）、utility（工具函数，如获取版本）、MilvusException（异常类型）
    from pymilvus import connections, utility, MilvusException

    # 记录 Milvus 配置信息到日志，便于排查连接问题
    logger.info(
        f"📋 Milvus 配置信息: host={MILVUS_HOST}, port={MILVUS_PORT}, ssl={MILVUS_USE_SSL}, timeout={MILVUS_TIMEOUT}s")

    # 构建基础连接参数字典，包含别名、主机、端口、超时、gRPC 消息大小限制
    connect_params = {
        "alias": _MILVUS_CONNECTION_ALIAS,                      # 连接别名，用于后续区分不同 Milvus 连接
        "host": MILVUS_HOST,                                    # Milvus 服务地址
        "port": MILVUS_PORT,                                    # Milvus 服务端口
        "timeout": MILVUS_TIMEOUT,                              # 连接超时时间（秒）
        "grpc_max_send_message_size": 1024 * 1024 * 1024,       # gRPC 发送消息最大 1GB，适应大数据量
        "grpc_max_receive_message_size": 1024 * 1024 * 1024,    # gRPC 接收消息最大 1GB
    }

    # 如果配置了用户名，加入认证参数
    if MILVUS_USER:
        connect_params["user"] = MILVUS_USER
        logger.info("🔐 Milvus 认证已启用")
    # 如果配置了密码，加入密码参数
    if MILVUS_PASSWORD:
        connect_params["password"] = MILVUS_PASSWORD

    # 如果启用 SSL，设置 secure 标志，使用加密连接
    if MILVUS_USE_SSL:
        connect_params["secure"] = True
        logger.info("🔒 Milvus SSL连接已启用")

    # 定义多种连接方式列表，以便在一种方式失败时尝试其他方式
    connection_methods = [
        ("标准连接", connect_params),   # 方式1：使用包含 alias 的完整参数
    ]

    # 构造备用连接方式：复制标准参数，但去掉 alias 键（某些 Milvus 版本或配置可能不支持显式 alias）
    backup_params = connect_params.copy()
    backup_params.pop("alias", None)   # 移除 alias 字段
    # 只有当 backup_params 与原始参数不同时才加入备用方式（避免重复）
    if backup_params != connect_params:
        connection_methods.append(("无别名连接", backup_params))

    last_error = None   # 保存最后一次连接失败的异常，用于最终抛出

    # 外层循环：重试机制，最多重试 max_retries 次
    for attempt in range(max_retries):
        # 内层循环：遍历所有连接方式，每种方式尝试一次
        for method_name, params in connection_methods:
            try:
                logger.info(
                    f"🔄 连接 Milvus (第 {attempt + 1}/{max_retries} 次尝试, {method_name}): {MILVUS_HOST}:{MILVUS_PORT}")

                # 尝试断开已有的同别名连接（如果存在），避免冲突或残留连接
                try:
                    connections.disconnect(_MILVUS_CONNECTION_ALIAS)
                except Exception:
                    # 忽略断开失败（可能原本就没有该连接），继续执行
                    pass

                # 根据参数中是否包含 alias 选择不同的连接方式
                if "alias" in params:
                    # 使用带别名的连接（推荐方式）
                    connections.connect(**params)
                else:
                    # 使用不带别名的连接（仅使用 host/port/timeout）
                    connections.connect(
                        host=params["host"],
                        port=params["port"],
                        timeout=params["timeout"],
                    )

                # 连接成功后，获取 Milvus 服务器版本，验证连接是否真正可用
                version = utility.get_server_version()
                logger.info(f"✅ Milvus 连接成功，版本: {version}")

                # 调用内部函数创建/初始化所需的集合（角色知识库、长期记忆、股票数据等）
                _init_milvus_collections()
                # 设置全局标志，表示 Milvus 可用
                set_milvus_available(True)
                return  # 成功完成，退出函数

            except Exception as e:
                # 当前连接方式失败，记录异常
                last_error = e
                error_msg = str(e)
                logger.debug(f"❌ {method_name} 连接失败: {error_msg}")

        # 一次尝试（所有连接方式均失败）结束，输出警告日志
        logger.warning(f"❌ Milvus 连接失败 (尝试 {attempt + 1}/{max_retries})")
        # 调用辅助函数分析错误类型，给出可能的原因提示（如防火墙、认证错误等）
        _analyze_milvus_error(str(last_error), attempt)

        # 如果还没到最大重试次数，等待 retry_delay 秒后继续下一轮尝试
        if attempt < max_retries - 1:
            logger.info(f"⏳ 等待 {retry_delay} 秒后重试...")
            import time
            time.sleep(retry_delay)
        else:
            # 最后一次尝试失败，检查是否启用了本地存储模式（降级方案）
            if MILVUS_ENABLE_LOCAL:
                logger.info("🔧 尝试使用本地存储模式...")
                # 尝试使用本地 Milvus Lite 或文件存储模式，如果成功则正常返回
                if _try_local_milvus():
                    return
            # 本地模式也未启用或失败，最终抛出最后一次捕获的异常
            logger.error(f"❌ Milvus 连接完全失败，错误: {last_error}")
            raise last_error


def _analyze_milvus_error(error_msg: str, attempt: int):
    """分析 Milvus 连接错误并给出建议"""
    if "connection refused" in error_msg.lower() or "fail connecting" in error_msg.lower():
        logger.error("🔍 连接被拒绝！请检查：")
        logger.error("   1. Milvus 服务是否正在运行")
        logger.error("   2. Milvus 主机地址是否正确")
        logger.error("   3. Milvus 端口是否正确 (默认19530)")
        logger.error("   4. 防火墙是否允许访问")
    elif "timeout" in error_msg.lower():
        logger.error("🔍 连接超时！请检查：")
        logger.error("   1. 网络连接是否稳定")
        logger.error("   2. Milvus 服务是否过载")
        logger.error("   3. 考虑增加 MILVUS_TIMEOUT 环境变量")
    elif "authentication" in error_msg.lower() or "invalid token" in error_msg.lower():
        logger.error("🔍 认证失败！请检查：")
        logger.error("   1. MILVUS_USER 环境变量是否正确")
        logger.error("   2. MILVUS_PASSWORD 环境变量是否正确")
    elif attempt == 0:
        logger.info("💡 提示：如果使用 Milvus Standalone，确保已启动服务：")
        logger.info("   - Linux: ./milvus start")
        logger.info("   - Docker: docker-compose up -d")

def _try_local_milvus():
    """尝试使用本地模式连接 Milvus（用于调试或降级）"""
    try:
        # 导入 Milvus 连接和工具函数
        from pymilvus import connections, utility
        logger.info("🔧 尝试使用本地存储模式连接...")

        # 定义要尝试的本地主机地址列表：优先使用原配置的 MILVUS_HOST，其次使用回环地址
        local_hosts = [
            MILVUS_HOST,   # 原有配置的地址（可能已不可达，但作为备选尝试）
            "127.0.0.1",   # IPv4 本地回环地址
            "localhost",   # 域名形式的本地主机
        ]

        # 双重循环：遍历所有主机地址，每种地址再分别尝试带别名和不带别名的连接方式
        for host in local_hosts:
            for use_alias in [True, False]:
                try:
                    # 构造本地连接的参数字典
                    local_params = {
                        "host": host,                                   # 要连接的主机地址
                        "port": MILVUS_PORT,                            # Milvus 端口（保持原配置）
                        "timeout": MILVUS_TIMEOUT * 2,                  # 本地连接超时时间加倍，给本地服务更多响应时间
                    }

                    # 如果本次尝试使用连接别名，则添加到参数字典中
                    if use_alias:
                        local_params["alias"] = _MILVUS_CONNECTION_ALIAS

                    # 如果配置了用户名和密码，也加入到连接参数中（本地模式也可能需要认证）
                    if MILVUS_USER:
                        local_params["user"] = MILVUS_USER
                    if MILVUS_PASSWORD:
                        local_params["password"] = MILVUS_PASSWORD

                    # 记录尝试的详细信息到调试日志
                    logger.debug(f"尝试连接: host={host}, port={MILVUS_PORT}, alias={use_alias}")

                    # 断开之前可能存在的同别名连接，避免冲突
                    try:
                        connections.disconnect(_MILVUS_CONNECTION_ALIAS)
                    except Exception:
                        # 如果之前没有该连接或断开失败，忽略异常继续执行
                        pass

                    # 根据是否使用别名选择不同的连接方式
                    if use_alias:
                        # 使用带别名的连接
                        connections.connect(**local_params)
                    else:
                        # 使用不带别名的连接（仅传 host, port, timeout）
                        connections.connect(
                            host=local_params["host"],
                            port=local_params["port"],
                            timeout=local_params["timeout"],
                        )

                    # 连接成功后获取服务器版本，确认连接真正有效
                    version = utility.get_server_version()
                    logger.info(f"✅ 本地模式连接成功，版本: {version}")

                    # 调用初始化集合的函数，创建所需的 Milvus 集合（角色知识库、长期记忆、股票数据等）
                    _init_milvus_collections()
                    # 设置全局标志，表示 Milvus 已可用
                    set_milvus_available(True)
                    return True  # 连接成功，函数返回 True

                except Exception as e:
                    # 当前主机和别名组合的连接失败，记录调试日志并继续尝试下一组
                    logger.debug(f"本地连接尝试失败: {host}:{MILVUS_PORT}, error: {e}")

        # 所有主机和别名组合都尝试完毕但均失败，记录警告日志
        logger.warning("⚠️ 所有本地连接方式均失败")
        return False  # 返回 False 表示本地模式连接失败

    except Exception as e:
        # 捕获 try 块之外的异常（如导入模块失败等），记录错误日志
        logger.error(f"❌ 本地模式连接也失败: {e}")
        return False


def _init_milvus_collections():
    """初始化所有 Milvus 集合（知识库集合、长期记忆、股票数据）"""
    try:
        # 获取所有角色列表（通常从数据库或配置中获取）
        roles = get_roles()
        # 记录日志，显示即将初始化的角色知识库集合数量
        logger.info(f"📚 初始化 {len(roles)} 个角色知识库集合...")
        # 遍历每个角色，为其创建或确保存在对应的知识库集合
        for r in roles:
            ensure_knowledge_collection(int(r["id"]))
        # 所有角色集合初始化成功
        logger.info("✅ 角色知识库集合初始化完成")
    except Exception as e:
        # 如果上述过程失败（例如无法获取角色列表），则降级为只为 role_id=1 创建集合
        logger.warning(f"⚠️ 初始化角色知识库集合失败，降级创建 role_id=1: {e}")
        ensure_knowledge_collection(1)

    try:
        # 创建长期记忆集合（用于存储角色与用户之间的对话历史摘要等）
        _create_long_term_memory_collection()
        logger.info("✅ 长期记忆集合初始化完成")
    except Exception as e:
        # 长期记忆集合创建失败则记录错误并抛出异常（该集合必须存在）
        logger.error(f"❌ 创建长期记忆集合失败: {e}")
        raise

    try:
        # 创建股票数据集合（用于存储股票相关信息，如代码、价格、新闻等）
        _create_stock_collection()
        logger.info("✅ 股票数据集合初始化完成")
    except Exception as e:
        # 股票数据集合创建失败同样记录错误并抛出异常
        logger.error(f"❌ 创建股票数据集合失败: {e}")
        raise


def _create_long_term_memory_collection():
    """创建长期记忆集合（存储用户对话历史向量）"""
    # 导入 Milvus 相关类：Collection（集合操作）、CollectionSchema（集合模式）、FieldSchema（字段定义）、DataType（数据类型）、utility（工具函数）
    from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

    # 定义长期记忆集合的名称
    memory_collection_name = "long_term_memory"
    # 检查 Milvus 中是否已存在该名称的集合
    if not utility.has_collection(memory_collection_name):
        # 如果不存在，记录创建日志
        logger.info(f"创建长期记忆集合: {memory_collection_name}")
        # 定义集合的字段结构
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 主键字段：自动生成的唯一ID
            FieldSchema(name="user_id", dtype=DataType.INT64),                             # 用户ID，用于区分不同用户
            FieldSchema(name="role_id", dtype=DataType.INT64),                             # 角色ID，关联智能体角色
            FieldSchema(name="memory_text", dtype=DataType.VARCHAR, max_length=65535),    # 记忆的文本内容（最长65535字符）
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM), # 稠密向量，维度由全局EMBEDDING_DIM决定
            FieldSchema(name="timestamp", dtype=DataType.INT64),                           # 时间戳，记录记忆产生时间
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512)           # 摘要，对记忆文本的简要概括（最长512字符）
        ]
        # 使用定义好的字段创建集合模式，添加描述信息
        schema = CollectionSchema(fields, "长期记忆集合")
        # 实例化集合对象
        collection = Collection(memory_collection_name, schema)
        # 为集合的稠密向量字段创建索引并加载到内存（加速检索）
        _create_index_and_load(collection, "dense_vector")
        logger.info(f"长期记忆集合 {memory_collection_name} 创建完成")
    else:
        # 如果集合已存在，直接获取该集合对象
        collection = Collection(memory_collection_name)
        # 确保集合已加载到内存中（如果未加载则加载）
        _ensure_collection_loaded(collection, memory_collection_name)


def _create_stock_collection():
    """创建股票数据集合（存储股票信息和向量）"""
    # 导入 Milvus 相关组件：Collection（集合操作）、CollectionSchema（集合模式）、FieldSchema（字段定义）、DataType（数据类型）、utility（工具函数）
    from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

    # 定义股票数据集合的名称
    stock_collection_name = "stock_data"
    # 检查 Milvus 中是否已存在该名称的集合
    if not utility.has_collection(stock_collection_name):
        # 如果不存在，记录创建日志
        logger.info(f"创建股票数据集合: {stock_collection_name}")
        # 定义集合的字段结构
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 主键字段：自动生成的唯一ID
            FieldSchema(name="stock_code", dtype=DataType.VARCHAR, max_length=20),        # 股票代码（如 "000001.SZ"）
            FieldSchema(name="stock_name", dtype=DataType.VARCHAR, max_length=100),       # 股票名称（如 "平安银行"）
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),           # 股票相关文本内容（新闻、公告、研报等）
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM), # 稠密向量，由文本内容编码得到，维度由全局 EMBEDDING_DIM 决定
            FieldSchema(name="sector", dtype=DataType.VARCHAR, max_length=50),            # 所属板块/行业（如 "银行"、"科技"）
            FieldSchema(name="fetch_time", dtype=DataType.INT64),                         # 数据获取时间戳（Unix 秒或毫秒）
            FieldSchema(name="latest_price", dtype=DataType.DOUBLE),                      # 最新股价
            FieldSchema(name="change_pct", dtype=DataType.DOUBLE),                        # 涨跌幅百分比（如 +2.5 表示上涨2.5%）
            FieldSchema(name="pe_ratio", dtype=DataType.DOUBLE),                          # 市盈率（Price-to-Earnings Ratio）
            FieldSchema(name="market_cap", dtype=DataType.DOUBLE)                         # 市值（总市值，单位通常为亿元或元）
        ]
        # 使用定义好的字段创建集合模式，添加描述信息
        schema = CollectionSchema(fields, "股票数据集合")
        # 实例化集合对象
        collection = Collection(stock_collection_name, schema)
        # 为集合的稠密向量字段创建索引并加载到内存（加速向量检索）
        _create_index_and_load(collection, "dense_vector")
        logger.info(f"股票数据集合 {stock_collection_name} 创建完成")
    else:
        # 如果集合已存在，直接获取该集合对象
        collection = Collection(stock_collection_name)
        # 确保集合已加载到内存中（如果未加载则加载）
        _ensure_collection_loaded(collection, stock_collection_name)

def _create_index_and_load(collection: Any, field_name: str):
    """为指定字段创建索引并加载集合"""
    # 获取当前集合的所有索引信息（返回包含索引字段名等信息的对象列表）
    index_info = collection.indexes
    # 检查是否已经存在针对指定字段的索引（通过判断索引列表中有没有字段名等于 field_name 的索引）
    if not any(idx.field_name == field_name for idx in index_info):
        # 如果不存在，则为该字段创建索引
        collection.create_index(
            field_name,                                           # 要创建索引的字段名（通常是向量字段）
            {"index_type": "IVF_FLAT",                            # 索引类型：IVF_FLAT（倒排文件，精度高）
             "metric_type": "IP",                                 # 距离度量类型：IP（内积，适用于归一化向量的余弦相似度）
             "params": {"nlist": 128}}                            # 索引参数：聚类中心数量为 128
        )
    # 确保集合已加载到内存中（若未加载则加载），以便进行搜索操作
    _ensure_collection_loaded(collection)


def init_long_term_memory():
    """确保长期记忆集合存在（独立调用，主要用于服务启动时）"""
    # 检查 Milvus 是否就绪（连接是否可用），如果不可用则直接返回，不做任何操作
    if not milvus_ready():
        return
    # 导入 Milvus 相关组件：Collection（集合操作）、CollectionSchema（模式）、FieldSchema（字段定义）、DataType（数据类型）、utility（工具函数）
    from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

    # 定义长期记忆集合的名称
    memory_collection_name = "long_term_memory"
    # 检查 Milvus 中是否已存在该名称的集合
    if not utility.has_collection(memory_collection_name):
        # 如果不存在，记录创建日志
        logger.info(f"创建长期记忆集合: {memory_collection_name}")
        # 定义集合的字段结构
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 主键，自增ID
            FieldSchema(name="user_id", dtype=DataType.INT64),                             # 用户ID
            FieldSchema(name="role_id", dtype=DataType.INT64),                             # 角色ID
            FieldSchema(name="memory_text", dtype=DataType.VARCHAR, max_length=65535),    # 记忆文本内容
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM), # 稠密向量
            FieldSchema(name="timestamp", dtype=DataType.INT64),                           # 时间戳
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512)           # 摘要
        ]
        # 使用字段列表创建集合模式，添加描述
        schema = CollectionSchema(fields, "长期记忆集合")
        # 实例化集合对象
        collection = Collection(memory_collection_name, schema)
        # 为稠密向量字段创建索引（IVF_FLAT，内积距离，nlist=128）
        collection.create_index(
            "dense_vector",
            {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
        )
        # 将集合加载到内存中，以便后续检索
        collection.load()


def rebuild_bm25_model_for_collection(collection_name: str):
    """为某个知识库 collection 重建 BM25 模型（基于现有文本）"""
    # 导入 Milvus 的 Collection 类和 pymilvus 中的 BM25 稀疏嵌入函数
    from pymilvus import Collection
    from pymilvus.model.sparse import BM25EmbeddingFunction

    try:
        # 根据集合名称获取 Milvus 集合对象
        collection = Collection(collection_name)
        # 检查集合中的实体数量（文档数），如果为 0 则无需构建 BM25 模型
        if collection.num_entities == 0:
            logger.warning(f"知识库为空，跳过 BM25 模型重建: {collection_name}")
            return

        # 记录开始重建模型的日志
        logger.info(f"开始重建 BM25 模型: {collection_name}")

        # 创建 BM25 嵌入函数实例（用于拟合文档和编码查询）
        bm25 = BM25EmbeddingFunction()

        # 从 Milvus 集合中查询所有实体的 text 字段（expr="id >= 0" 表示匹配所有文档）
        results = collection.query(expr="id >= 0", output_fields=["text"])
        # 提取查询结果中的文本内容，组成文本列表
        texts = [r["text"] for r in results]

        # 用文本列表拟合 BM25 模型（计算词频、文档频率等统计量）
        bm25.fit(texts)
        # 将拟合好的 BM25 模型存储到全局字典中，键为集合名称
        set_bm25_model(collection_name, bm25)
        # 记录重建成功的日志，包含文档数量
        logger.info(f"BM25 模型重建完成: {collection_name}，包含 {len(texts)} 条文档")
    except Exception as e:
        # 捕获异常，记录警告日志（不抛出，避免影响主流程）
        logger.warning(f"⚠️ 重建 BM25 模型失败: {collection_name}: {e}")


def add_documents_to_milvus(texts: List[str], source: str, doc_id: str = None, role_id: int = 1):
    """添加文档到指定角色的知识库（文本分块已预先完成）"""
    if not texts:
        return

    collection = ensure_knowledge_collection(role_id)
    model = get_embedding_model()

    dense_vectors = model.encode(texts, normalize_embeddings=True)

    # 获取该 collection 的 BM25 模型（如果没有则使用空字典）
    bm25 = get_bm25_models().get(collection.name)
    if bm25:
        sparse_vectors = bm25.encode_documents(texts)
    else:
        sparse_vectors = [{} for _ in texts]

    create_time = int(datetime.now().timestamp())
    data = []
    for idx, text in enumerate(texts):
        summary = text[:50] + "..." if len(text) > 50 else text
        data.append({
            "text": text,
            "dense_vector": dense_vectors[idx].tolist(),
            "sparse_vector": sparse_vectors[idx],
            "source": source,
            "create_time": create_time,
            "summary": summary,
            "doc_id": doc_id or f"doc_{create_time}_{idx}"
        })

    collection.insert(data)
    collection.flush()

    # 插入后重建 BM25 模型，以包含新文档
    rebuild_bm25_model_for_collection(collection.name)

    logger.info(f"已添加 {len(texts)} 条文档到知识库: role_id={role_id}")


# 检索缓存（避免短时间内相同查询重复检索）
_search_cache = {}
_search_cache_lock = threading.Lock()
_DISABLE_CACHE = os.getenv("DISABLE_CACHE", "false").lower() == "true"


def hybrid_search(query: str, role_id: int, top_k: int = TOP_K_RETRIEVE) -> List[Dict[str, str]]:
    """
    混合检索：稠密向量 + BM25 稀疏向量，使用 RRF 融合。
    支持缓存（默认缓存 300 秒）。
    """
    # 导入 Milvus 相关的类：Collection（集合操作）、AnnSearchRequest（向量搜索请求）、RRFRanker（RRF融合排序器）
    from pymilvus import Collection, AnnSearchRequest, RRFRanker
    import scipy.sparse as sp  # 用于处理稀疏向量（BM25 产生的稀疏矩阵）

    # 检查是否禁用了缓存（全局变量 _DISABLE_CACHE 控制）
    if not _DISABLE_CACHE:
        # 构造缓存键：角色ID + 查询文本前100字符（避免过长）
        cache_key = f"{role_id}:{query[:100]}"
        # 使用线程锁保护缓存字典的并发访问
        with _search_cache_lock:
            # 如果缓存中存在该键值
            if cache_key in _search_cache:
                cache_entry = _search_cache[cache_key]  # 获取缓存条目（包含结果和时间戳）
                # 检查缓存是否在有效期内（300秒）
                if time.time() - cache_entry["time"] < 300:
                    logger.debug(f"命中检索缓存: {query[:30]}...")
                    return cache_entry["result"]  # 直接返回缓存结果，跳过实际检索

    # 检查 Milvus 是否可用（milvus_ready 函数判断连接状态）
    if not milvus_ready():
        logger.debug("Milvus 不可用，跳过向量检索")
        return []  # 不可用时返回空列表

    try:
        # 获取或创建当前角色对应的知识库集合（Collection）
        collection = ensure_knowledge_collection(role_id)
        # 确保集合已加载到内存中，以便进行搜索
        _ensure_collection_loaded(collection)

        # 如果集合中没有实体数据，直接返回空结果
        if collection.num_entities == 0:
            logger.debug(f"集合 {collection.name} 为空，返回空结果")
            return []

        # ----- 稠密向量检索部分 -----
        # 获取稠密嵌入模型（例如 sentence-transformers）
        dense_model = get_embedding_model()
        # 将查询文本编码为稠密向量，并进行 L2 归一化（IP 内积距离需要归一化）
        query_dense_vec = dense_model.encode([query], normalize_embeddings=True)[0].tolist()
        # 设置稠密检索参数：使用内积距离（IP），nprobe 控制搜索聚类数量
        dense_search_params = {"metric_type": "IP", "params": {"nprobe": 64}}
        # 创建稠密向量搜索请求，限制返回数量为 min(top_k, 5)（防止稀疏检索失效时取太多）
        dense_req = AnnSearchRequest(
            data=[query_dense_vec],
            anns_field="dense_vector",   # 集合中的稠密向量字段名
            param=dense_search_params,
            limit=min(top_k, 5)
        )

        # ----- BM25 稀疏向量检索部分 -----
        # 获取当前集合对应的 BM25 模型（字典存储，键为集合名）
        bm25 = get_bm25_models().get(collection.name)
        # 如果 BM25 模型不存在但集合中有数据，则尝试构建该集合的 BM25 模型
        if bm25 is None and collection.num_entities > 0:
            rebuild_bm25_model_for_collection(collection.name)
            bm25 = get_bm25_models().get(collection.name)  # 重新获取

        # 如果 BM25 索引可用，进行稀疏检索
        if bm25:
            try:
                # 使用 BM25 模型将查询文本编码为稀疏向量（词袋形式）
                sparse_mat = bm25.encode_queries([query])
                # 取出第一行（对应查询的稀疏向量）
                query_sparse_vec = sparse_mat.getrow(0)
                # 如果是稀疏矩阵格式，将可能出现的负值截断为 0（BM25 原始值非负，但某些实现可能产生负值）
                if sp.issparse(query_sparse_vec):
                    query_sparse_vec.data[query_sparse_vec.data < 0] = 0.0
                    query_sparse_vec.eliminate_zeros()  # 移除值为 0 的元素，节省存储和计算
                # 稀疏检索参数：使用内积距离（IP）计算相似度
                sparse_search_params = {"metric_type": "IP"}
                # 创建稀疏向量搜索请求，返回数量为 top_k
                sparse_req = AnnSearchRequest(
                    data=[query_sparse_vec],
                    anns_field="sparse_vector",  # 集合中的稀疏向量字段名
                    param=sparse_search_params,
                    limit=top_k
                )

                # 使用 RRF（倒数排名融合）算法融合稠密和稀疏的搜索结果
                rerank = RRFRanker()
                # 执行混合搜索
                results = collection.hybrid_search(
                    reqs=[dense_req, sparse_req],  # 两个搜索请求
                    rerank=rerank,                 # RRF 融合排序器
                    limit=top_k,                   # 最终返回 top_k 条
                    output_fields=["text", "source", "create_time", "summary"]  # 需要返回的字段
                )
            except Exception as e:
                # 如果稀疏检索失败（例如集合未建立稀疏索引），降级为纯稠密检索
                logger.warning(f"稀疏检索失败，使用纯稠密检索: {e}")
                results = _execute_dense_search(collection, query_dense_vec, dense_search_params, top_k)
        else:
            # 没有 BM25 模型时仅使用稠密检索
            logger.debug("BM25 模型未构建，仅使用稠密检索")
            results = _execute_dense_search(collection, query_dense_vec, dense_search_params, top_k)

        # 解析 Milvus 返回的结果格式，转换为 Python 字典列表
        results = _parse_search_results(results)

        # ----- 缓存处理 -----
        with _search_cache_lock:
            # 将本次检索结果存入缓存，记录当前时间戳
            _search_cache[cache_key] = {"result": results, "time": time.time()}
            # 控制缓存大小不超过 500 条，超过时删除最早的一条（根据时间戳）
            if len(_search_cache) > 500:
                oldest_key = min(_search_cache.keys(), key=lambda k: _search_cache[k]["time"])
                del _search_cache[oldest_key]

        return results

    except Exception as e:
        # 捕获任何意外异常，记录错误日志并返回空列表，避免中断主流程
        logger.error(f"混合检索失败: {e}")
        return []


def _execute_dense_search(collection: Any, query_vec: list, params: dict, top_k: int):
    """执行纯稠密向量检索（备用）"""
    # 执行单路稠密向量搜索：只使用 dense_vector 字段进行 ANN 搜索
    return collection.search(
        data=[query_vec],           # 查询向量列表（单条查询时也是一个元素）
        anns_field="dense_vector",  # 要搜索的向量字段
        param=params,               # 搜索参数（如 metric_type, nprobe）
        limit=top_k,                # 返回的最大结果数
        output_fields=["text", "source", "create_time", "summary"]  # 返回的实体字段
    )


def _parse_search_results(results) -> List[Dict[str, str]]:
    """解析 Milvus 检索结果，提取文本和元信息"""
    retrieved = []  # 存储解析后的文档列表
    # Milvus 返回的结果是嵌套结构：results 是一个列表，每个元素是一次搜索请求的结果（这里只有一个请求或两个融合后也只有一个）
    for hits in results:          # results 通常只有一个元素（因为 hybrid_search 返回单个搜索结果集）
        for hit in hits:          # 遍历每个匹配到的实体（hit）
            retrieved.append({
                "text": hit.entity.get("text"),           # 文档正文
                "source": hit.entity.get("source"),       # 来源文件名或标识
                "create_time": hit.entity.get("create_time"),  # 创建时间
                "summary": hit.entity.get("summary")      # 文档摘要
            })
    return retrieved


def rerank_documents(query: str, documents: List[Dict[str, str]], top_n: int = 3) -> List[Dict[str, str]]:
    """使用 BGE-reranker 对检索结果进行重排序，返回最相关的 top_n 条"""
    # 如果没有传入文档，直接返回空列表
    if not documents:
        return []

    # 获取全局的重排序模型（通常是 Cross-Encoder 类模型）
    reranker = get_reranker_model()
    # 模型不可用时，保持原顺序并只取前 top_n 条（有损降级）
    if reranker is None:
        logger.debug("重排序模型不可用，使用原始顺序")
        return documents[:top_n]

    try:
        # 构造每一对 (query, doc_text) 用于模型打分
        pairs = [[query, doc["text"]] for doc in documents]

        scores = None
        try:
            # 尝试一次计算所有句对的相关性分数
            scores = reranker.compute_score(pairs)
        except Exception as e:
            # 如果一次性计算失败（可能由于显存限制或序列过长），采用分批处理
            logger.debug(f"标准计算失败，尝试分批处理: {e}")
            batch_size = 10
            scores = []
            # 分批遍历所有句对
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                try:
                    batch_scores = reranker.compute_score(batch)  # 计算当前批次分数
                    scores.extend(batch_scores)                  # 累加结果
                except Exception as batch_e:
                    # 如果任一批次失败，则放弃重排序，返回原始顺序的前 top_n 条
                    logger.debug(f"批次处理失败: {batch_e}")
                    return documents[:top_n]

        # 将文档和对应的分数配对
        scored_docs = list(zip(documents, scores))
        # 按分数降序排序（分数越高表示越相关）
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 取出前 top_n 条文档（丢弃分数）
        result = [doc for doc, score in scored_docs[:top_n]]
        logger.debug(f"重排序完成，原始 {len(documents)} 条，返回 {len(result)} 条")
        return result

    except Exception as e:
        # 任何未预期的错误都降级为原始顺序
        logger.warning(f"重排序失败，使用原始顺序: {e}")
        return documents[:top_n]


def retrieve_long_term_memory(user_id: int, role_id: int, query: str, top_k: int = 2) -> List[str]:
    """从 Milvus 中检索相关的长期记忆"""
    from pymilvus import Collection
    
    if not milvus_ready():
        logger.debug("Milvus 不可用，跳过长期记忆检索")
        return []
    
    try:
        collection = Collection("long_term_memory")
        _ensure_collection_loaded(collection)

        model = get_embedding_model()
        query_vec = model.encode([query], normalize_embeddings=True)[0].tolist()
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

        expr = f"user_id == {user_id} && role_id == {role_id}"

        results = collection.search(
            data=[query_vec],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["memory_text"]
        )

        memories = []
        for hits in results:
            for hit in hits:
                memories.append(hit.entity.get("memory_text"))

        return memories

    except Exception as e:
        logger.error(f"检索长期记忆失败: {e}")
        return []


def store_long_term_memory(user_id: int, role_id: int, user_msg: str, assistant_msg: str):
    """存储一次对话为长期记忆（向量化后存入 Milvus）"""
    # 导入 Milvus 的 Collection 类
    from pymilvus import Collection

    # 检查 Milvus 是否可用，不可用时跳过存储并记录调试日志
    if not milvus_ready():
        logger.debug("Milvus 不可用，跳过长期记忆存储")
        return

    try:
        # 获取长期记忆集合对象（集合名称为 "long_term_memory"）
        collection = Collection("long_term_memory")
        # 确保集合已加载到内存中（若未加载则加载）
        _ensure_collection_loaded(collection)

        # 获取全局稠密嵌入模型（用于将文本转换为向量）
        model = get_embedding_model()

        # 构造记忆文本：将用户问题和助手回答拼接成一个完整的对话记录
        memory_text = f"用户问: {user_msg}\n助手回答: {assistant_msg}"
        # 将记忆文本编码为稠密向量，并进行 L2 归一化（因为后续使用内积距离）
        vector = model.encode([memory_text], normalize_embeddings=True)[0].tolist()
        # 生成摘要：如果用户消息超过100字符则截取前100个字符并加省略号，否则直接使用原消息
        summary = user_msg[:100] + "..." if len(user_msg) > 100 else user_msg

        # 准备要插入的数据，按字段组织成字典
        data = [{
            "user_id": user_id,                                    # 用户ID
            "role_id": role_id,                                    # 角色ID
            "memory_text": memory_text,                            # 完整的对话记忆文本
            "dense_vector": vector,                                # 稠密向量
            "timestamp": int(datetime.now().timestamp()),          # 当前时间戳（秒）
            "summary": summary                                      # 摘要
        }]
        # 将数据插入到集合中
        collection.insert(data)
        # 强制刷新，确保持久化到磁盘（使数据可立即被检索）
        collection.flush()
        # 记录成功存储的信息日志
        logger.info(f"已存储长期记忆: user={user_id}, role={role_id}")

    except Exception as e:
        # 捕获异常并记录错误日志，不抛出，避免影响主流程
        logger.error(f"存储长期记忆失败: {e}")


# ==================== Redis 短期记忆操作 ====================

_in_memory_history: Dict[str, List[Dict[str, str]]] = {}


def get_chat_history(user_id: int, role_id: int) -> List[Dict[str, str]]:
    """获取短期聊天历史（优先 Redis，降级到内存）"""
    r = get_redis_client()
    key = f"chat:{user_id}:{role_id}"
    if r is None:
        return _in_memory_history.get(key, [])
    try:
        history_str = r.get(key)
        if history_str:
            import json
            return json.loads(history_str)
    except Exception as e:
        logger.warning(f"Redis数据类型错误，清理无效数据: {e}")
        try:
            r.delete(key)
        except:
            pass
    return []


def add_to_history(user_id: int, role_id: int, user_msg: str, assistant_msg: str):
    """
    添加一轮对话到短期记忆（自动保留最近 MEMORY_LENGTH 轮对话）
    :param user_id: 用户唯一ID
    :param role_id: 角色/会话ID，用于区分不同对话场景
    :param user_msg: 用户本次发送的消息内容
    :param assistant_msg: 助手回复的消息内容
    """
    # 获取 Redis 客户端实例（用于持久化存储聊天记录）
    r = get_redis_client()

    # 拼接 Redis 存储键名：chat:用户ID:角色ID，确保每个用户+角色会话唯一
    key = f"chat:{user_id}:{role_id}"

    # 从存储（Redis/内存）中获取该用户+角色的历史对话列表
    history = get_chat_history(user_id, role_id)

    # 将用户消息追加到历史记录（固定格式：role=user）
    history.append({"role": "user", "content": user_msg})
    # 将助手回复追加到历史记录（固定格式：role=assistant）
    history.append({"role": "assistant", "content": assistant_msg})

    # 限制历史记录长度：一轮对话 = 1用户 + 1助手，共2条记录
    # 超过最大记忆轮数时，只保留最近的 MEMORY_LENGTH 轮（即最新的 2*MEMORY_LENGTH 条）
    if len(history) > MEMORY_LENGTH * 2:
        history = history[-MEMORY_LENGTH * 2:]

    # 根据是否启用 Redis，选择存储方式
    if r is None:
        # Redis 未连接/不可用：使用全局字典 _in_memory_history 做内存临时存储
        _in_memory_history[key] = history
    else:
        # Redis 可用：导入 json 模块，将历史记录序列化为 JSON 字符串存入 Redis
        import json
        r.set(key, json.dumps(history))


def clear_history(user_id: int, role_id: int):
    """清空指定用户和角色的短期聊天历史"""
    r = get_redis_client()
    key = f"chat:{user_id}:{role_id}"
    if r is None:
        _in_memory_history.pop(key, None)
    else:
        r.delete(key)


# ==================== 股票数据相关函数 ====================

def fetch_stock_data_from_eastmoney() -> List[Dict[str, Any]]:
    """从东方财富网（AKShare）获取全量A股实时行情数据"""
    try:
        import akshare as ak
        logger.info("开始从 AKShare 获取股票数据...")
        df = ak.stock_zh_a_spot_em()

        if df.empty:
            logger.warning("获取到的股票数据为空")
            return []

        stock_list = []
        for _, row in df.iterrows():
            stock_info = {
                "stock_code": str(row.get("代码", "")).strip(),
                "stock_name": str(row.get("名称", "")).strip(),
                "latest_price": float(row.get("最新价", 0)),
                "change_pct": float(row.get("涨跌幅", 0)),
                "volume": float(row.get("成交量", 0)),
                "turnover": float(row.get("成交额", 0)),
                "pe_ratio": float(row.get("市盈率-动态", 0)),
                "market_cap": float(row.get("总市值", 0)),
                "circulating_cap": float(row.get("流通市值", 0)),
                "sector": str(row.get("所属行业", "")).strip()
            }
            stock_list.append(stock_info)

        logger.info(f"成功获取 {len(stock_list)} 只股票数据")
        return stock_list
    except Exception as e:
        logger.error(f"获取股票数据失败: {e}")
        return []


def store_stock_data_to_milvus(stock_list: List[Dict[str, Any]]):
    """将股票数据存入 Milvus（先清空旧数据）"""
    if not stock_list:
        return

    from pymilvus import Collection

    collection = Collection(STOCK_COLLECTION_NAME)
    model = get_embedding_model()

    texts = []
    for stock in stock_list:
        text = f"{stock['stock_name']}({stock['stock_code']}) 最新价{stock['latest_price']}元 涨跌幅{stock['change_pct']}% 市盈率{stock['pe_ratio']}"
        texts.append(text)

    vectors = model.encode(texts, normalize_embeddings=True)

    data = []
    fetch_time = int(datetime.now().timestamp())
    for idx, stock in enumerate(stock_list):
        data.append({
            "stock_code": stock["stock_code"],
            "stock_name": stock["stock_name"],
            "text": texts[idx],
            "dense_vector": vectors[idx].tolist(),
            "sector": stock.get("sector", ""),
            "fetch_time": fetch_time,
            "latest_price": stock["latest_price"],
            "change_pct": stock["change_pct"],
            "pe_ratio": stock["pe_ratio"],
            "market_cap": stock.get("market_cap", 0)
        })

    collection.delete("id >= 0")
    collection.insert(data)
    collection.flush()
    logger.info(f"已存储 {len(stock_list)} 条股票数据")


def sync_stock_data_to_milvus():
    """手动同步股票数据（由 API 调用触发）"""
    stock_list = fetch_stock_data_from_eastmoney()
    if stock_list:
        store_stock_data_to_milvus(stock_list)
    else:
        logger.warning("未获取到股票数据，同步失败")


def retrieve_stock_data_from_local_cache(query: str, top_k: int = TOP_K_STOCK) -> List[Dict[str, Any]]:
    """从本地 JSON 缓存中检索股票信息（当 Milvus 不可用时降级使用）"""
    stocks = _load_local_stock_cache()
    if not stocks:
        return []

    tokens = [t.lower() for t in simple_chinese_tokenizer(query) if t]
    query_text = (query or "").lower()
    ranked: List[Any] = []
    for item in stocks:
        code = str(item.get("stock_code") or item.get("代码") or item.get("code") or "").strip()
        name = str(item.get("stock_name") or item.get("名称") or item.get("name") or "").strip()
        sector = str(item.get("sector") or item.get("所属行业") or item.get("industry") or "").strip()
        haystack = f"{code} {name} {sector}".lower()
        score = 0
        if code and code in query_text:
            score += 10
        if name and name.lower() in query_text:
            score += 8
        for token in tokens:
            if token and token in haystack:
                score += 2 if len(token) > 1 else 1
        if score <= 0:
            continue
        ranked.append((
            score,
            {
                "stock_code": code,
                "stock_name": name,
                "latest_price": item.get("latest_price") or item.get("最新价") or item.get("price") or 0,
                "change_pct": item.get("change_pct") or item.get("涨跌幅") or item.get("changePercent") or 0,
                "pe_ratio": item.get("pe_ratio") or item.get("市盈率") or item.get("市盈率-动态") or 0,
                "sector": sector,
            },
        ))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [stock for _, stock in ranked[:top_k]]


def retrieve_stock_data(query: str, top_k: int = TOP_K_STOCK) -> List[Dict[str, Any]]:
    """检索股票数据：优先使用 Milvus 向量检索，失败则降级到本地缓存检索"""
    from pymilvus import Collection

    if not milvus_ready():
        return retrieve_stock_data_from_local_cache(query, top_k=top_k)
    collection = Collection(STOCK_COLLECTION_NAME)
    model = get_embedding_model()

    query_vec = model.encode([query], normalize_embeddings=True)[0].tolist()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    try:
        results = collection.search(
            data=[query_vec],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            output_fields=["stock_code", "stock_name", "latest_price", "change_pct", "pe_ratio", "sector"]
        )

        stocks = []
        for hits in results:
            for hit in hits:
                stocks.append({
                    "stock_code": hit.entity.get("stock_code"),
                    "stock_name": hit.entity.get("stock_name"),
                    "latest_price": hit.entity.get("latest_price"),
                    "change_pct": hit.entity.get("change_pct"),
                    "pe_ratio": hit.entity.get("pe_ratio"),
                    "sector": hit.entity.get("sector")
                })

        return stocks
    except Exception as e:
        logger.warning(f"检索股票数据失败: {e}")
        return retrieve_stock_data_from_local_cache(query, top_k=top_k)


# ==================== 角色知识库自动入库 ====================

def seed_role_knowledge_if_empty(role_id: int, role_name: str):
    """
    如果指定角色的知识库为空，则自动从 data/ 目录中查找与角色相关的数据集（CSV/TXT）
    并加载入库。主要用于首次启动时自动填充基础金融问答数据。
    """
    # 确保角色对应的知识库集合存在，并获取集合对象
    col = ensure_knowledge_collection(role_id)
    # 检查集合中已有的文档数量，如果大于0则说明已存在数据，直接返回不再重复填充
    if col.num_entities > 0:
        return

    # 根据角色名称查找 data/ 目录下匹配的数据集文件（例如 role_name.csv 或 role_name.txt）
    dataset_files = _find_role_dataset_files(role_name)
    # 如果没有找到任何数据集文件，记录警告日志并退出
    if not dataset_files:
        logger.warning(f"未找到角色数据集，跳过自动入库: role={role_name}({role_id})")
        return

    # 初始化一个列表，用于存储从所有数据集文件中读取出的文本行/条目
    all_texts: List[str] = []
    # 遍历每个找到的数据集文件路径
    for p in dataset_files:
        try:
            # 调用 _load_texts_from_dataset 读取该文件中的所有文本（CSV 按行或 TXT 按段落），并扩展到 all_texts 列表
            all_texts.extend(_load_texts_from_dataset(p))
        except Exception as e:
            # 如果读取某个文件失败，记录警告日志并跳过该文件，继续处理其他文件
            logger.warning(f"读取数据集失败，跳过: {p}: {e}")
            continue

    # 如果所有文件都未能读取到任何文本，则直接返回（没有数据可插入）
    if not all_texts:
        return
    # 调用 add_documents_to_milvus 将读取到的所有文本批量插入到 Milvus 集合中
    # 参数：texts=文本列表，source=标注来源为数据集自动导入，doc_id=种子数据标识，role_id=关联的角色ID
    add_documents_to_milvus(
        texts=all_texts,
        source="dataset:auto",
        doc_id=f"seed_{role_name}",
        role_id=role_id,
    )