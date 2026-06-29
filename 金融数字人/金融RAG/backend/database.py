# -*- coding: utf-8 -*-
"""
数据库模块：MySQL用户管理 + 对话历史存储
提供用户注册/登录、聊天记录保存/查询功能
如果MySQL不可用，系统会降级到无状态模式（内存缓存）运行
"""

import pymysql
import logging
from utils import MYSQL_CONFIG  # 从配置文件读取MySQL连接参数

logger = logging.getLogger(__name__)


def get_connection():
    """获取MySQL数据库连接"""
    return pymysql.connect(**MYSQL_CONFIG)


def init_database():
    """
    初始化数据库表（首次运行时自动创建）
    创建3张表：
    - users: 用户表（存储用户名和密码哈希）
    - chat_history: 聊天记录表（存储对话历史）
    - feedback: 反馈表（用户对回答的评分）
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # 用户表：存储注册用户信息
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,      -- 用户名（唯一）
                    password_hash VARCHAR(255) NOT NULL,        -- 密码的SHA256哈希值
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 聊天记录表：存储每条对话
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,               -- 关联用户ID
                    role VARCHAR(10) DEFAULT 'user',     -- 角色：user（用户）或 assistant（助手）
                    message TEXT,                        -- 消息内容
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user (user_id)             -- 按用户ID建索引，加速查询
                )
            """)
            # 反馈表：用户可以对回答打分（1-5分）
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    query TEXT,           -- 用户的问题
                    answer TEXT,          -- 系统的回答
                    rating INT COMMENT '1-5分',  -- 用户评分
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
        logger.info("数据库表初始化完成")
    finally:
        conn.close()


def register_user(username, password_hash):
    """
    注册新用户
    返回：新用户的ID，如果用户名已存在则返回None
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, password_hash)
            )
            conn.commit()
            return cur.lastrowid  # 返回新插入的用户ID
    except pymysql.IntegrityError:
        # 用户名重复（UNIQUE约束冲突）
        return None
    finally:
        conn.close()


def login_user(username, password_hash):
    """
    登录验证：检查用户名和密码是否匹配
    返回：(用户ID, 用户名) 元组，验证失败返回None
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username FROM users WHERE username=%s AND password_hash=%s",
                (username, password_hash)
            )
            return cur.fetchone()
    finally:
        conn.close()


def save_chat(user_id, role, message):
    """
    保存一条聊天记录到数据库
    参数：
        user_id: 用户ID
        role: 角色（'user'用户提问 / 'assistant'系统回答）
        message: 消息内容
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (user_id, role, message) VALUES (%s, %s, %s)",
                (user_id, role, message)
            )
            conn.commit()
    finally:
        conn.close()


def get_recent_history(user_id, limit=10):
    """
    获取用户最近的对话历史（按时间正序）
    用于多轮对话时给大模型提供上下文
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, message FROM chat_history WHERE user_id=%s ORDER BY id DESC LIMIT %s",
                (user_id, limit)
            )
            rows = cur.fetchall()
            rows.reverse()  # 数据库是倒序查的，反转回正序（最早的在前）
            return [{"role": r[0], "content": r[1]} for r in rows]
    finally:
        conn.close()
