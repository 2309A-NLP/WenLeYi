# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
"""
配置文件：数据库连接、LLM接口等常量配置
优先从环境变量读取，兼容 .env 文件
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# MySQL数据库配置
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "172.22.224.1"),
    "port": int(os.environ.get("DB_PORT", 3306)),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "root"),
    "database": os.environ.get("DB_NAME", "accounting_db"),
    "charset": os.environ.get("DB_CHARSET", "utf8mb4"),
}

# DeepSeek LLM配置
LLM_CONFIG = {
    "api_key": os.environ.get("LLM_API_KEY", "sk-346e8d55cd0d4123ae308c6e48d94b3f77d508e8f42f4764b6bf3e4df52c772d"),
    "base_url": os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
    "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
}

# 表名常量
TABLE_NAME = os.environ.get("TABLE_NAME", "money_notes01")
CHAT_TABLE = os.environ.get("CHAT_TABLE", "chat_history01")
