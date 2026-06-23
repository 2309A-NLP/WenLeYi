# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-Agent 数字人项目-日程提醒智能体任务
配置文件：API Key、数据库连接、调度器参数
从 .env 文件读取环境变量，支持 os.environ.get() 覆盖
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-346e8d55cd0d4123ae308c6e48d94b3f77d508e8f42f4764b6bf3e4df52c772d")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# MySQL 数据库配置
DB_HOST = os.environ.get("DB_HOST", "172.22.224.1")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "root")
DB_NAME = os.environ.get("DB_NAME", "accounting_db")
DB_CHARSET = os.environ.get("DB_CHARSET", "utf8mb4")

# APScheduler 配置（每60秒扫描一次）
SCAN_INTERVAL_SECONDS = int(os.environ.get("SCAN_INTERVAL_SECONDS", "60"))

# Flask 配置
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5002"))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
