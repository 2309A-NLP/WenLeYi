# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-Agent数字人项目-Agent编排器任务
配置文件：LLM配置 + 各工具服务地址
所有配置优先从环境变量/.env文件读取
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))


# ==================== LLM 配置 ====================
# 使用小米 MiMo API 作为编排器的意图识别模型
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "mimo-v2.5")


# ==================== 工具服务地址 ====================
# 每个工单对应一个独立的Flask服务
TOOL_SERVICES = {
    "记账": {
        "name": "记账本智能体",
        "base_url": os.environ.get("TOOL_ACCOUNTING_URL", "http://127.0.0.1:5001"),
        "health": "/health",
        "endpoints": {
            "add":    "/api/v1/record/add",
            "query":  "/api/v1/record/query",
            "delete": "/api/v1/record/delete",
            "update": "/api/v1/record/update",
        }
    },
    "日程": {
        "name": "日程提醒智能体",
        "base_url": os.environ.get("TOOL_SCHEDULE_URL", "http://127.0.0.1:5002"),
        "health": "/health",
        "endpoints": {
            "add":    "/api/v1/schedule/add",
            "query":  "/api/v1/schedule/query",
            "delete": "/api/v1/schedule/delete",
            "update": "/api/v1/schedule/update",
        }
    },
    "文生图": {
        "name": "文生图智能体",
        "base_url": os.environ.get("TOOL_IMAGE_URL", "http://127.0.0.1:5003"),
        "health": "/health",
        "endpoints": {
            "generate":     "/api/v1/image/generate",
            "generate_all": "/api/v1/image/generate_all",
        }
    },
    "基金": {
        "name": "基金问答智能体",
        "base_url": os.environ.get("TOOL_FUND_URL", "http://127.0.0.1:5000"),
        "health": "/",
        "endpoints": {
            "ask": "/api/ask",
        }
    },
    "招股书": {
        "name": "招股书问答智能体",
        "base_url": os.environ.get("TOOL_PROSPECTUS_URL", "http://127.0.0.1:5005"),
        "health": "/health",
        "endpoints": {
            "query": "/api/v1/prospectus/query",
        }
    },
}


# ==================== Flask 配置 ====================
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "8080"))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() in ("true", "1", "yes")


# ==================== 请求超时 ====================
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "120"))  # 秒（工单4/5问答需要较长时间）
