# -*- coding: utf-8 -*-
"""
last_duan 包：金融领域 RAG 智能对话系统
包含以下模块：
- main.py: 项目入口文件
- api.py: API接口和业务逻辑
- database.py: 数据库操作
- utils.py: 工具函数和配置
"""

from .utils import logger

__version__ = "1.0.0"
__all__ = ["main", "api", "database", "utils"]

logger.info("📦 last_duan 包加载成功")
