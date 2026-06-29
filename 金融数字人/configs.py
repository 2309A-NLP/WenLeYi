# -*- coding: utf-8 -*-
"""
Linly-Talker配置文件
定义端口、IP地址、模型路径、SSL证书等配置参数
"""

# Linly-Talker WebUI运行端口（Gradio网页界面）
port = 6006

# API模式配置（暂时仅适用于Linly API模式）
mode = "api"

# 【改动2：configs.py — 改了监听IP】
# 把ip = "127.0.0.1"改成了ip = "0.0.0.0"
# 原来只允许本地访问，改成0.0.0.0后允许AutoDL云服务器外部访问。
# 监听IP：0.0.0.0表示允许外部访问（AutoDL云服务器需要）
ip = "0.0.0.0"
api_port = 7871

# 离线模式（使用本地Qwen模型，已不再使用）
mode = "offline"
model_path = "Qwen/Qwen-1_8B-Chat"

# SSL证书路径（麦克风录音功能需要HTTPS）
ssl_certfile = "./https_cert/cert.pem"
ssl_keyfile = "./https_cert/key.pem"
