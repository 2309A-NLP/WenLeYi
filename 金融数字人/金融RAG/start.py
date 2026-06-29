# -*- coding: utf-8 -*-
"""
金融对话系统启动脚本
用法：python start.py
功能：切换到backend目录，启动main.py（FastAPI服务）
"""

import subprocess
import sys
import os

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
# 后端代码目录
BACKEND_DIR = os.path.join(ROOT, "backend")


def main():
    """启动后端服务"""
    os.chdir(BACKEND_DIR)  # 切换到backend目录
    # 用当前Python解释器启动main.py
    proc = subprocess.Popen([sys.executable, "main.py"])

    try:
        proc.wait()  # 等待进程结束
    except KeyboardInterrupt:
        # Ctrl+C优雅关闭
        print("\n正在关闭...")
        proc.terminate()
        proc.wait()
        print("已关闭")


if __name__ == "__main__":
    main()
