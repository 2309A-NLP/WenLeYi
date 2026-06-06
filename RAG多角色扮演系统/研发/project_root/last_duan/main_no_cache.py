#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""禁用缓存启动脚本 - 用于真实性能测试"""

import os
import sys

# 禁用缓存
os.environ['DISABLE_CACHE'] = 'true'

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 启动服务
exec(open(os.path.join(current_dir, 'main.py'), encoding='utf-8').read())
