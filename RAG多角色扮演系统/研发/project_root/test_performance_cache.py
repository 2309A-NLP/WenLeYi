#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""性能测试脚本（启用缓存）"""

import http.client
import json
import time

def test_single_request(message):
    conn = http.client.HTTPConnection('127.0.0.1', 8000, timeout=60)
    payload = json.dumps({'user_id': 1, 'role_id': 1, 'message': message})
    headers = {'Content-Type': 'application/json', 'Content-Length': str(len(payload))}
    
    start = time.time()
    conn.request('POST', '/chat', payload, headers)
    response = conn.getresponse()
    data = json.loads(response.read().decode())
    duration = (time.time() - start) * 1000
    
    return duration, data

# 测试问题列表
test_questions = [
    "什么是指数基金？它有什么特点？",
    "如何进行资产配置？",
    "解释一下市盈率PE指标",
    "什么是ETF？",
    "债券投资有哪些风险？"
]

print("=" * 60)
print("RAG系统性能测试（启用缓存）")
print("=" * 60)

# 第一轮测试（冷缓存）
print("\n--- 第一轮测试（冷缓存）---")
total_time = 0
times = []

for i, question in enumerate(test_questions):
    print(f"\n[{i+1}/{len(test_questions)}] 测试: {question}")
    try:
        duration, data = test_single_request(question)
        times.append(duration)
        total_time += duration
        print(f"  响应时间: {duration:.0f}ms")
        print(f"  回答预览: {data.get('answer', '')[:60]}...")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

print("\n第一轮测试结果:")
valid_times = [t for t in times if t > 0]
if valid_times:
    print(f"  平均响应时间: {sum(valid_times)/len(valid_times):.0f}ms")

# 第二轮测试（热缓存）
print("\n\n--- 第二轮测试（热缓存）---")
total_time = 0
times = []

for i, question in enumerate(test_questions):
    print(f"\n[{i+1}/{len(test_questions)}] 测试: {question}")
    try:
        duration, data = test_single_request(question)
        times.append(duration)
        total_time += duration
        print(f"  响应时间: {duration:.0f}ms")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

print("\n第二轮测试结果（热缓存）:")
valid_times = [t for t in times if t > 0]
if valid_times:
    print(f"  平均响应时间: {sum(valid_times)/len(valid_times):.0f}ms")
    print(f"  最大响应时间: {max(valid_times):.0f}ms")
    print(f"  最小响应时间: {min(valid_times):.0f}ms")
