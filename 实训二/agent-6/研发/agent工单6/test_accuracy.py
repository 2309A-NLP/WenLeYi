#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-Agent数字人项目-智能体任务
准确率验证脚本 - 10条测试用例覆盖5个工具
目标：工具选择准确率 >= 90%
"""
import sys
import os
import json
import time

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import AgentOrchestrator

# 10条测试用例，覆盖5个工具
TEST_CASES = [
    # 工单1 记账本（2条）
    {
        "input": "今天午饭花了35块",
        "expected_tool": "accounting",
        "description": "记账-支出"
    },
    {
        "input": "记录一笔收入8000元工资",
        "expected_tool": "accounting",
        "description": "记账-收入"
    },
    # 工单2 日程提醒（2条）
    {
        "input": "提醒我明天下午3点开会",
        "expected_tool": "schedule",
        "description": "日程-单次提醒"
    },
    {
        "input": "帮我设置每周一早上9点的例会提醒",
        "expected_tool": "schedule",
        "description": "日程-循环提醒"
    },
    # 工单3 文生图（2条）
    {
        "input": "生成一张海边日落的图片",
        "expected_tool": "image",
        "description": "文生图-风景"
    },
    {
        "input": "画一只可爱的猫咪",
        "expected_tool": "image",
        "description": "文生图-动物"
    },
    # 工单4 基金数据问答（2条）
    {
        "input": "查询易方达蓝筹精选的收益率",
        "expected_tool": "fund",
        "description": "基金-查询收益率"
    },
    {
        "input": "哪些基金最近表现好",
        "expected_tool": "fund",
        "description": "基金-推荐"
    },
    # 工单5 招股书数据问答（2条）
    {
        "input": "招股说明书中提到的募集资金用途是什么",
        "expected_tool": "prospectus",
        "description": "招股书-募集资金"
    },
    {
        "input": "华铭智能的招股说明书里有什么风险因素",
        "expected_tool": "prospectus",
        "description": "招股书-风险因素"
    },
]


def run_accuracy_test():
    """运行准确率测试"""
    print("=" * 60)
    print("工单6 智能体 - 工具选择准确率测试")
    print("工单编号：人工智能NLP-Agent数字人项目-智能体任务")
    print("=" * 60)
    print()

    # 初始化编排器
    print("[初始化] 正在加载Agent编排器...")
    try:
        agent = AgentOrchestrator()
        print("[初始化] Agent编排器加载成功")
    except Exception as e:
        print(f"[错误] Agent编排器加载失败: {e}")
        print("[提示] 请确保 .env 文件中的 API Key 已正确配置")
        return

    print()
    print("-" * 60)
    print(f"共 {len(TEST_CASES)} 条测试用例")
    print("-" * 60)
    print()

    correct = 0
    total = len(TEST_CASES)
    results = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"[测试 {i}/{total}] {case['description']}")
        print(f"  输入: {case['input']}")
        print(f"  期望工具: {case['expected_tool']}")

        try:
            # 调用编排器解析意图
            start_time = time.time()
            result = agent.parse_intent(case["input"])
            elapsed = time.time() - start_time

            actual_tool = result.get("tool", "unknown")
            is_correct = actual_tool == case["expected_tool"]

            if is_correct:
                correct += 1
                status = "✓ 正确"
            else:
                status = f"✗ 错误 (实际: {actual_tool})"

            print(f"  实际工具: {actual_tool}")
            print(f"  结果: {status}")
            print(f"  耗时: {elapsed:.2f}秒")

            results.append({
                "case": case["description"],
                "input": case["input"],
                "expected": case["expected_tool"],
                "actual": actual_tool,
                "correct": is_correct,
                "time": elapsed
            })

        except Exception as e:
            print(f"  结果: ✗ 异常 ({e})")
            results.append({
                "case": case["description"],
                "input": case["input"],
                "expected": case["expected_tool"],
                "actual": "error",
                "correct": False,
                "time": 0
            })

        print()

    # 汇总结果
    accuracy = correct / total * 100
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"总用例数: {total}")
    print(f"正确数:   {correct}")
    print(f"错误数:   {total - correct}")
    print(f"准确率:   {accuracy:.1f}%")
    print()

    if accuracy >= 90:
        print("[通过] 准确率 >= 90%，满足验收标准")
    else:
        print("[未通过] 准确率 < 90%，需要优化提示词或路由逻辑")

    # 输出详细结果
    print()
    print("-" * 60)
    print("详细结果:")
    print("-" * 60)
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} {r['case']}: {r['expected']} -> {r['actual']} ({r['time']:.2f}s)")

    print()
    return accuracy >= 90


if __name__ == "__main__":
    success = run_accuracy_test()
    sys.exit(0 if success else 1)
