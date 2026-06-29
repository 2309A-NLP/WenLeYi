# -*- coding: utf-8 -*-
"""
金融知识库管理模块
负责从多个JSON/TXT/MD文件中加载金融知识数据，统一管理所有问答对
数据来源包括：金融知识问答数据集、术语词典、A股规则、博金杯数据等
"""

import json
import os
import re
import logging
import time

logger = logging.getLogger(__name__)

# 知识数据目录路径（金融数字人/金融RAG/数据/）
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "数据")

# 缓存机制：避免每次请求都重新加载5447条知识
_knowledge_cache = None       # 缓存的知识库数据
_last_load_time = 0           # 上次加载的时间戳
CACHE_TTL = 300               # 缓存有效期：300秒（5分钟）


def _parse_terms_txt():
    """
    解析 金融术语词典.txt 为问答对
    格式：【术语名】 术语解释
    例如：【市盈率】 股价与每股收益的比率，用于衡量股票估值水平
    """
    txt_path = os.path.join(_DATA_DIR, "金融术语词典.txt")
    if not os.path.exists(txt_path):
        return []
    items = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 用正则匹配 【xxx】 xxx 的格式
            m = re.match(r'【(.+?)】\s*(.+)$', line)
            if m:
                term = m.group(1)       # 术语名
                definition = m.group(2)  # 术语解释
                # 自动生成问题："什么是市盈率？"
                items.append({
                    "q": f"什么是{term}？",
                    "a": definition,
                    "category": "术语词典",
                    "source": "金融术语词典"
                })
    logger.info(f"从术语词典解析到 {len(items)} 条")
    return items


def _parse_rules_md():
    """
    解析 A股交易规则手册.md 为问答对
    Markdown文件按章节组织，提取表格和段落内容转为问答对
    """
    md_path = os.path.join(_DATA_DIR, "A股交易规则手册.md")
    if not os.path.exists(md_path):
        return []
    items = []
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按 ## 标题分割章节
    sections = re.split(r'##\s+', content)
    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split('\n')
        title = lines[0].strip()  # 章节标题
        body = '\n'.join(lines[1:]).strip()
        if body:
            # 提取Markdown表格行（|xxx|xxx|xxx|）
            table_rows = re.findall(r'\|(.+?)\|(.+?)\|(.+?)\|', body)
            for row in table_rows:
                cells = [c.strip() for c in row if c.strip()]
                if len(cells) >= 2:
                    q = f"{title}：{cells[0]}是什么？"
                    a = cells[1] if len(cells) >= 2 else ""
                    if a and len(a) > 2:
                        items.append({"q": q, "a": a, "category": f"A股规则-{title}", "source": "A股交易规则手册"})
            # 提取纯文本段落（排除表格、分隔线、引用等）
            text_lines = [l for l in body.split('\n')
                         if l.strip() and not l.startswith('|') and not l.startswith('---')
                         and len(l.strip()) > 15 and not l.startswith('>')]
            for t in text_lines[:3]:  # 每个章节最多取3段
                t = t.strip()
                items.append({"q": f"请介绍{title}", "a": t, "category": f"A股规则-{title}", "source": "A股交易规则手册"})

    logger.info(f"从A股规则解析到 {len(items)} 条")
    return items


def load_knowledge(refresh=False):
    """
    【核心函数】加载所有金融知识数据
    从9个数据源加载问答对，合并成一个大列表
    带缓存机制：5分钟内重复调用直接返回缓存，不重新加载
    """
    global _knowledge_cache, _last_load_time

    # 检查缓存是否有效（5分钟内直接用缓存）
    current_time = time.time()
    if _knowledge_cache is not None and not refresh and (current_time - _last_load_time) < CACHE_TTL:
        logger.debug("使用缓存的知识库数据")
        return _knowledge_cache

    start_time = time.time()
    logger.info("开始加载知识库...")

    all_items = []  # 所有知识条目的总列表

    # 1. 加载金融知识问答数据集（主要的问答对来源）
    json_path = os.path.join(_DATA_DIR, "金融知识问答数据集.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cat, items in data.get("categories", {}).items():
            for item in items:
                item["category"] = cat  # 给每条数据打上分类标签
                all_items.append(item)
        logger.info(f"JSON问答对: {len(data.get('categories', {}))} 个分类")

    # 2. 加载金融术语词典（"什么是市盈率？"→"市盈率是..."）
    all_items.extend(_parse_terms_txt())

    # 3. 加载A股交易规则手册（表格+段落转问答对）
    all_items.extend(_parse_rules_md())

    # 4. 加载博金杯数据库知识
    bojin_path = os.path.join(_DATA_DIR, "博金杯数据库知识.json")
    if os.path.exists(bojin_path):
        with open(bojin_path, "r", encoding="utf-8") as f:
            bojin_data = json.load(f)
        for cat, items in bojin_data.get("categories", {}).items():
            for item in items:
                item["category"] = cat
                all_items.append(item)

    # 5. 加载博金杯数据问答（从实际数据中提取的问答对）
    bojin_qa_path = os.path.join(_DATA_DIR, "博金杯数据问答.json")
    if os.path.exists(bojin_qa_path):
        with open(bojin_qa_path, "r", encoding="utf-8") as f:
            bojin_qa_data = json.load(f)
        for cat, items in bojin_qa_data.get("categories", {}).items():
            for item in items:
                item["category"] = cat
                all_items.append(item)

    # 6. 加载博金杯数据问答_持仓（基金持仓相关问答）
    bojin_hold_path = os.path.join(_DATA_DIR, "博金杯数据问答_持仓.json")
    if os.path.exists(bojin_hold_path):
        with open(bojin_hold_path, "r", encoding="utf-8") as f:
            bojin_hold_data = json.load(f)
        for cat, items in bojin_hold_data.get("categories", {}).items():
            for item in items:
                item["category"] = cat
                all_items.append(item)

    # 7. 加载博金杯问答_基金（大量基金数据问答）
    bojin_fund_path = os.path.join(_DATA_DIR, "博金杯问答_基金.json")
    if os.path.exists(bojin_fund_path):
        with open(bojin_fund_path, "r", encoding="utf-8") as f:
            bojin_fund_data = json.load(f)
        for cat, items in bojin_fund_data.get("categories", {}).items():
            for item in items:
                item["category"] = cat
                all_items.append(item)

    # 8. 加载金融题目与答案（856条金融计算题）
    exam_path = os.path.join(_DATA_DIR, "金融题目与答案.json")
    if os.path.exists(exam_path):
        with open(exam_path, "r", encoding="utf-8") as f:
            exam_data = json.load(f)
        for cat, items in exam_data.get("categories", {}).items():
            for item in items:
                item["category"] = cat
                all_items.append(item)

    # 9. 加载行情数据问答
    market_path = os.path.join(_DATA_DIR, "行情数据问答.json")
    if os.path.exists(market_path):
        with open(market_path, "r", encoding="utf-8") as f:
            market_data = json.load(f)
        for cat, items in market_data.get("categories", {}).items():
            for item in items:
                item["category"] = cat
                all_items.append(item)

    # 统计所有分类名称
    categories = list(set(item.get("category", "其他") for item in all_items))

    # 更新缓存
    _last_load_time = time.time()
    load_duration = _last_load_time - start_time

    _knowledge_cache = {
        "items": all_items,          # 全部知识条目列表
        "total": len(all_items),     # 总条数（约5447条）
        "categories": categories,    # 分类列表
    }
    logger.info(f"知识库加载完成: {_knowledge_cache['total']} 条, {len(categories)} 个分类, 耗时 {load_duration:.2f}秒")
    return _knowledge_cache


def get_all_items():
    """获取所有问答对（触发加载）"""
    kb = load_knowledge()
    return kb["items"]


def get_by_category(category):
    """按分类获取问答对"""
    kb = load_knowledge()
    return [item for item in kb["items"] if item.get("category") == category]


def get_statistics():
    """获取知识库统计信息：每个分类各有多少条"""
    kb = load_knowledge()
    stats = {"total": kb["total"], "categories": {}}
    for item in kb["items"]:
        cat = item.get("category", "其他")
        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
    return stats
