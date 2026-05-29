#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强制重新加载角色知识库数据脚本

用途：
- 将 data/ 目录中的新数据重新导入到 Milvus
- 支持清空现有数据后重新加载
- 支持指定角色加载

使用方法：
python reload_knowledge.py                    # 重新加载所有角色
python reload_knowledge.py --role financial_advisor  # 只加载指定角色
python reload_knowledge.py --clear           # 先清空再加载
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'peizhi_huanjing', '.env')
load_dotenv(env_path)

from last_duan.database import (
    init_database, init_milvus, ensure_knowledge_collection,
    add_documents_to_milvus, _load_texts_from_dataset,
    get_db_connection
)
from last_duan.utils import ROLE_DISPLAY_NAMES, _data_dir

PROJECT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def _find_role_dataset_files(role_name: str) -> list:
    """查找与角色名相关的数据集文件（使用项目根目录的data目录）"""
    base = PROJECT_DATA_DIR
    role_lower = (role_name or "").lower()
    candidates = []
    try:
        for fn in os.listdir(base):
            lower = fn.lower()
            if not (lower.endswith(".csv") or lower.endswith(".txt")):
                continue
            if role_lower and role_lower in lower:
                candidates.append(os.path.join(base, fn))
    except Exception:
        candidates = []

    if candidates:
        return sorted(candidates)

    fallback = os.path.join(base, "financial_qa.csv")
    if os.path.exists(fallback):
        return [fallback]
    
    general_fallback = os.path.join(base, "financial_advisor_qa.csv")
    if os.path.exists(general_fallback):
        return [general_fallback]
    return []


def load_role_knowledge(role_id: int, role_name: str, clear_first: bool = False):
    """加载指定角色的知识库"""
    print(f"\n📚 处理角色: {role_name} (ID: {role_id})")
    
    col = ensure_knowledge_collection(role_id)
    
    # 清空现有数据
    if clear_first:
        if col.num_entities > 0:
            print(f"   清空现有数据 ({col.num_entities} 条)...")
            col.delete("id >= 0")
            col.flush()
    
    # 查找数据集文件
    dataset_files = _find_role_dataset_files(role_name)
    if not dataset_files:
        print(f"   ❌ 未找到相关数据集文件")
        return
    
    print(f"   找到 {len(dataset_files)} 个数据集文件:")
    for p in dataset_files:
        print(f"      - {os.path.basename(p)}")
    
    # 加载数据
    all_texts = []
    for p in dataset_files:
        try:
            texts = _load_texts_from_dataset(p)
            all_texts.extend(texts)
            print(f"      ✓ 加载成功: {len(texts)} 条")
        except Exception as e:
            print(f"      ✗ 加载失败: {e}")
    
    if not all_texts:
        print(f"   ❌ 未加载到任何数据")
        return
    
    # 添加到 Milvus
    print(f"   添加 {len(all_texts)} 条记录到 Milvus...")
    add_documents_to_milvus(
        texts=all_texts,
        source="dataset:reload",
        doc_id=f"reload_{role_name}",
        role_id=role_id
    )
    
    col.load()
    print(f"   ✅ 完成，当前数据量: {col.num_entities} 条")


def main():
    parser = argparse.ArgumentParser(description="重新加载角色知识库")
    parser.add_argument("--role", type=str, help="指定角色名（如 financial_advisor）")
    parser.add_argument("--clear", action="store_true", help="先清空现有数据")
    args = parser.parse_args()
    
    print("🚀 初始化数据库连接...")
    init_database()
    init_milvus()
    
    if args.role:
        # 只加载指定角色
        role_name = args.role.lower()
        for role_id, name in enumerate(ROLE_DISPLAY_NAMES.keys(), 1):
            if name.lower() == role_name or ROLE_DISPLAY_NAMES[name].lower() == role_name:
                load_role_knowledge(role_id, name, args.clear)
                return
        print(f"❌ 未找到角色: {args.role}")
        print(f"可用角色: {', '.join(ROLE_DISPLAY_NAMES.keys())}")
    else:
        # 加载所有角色
        print(f"📋 准备重新加载 {len(ROLE_DISPLAY_NAMES)} 个角色的知识库")
        for role_id, role_name in enumerate(ROLE_DISPLAY_NAMES.keys(), 1):
            load_role_knowledge(role_id, role_name, args.clear)
    
    print("\n🎉 知识库重新加载完成！")


if __name__ == "__main__":
    main()