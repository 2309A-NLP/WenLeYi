#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据加载功能
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from last_duan.utils import _find_role_dataset_files, _load_texts_from_dataset, _get_role_data_mapping

def test_role_data_mapping():
    """测试角色与数据集的映射"""
    print("=== 测试角色与数据集映射 ===")
    mapping = _get_role_data_mapping()
    
    for role_name, files in mapping.items():
        print(f"\n角色: {role_name}")
        print(f"  数据集文件:")
        for f in files:
            full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f)
            exists = "OK" if os.path.exists(full_path) else "NO"
            print(f"    [{exists}] {f}")

def test_find_role_dataset_files():
    """测试查找角色数据集文件"""
    print("\n=== 测试查找角色数据集文件 ===")
    
    roles = ["doctor", "teacher", "psychologist", "stock_analyst", "financial_advisor"]
    
    for role in roles:
        files = _find_role_dataset_files(role)
        print(f"\n角色 '{role}' 找到的文件:")
        if files:
            for f in files:
                print(f"    [OK] {os.path.basename(f)}")
        else:
            print("    [NO] 未找到")

def test_load_dataset():
    """测试加载数据集"""
    print("\n=== 测试加载数据集 ===")
    
    test_cases = [
        ("medical dataset/medical.json", "医疗数据"),
        ("Flash_distill ls/Python编程基础_merged.json", "Python编程基础"),
        ("financial_advisor_qa.csv", "金融问答"),
    ]
    
    for rel_path, desc in test_cases:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", rel_path)
        if not os.path.exists(full_path):
            print(f"\n{desc}: ❌ 文件不存在")
            continue
        
        print(f"\n{desc}:")
        texts = _load_texts_from_dataset(full_path)
        print(f"  加载到 {len(texts)} 条数据")
        if texts:
            print(f"  第一条数据预览:")
            first_text = texts[0][:200] + "..." if len(texts[0]) > 200 else texts[0]
            print(f"    {first_text}")

if __name__ == "__main__":
    test_role_data_mapping()
    test_find_role_dataset_files()
    test_load_dataset()
    print("\n=== 测试完成 ===")