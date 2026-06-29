# -*- coding: utf-8 -*-
"""导入金融知识库到Milvus（在Windows环境运行）"""

import json
import os
import sys
import time
from transformers import AutoModel, AutoTokenizer
import torch
from pymilvus import connections, Collection, utility

# ===== 配置 =====
DATA_DIR = r"D:\桌面\ai对话系统\金融对话系统\数据"
MODEL_PATH = r"D:\桌面\模型\m3e-base"
COLLECTION_NAME = "financial_knowledge"
BATCH_SIZE = 100  # 每次插入100条

def load_all_items():
    """加载所有问答对"""
    all_items = []
    files = [
        ("金融知识问答数据集.json", "json"),
        ("博金杯问答_基金.json", "json"),
        ("金融题目与答案.json", "json"),
        ("博金杯数据库知识.json", "json"),
        ("博金杯数据问答.json", "json"),
        ("博金杯数据问答_持仓.json", "json"),
    ]
    for fname, fmt in files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cat, items in data.get("categories", {}).items():
            for item in items:
                all_items.append({
                    "question": item.get("q", ""),
                    "answer": item.get("a", ""),
                    "category": item.get("category", cat),
                    "source": item.get("source", fname),
                })
    # 去重
    seen = set()
    unique = []
    for item in all_items:
        key = item["question"][:50]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    print(f"共 {len(unique)} 条唯一问答对")
    return unique


def main():
    t0 = time.time()

    # 1. 连接Milvus
    connections.connect(host="127.0.0.1", port=19530)
    if not utility.has_collection(COLLECTION_NAME):
        print(f"集合 {COLLECTION_NAME} 不存在，请先创建")
        return

    col = Collection(COLLECTION_NAME)
    col.load()

    # 2. 加载模型
    print("加载嵌入模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()
    print(f"模型加载完成 ({time.time()-t0:.1f}s)")

    # 3. 加载数据
    items = load_all_items()

    # 4. 批量编码+插入
    def embed(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded)
        return output.last_hidden_state[:, 0, :].numpy()

    inserted = 0
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i+BATCH_SIZE]
        questions = [item["question"] for item in batch]

        # 编码
        vectors = embed(questions)

        # 截断过长的字段
        entities = [
            [vectors[j].tolist() for j in range(len(batch))],
            [batch[j]["question"][:1900] for j in range(len(batch))],
            [batch[j]["answer"][:4900] for j in range(len(batch))],
            [batch[j]["category"][:190] for j in range(len(batch))],
            [batch[j]["source"][:190] for j in range(len(batch))],
        ]
        try:
            col.insert(entities)
            inserted += len(batch)
        except Exception as e:
            print(f"  第{i}批插入失败: {e}")
            # 逐条插入
            for j in range(len(batch)):
                try:
                    col.insert([[vectors[j].tolist()], [batch[j]["question"][:1900]], [batch[j]["answer"][:4900]], [batch[j]["category"][:190]], [batch[j]["source"][:190]]])
                    inserted += 1
                except:
                    pass

        if (i + BATCH_SIZE) % 1000 == 0 or (i + BATCH_SIZE) >= len(items):
            print(f"  已导入 {inserted}/{len(items)} 条 ({time.time()-t0:.1f}s)")

    # 5. 刷新
    col.flush()
    print(f"\n导入完成！共 {inserted} 条，耗时 {time.time()-t0:.1f} 秒")
    print(f"集合 {COLLECTION_NAME} 行数: {col.num_entities}")


if __name__ == "__main__":
    main()
