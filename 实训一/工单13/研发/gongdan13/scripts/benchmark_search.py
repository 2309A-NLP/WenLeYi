#!/usr/bin/env python3
"""排查向量搜索慢的原因 — 拆开计时"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config

config = Config()

# 1. 测试 Embedding 编码速度
print("=" * 50)
print("1. 测试 Embedding 编码速度")
print("=" * 50)
t0 = time.time()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(config.EMBEDDING_MODEL_PATH)
t1 = time.time()
print(f"模型加载: {t1-t0:.3f}s")

query = "公司的组织架构是什么？"
t2 = time.time()
import numpy as np
q_vec = model.encode([query], normalize_embeddings=True)
q_vec = np.array(q_vec, dtype=np.float32)
q_vec = np.nan_to_num(q_vec, nan=0.0)
q_list = [float(x) for x in q_vec.flatten().tolist()]
t3 = time.time()
print(f"Query编码: {t3-t2:.3f}s")
print(f"向量维度: {len(q_list)}")

# 2. 测试 Milvus 搜索速度
print("\n" + "=" * 50)
print("2. 测试 Milvus 搜索速度")
print("=" * 50)
from pymilvus import connections, Collection
connections.connect('default', host='127.0.0.1', port='19530')
col = Collection(config.MILVUS_COLLECTION)
print(f"Collection: {config.MILVUS_COLLECTION}, 数据量: {col.num_entities}")

# 检查索引信息
print(f"索引类型: {col.index().params}")

# 预热（第一次搜索可能慢）
t4 = time.time()
results = col.search(
    data=[q_list],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
    limit=5,
    output_fields=["text", "source", "chunk_id"],
)
t5 = time.time()
print(f"第一次搜索(含加载): {t5-t4:.3f}s, 结果数: {len(results[0])}")

# 第二次搜索（索引已加载到内存）
t6 = time.time()
results = col.search(
    data=[q_list],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
    limit=5,
    output_fields=["text", "source", "chunk_id"],
)
t7 = time.time()
print(f"第二次搜索(已加载): {t7-t6:.3f}s, 结果数: {len(results[0])}")

# 第三次搜索（确认稳定）
t8 = time.time()
results = col.search(
    data=[q_list],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
    limit=5,
    output_fields=["text", "source", "chunk_id"],
)
t9 = time.time()
print(f"第三次搜索(稳定): {t9-t8:.3f}s, 结果数: {len(results[0])}")

# 3. 总结
print("\n" + "=" * 50)
print("3. 总结")
print("=" * 50)
print(f"Embedding编码: {t3-t2:.3f}s")
print(f"Milvus搜索(稳定): {t9-t8:.3f}s")
print(f"总耗时: {(t3-t2)+(t9-t8):.3f}s")
print(f"对比: 之前是19.09s, 现在是{(t3-t2)+(t9-t8):.3f}s")
