#!/usr/bin/env python3
"""直接测试Milvus搜索速度"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
config = Config()

# 1. 测试直接用pymilvus搜索（和benchmark_search.py一样）
print("=" * 50)
print("1. 直接用pymilvus搜索")
print("=" * 50)

from pymilvus import connections, Collection
import numpy as np

connections.connect('default', host=config.MILVUS_HOST, port=config.MILVUS_PORT)
col = Collection(config.MILVUS_COLLECTION)

# 加载模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(config.EMBEDDING_MODEL_PATH)
q_vec = model.encode(["公司的组织架构是什么？"], normalize_embeddings=True)
q_list = [float(x) for x in np.array(q_vec, dtype=np.float32).flatten().tolist()]

# 搜索3次
for i in range(3):
    t = time.time()
    results = col.search(
        data=[q_list],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=5,
        output_fields=["text", "source", "chunk_id"],
    )
    print(f"第{i+1}次搜索: {time.time()-t:.4f}s, 结果数: {len(results[0])}")

# 2. 测试通过VectorStore搜索
print("\n" + "=" * 50)
print("2. 通过VectorStore搜索")
print("=" * 50)

from core.vector_store import VectorStore
v = VectorStore(
    milvus_host=config.MILVUS_HOST,
    milvus_port=config.MILVUS_PORT,
    collection_name=config.MILVUS_COLLECTION,
    embedding_model_path=config.EMBEDDING_MODEL_PATH,
)
v.load_chunks()

for i in range(3):
    t = time.time()
    results = v.search("公司的组织架构是什么？", top_k=5)
    print(f"第{i+1}次搜索: {time.time()-t:.4f}s, 结果数: {len(results)}")
