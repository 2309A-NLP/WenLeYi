#!/usr/bin/env python3
"""在app.py完整环境下测试VectorStore搜索速度"""
import sys, os, io, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from core.config import Config
config = Config()

# 模拟app.py的完整初始化流程
print("=" * 50)
print("模拟app.py完整初始化")
print("=" * 50)

# 1. VectorStore初始化
from core.vector_store import VectorStore, QuestionStore
t0 = time.time()
vector_store = VectorStore(
    milvus_host=config.MILVUS_HOST,
    milvus_port=config.MILVUS_PORT,
    collection_name=config.MILVUS_COLLECTION,
    embedding_model_path=config.EMBEDDING_MODEL_PATH,
)
print(f"VectorStore初始化: {time.time()-t0:.2f}s")

# 2. load_chunks
t1 = time.time()
vector_store.load_chunks()
chunks = vector_store._chunks
print(f"load_chunks: {time.time()-t1:.2f}s, chunks数: {len(chunks)}")

# 3. QuestionStore初始化（使用MilvusClient，可能干扰gRPC）
t2 = time.time()
question_store = QuestionStore(
    milvus_host=config.MILVUS_HOST,
    milvus_port=config.MILVUS_PORT,
    collection_name="qa_history",
    embedding_model_path=config.EMBEDDING_MODEL_PATH,
)
print(f"QuestionStore初始化: {time.time()-t2:.2f}s")

# 4. 测试VectorStore搜索
print("\n" + "=" * 50)
print("测试VectorStore搜索")
print("=" * 50)
for i in range(3):
    t = time.time()
    results = vector_store.search("公司的组织架构是什么？", top_k=5)
    print(f"第{i+1}次搜索: {time.time()-t:.4f}s, 结果数: {len(results)}")
