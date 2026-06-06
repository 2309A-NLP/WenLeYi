"""只构建 Neo4j 图谱，不碰 Milvus 索引。从 Milvus 读取已有 chunks，提取三元组写入 Neo4j。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.vector_store import VectorStore
from core.graph_store import Neo4jGraphStore
from core.graph_builder import GraphBuilder

config = Config()

# 1. 从 Milvus 读取已有 chunks（只读，不重建）
print("[BUILD-GRAPH] 从 Milvus 读取已有 chunks ...")
vs = VectorStore(
    milvus_host=config.MILVUS_HOST,
    milvus_port=config.MILVUS_PORT,
    collection_name=config.MILVUS_COLLECTION,
    embedding_model_path=config.EMBEDDING_MODEL_PATH,
)
if not vs.is_ready:
    print("[ERROR] Milvus 集合不存在或为空")
    sys.exit(1)
vs.load_chunks()
chunks = vs._chunks
print(f"[BUILD-GRAPH] 读取到 {len(chunks)} 个 chunks")

# 2. 连接 Neo4j 并构建图谱
print("[BUILD-GRAPH] 连接 Neo4j ...")
gs = Neo4jGraphStore(
    uri=config.NEO4J_URI,
    user=config.NEO4J_USER,
    password=config.NEO4J_PASSWORD,
)
if not gs.is_ready:
    print("[ERROR] Neo4j 连接失败")
    sys.exit(1)

builder = GraphBuilder(gs)
stats = builder.build_from_chunks(chunks, clear=True)
print(f"[BUILD-GRAPH] 完成: {stats}")

gs.close()
print("[BUILD-GRAPH] 图谱构建完毕，现在可以跑 graph_visualization.py")
