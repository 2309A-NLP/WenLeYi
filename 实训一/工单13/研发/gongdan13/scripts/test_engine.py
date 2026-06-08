"""测试 LightRAG 引擎是否能正常初始化。"""
import sys
sys.path.insert(0, ".")
from core.lightrag_engine import LightRAGEngine

engine = LightRAGEngine(
    api_key="test",
    base_url="https://api.xiaomimimo.com/v1",
    model="mimo-v2.5",
    working_dir="./lightrag_storage",
    embedding_model_path="",
)
print("LightRAG 引擎初始化成功!")
