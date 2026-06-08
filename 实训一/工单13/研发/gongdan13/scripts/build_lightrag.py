"""独立构建 LightRAG 知识图谱（纯同步，避免事件循环冲突）。
运行方式: D:\an10-1\envs\nlp_1\python.exe scripts\build_lightrag.py
"""
import os
import sys
import json
import argparse
import shutil

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from core.config import Config
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

config = Config()

# 确保存在事件循环
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


def main():
    from core.lightrag_engine import LightRAGEngine

    parser = argparse.ArgumentParser(description="构建 LightRAG 知识图谱")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="构建前清空 lightrag_storage，避免 processing/duplicate 状态阻塞重建",
    )
    args = parser.parse_args()

    working_dir = os.path.join(PROJECT_DIR, "lightrag_storage")
    if args.reset and os.path.isdir(working_dir):
        resolved = os.path.abspath(working_dir)
        project_resolved = os.path.abspath(PROJECT_DIR)
        if not resolved.startswith(project_resolved):
            raise RuntimeError(f"拒绝清理项目目录外的路径: {resolved}")
        print(f"[LIGHTRAG] 清空旧存储: {resolved}")
        shutil.rmtree(resolved)

    # 初始化引擎
    engine = LightRAGEngine(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        model=config.LLM_MODEL,
        working_dir=working_dir,
    )

    # 索引两个 PDF
    docs_dir = config.DOCS_DIR
    pdf_files = [
        os.path.join(docs_dir, "招股说明书1.pdf"),
        os.path.join(docs_dir, "招股说明书2.pdf"),
    ]

    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"文件不存在: {pdf_path}")
            continue
        print(f"\n{'='*60}")
        print(f"开始索引: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")
        result = engine.insert_pdf(pdf_path)
        print(f"结果: {result}")

    # 导出知识图谱
    graph_path = os.path.join(engine.working_dir, "knowledge_graph.json")
    engine.save_graph_to_file(graph_path)

    # 打印图谱统计
    graph_data = engine.export_knowledge_graph()
    entity_count = len(graph_data.get('entities', []))
    relation_count = len(graph_data.get('relations', []))
    print(f"\n{'='*60}")
    print(f"构建完成!")
    print(f"实体数量: {entity_count}")
    print(f"关系数量: {relation_count}")
    print(f"图谱文件: {graph_path}")
    if entity_count == 0 or relation_count == 0:
        print("[WARN] 知识图谱为空。请检查 LLM/Embedding API 是否可用，必要时使用 --reset 重建。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
