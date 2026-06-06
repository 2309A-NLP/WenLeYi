"""构建索引脚本 -- 支持全量构建和增量更新。

用法:
    python scripts/build_index.py              # 全量构建索引
    python scripts/build_index.py -i           # 增量更新 (只索引新文件)
    python scripts/build_index.py -r           # 强制全量重建索引
"""
import os
import sys
import argparse
import time

# 将项目根目录加入 Python 搜索路径，确保能正确导入 core 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.document_processor import process_documents
from core.vector_store import VectorStore
from core.graph_builder import GraphBuilder
from core.graph_store import Neo4jGraphStore


def main():
    """主函数: 解析参数 -> 加载配置 -> 处理文档 -> 构建索引"""
    # ========== 第一步: 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description="RAG 索引构建工具")
    parser.add_argument(
        "--incremental", "-i",
        action="store_true",
        help="增量更新（只索引新文件）",
    )
    parser.add_argument(
        "--rebuild", "-r",
        action="store_true",
        help="强制全量重建索引",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="强制构建 Neo4j 图谱（即使 ENABLE_NEO4J=0）",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="跳过 Neo4j 图谱构建",
    )
    parser.add_argument(
        "--clear-graph",
        action="store_true",
        help="构建前清空本项目已写入的 Neo4j 图谱",
    )
    parser.add_argument(
        "--graph-max-chunks",
        type=int,
        default=0,
        help="限制参与图谱构建的 chunk 数，0 表示不限制",
    )
    args = parser.parse_args()

    # 记录脚本启动时间，用于最终计算总耗时
    start_time = time.time()

    # ========== 脚本启动日志: 显示参数和配置 ==========
    print("[BUILD] 脚本启动")
    print(f"[BUILD]   命令行参数: incremental={args.incremental}, rebuild={args.rebuild}")

    # ========== 加载项目配置 ==========
    config = Config()
    print("[BUILD]   加载配置完成:")
    print(f"[BUILD]     DOCS_DIR           = {config.DOCS_DIR}")
    print(f"[BUILD]     CHUNK_SIZE         = {config.CHUNK_SIZE}")
    print(f"[BUILD]     CHUNK_OVERLAP      = {config.CHUNK_OVERLAP}")
    print(f"[BUILD]     MILVUS_HOST        = {config.MILVUS_HOST}")
    print(f"[BUILD]     MILVUS_PORT        = {config.MILVUS_PORT}")
    print(f"[BUILD]     MILVUS_COLLECTION  = {config.MILVUS_COLLECTION}")
    print(f"[BUILD]     EMBEDDING_MODEL    = {config.EMBEDDING_MODEL_PATH}")
    print(f"[BUILD]     VECTOR_STORE_DIR   = {config.VECTOR_STORE_DIR}")
    print(f"[BUILD]     ENABLE_NEO4J       = {config.ENABLE_NEO4J}")
    print(f"[BUILD]     NEO4J_URI          = {config.NEO4J_URI}")

    # ========== 检查文档目录是否存在 ==========
    print(f"[BUILD] 检查文档目录: {config.DOCS_DIR}")
    if os.path.isdir(config.DOCS_DIR):
        # 统计目录下文件数量
        doc_files = os.listdir(config.DOCS_DIR)
        print(f"[BUILD]   文档目录存在，包含 {len(doc_files)} 个文件/子目录")
    else:
        print(f"[BUILD]   文档目录不存在: {config.DOCS_DIR}")
        print("[BUILD]   [ERROR] 无法继续，请先创建文档目录并放入文件。")
        return

    # ========== 第二步: 解析文档并生成文本块 ==========
    process_start = time.time()
    print("[BUILD] [1/2] 开始处理文档...")
    chunks = process_documents(
        config.DOCS_DIR,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        max_workers=config.DOC_PROCESS_WORKERS,
    )
    process_done = time.time()
    process_elapsed = process_done - process_start

    # 检查是否有文档被成功处理
    if not chunks:
        print("[BUILD] [ERROR] documents/ 目录下没有找到文档，请先放入文件。")
        return

    # 收集所有被处理过的文件名 (从 chunks 的元数据中提取)
    processed_files = set()
    for chunk in chunks:
        # chunk 是字典，通常包含 metadata 信息
        if isinstance(chunk, dict) and "metadata" in chunk:
            source = chunk["metadata"].get("source", "")
            if source:
                processed_files.add(source)
        elif isinstance(chunk, dict) and "source" in chunk:
            processed_files.add(chunk["source"])

    print(f"[BUILD] 文档处理完成 (耗时 {process_elapsed:.2f}s):")
    print(f"[BUILD]   生成文本块数: {len(chunks)}")
    if processed_files:
        print(f"[BUILD]   处理的文件列表:")
        for f in sorted(processed_files):
            print(f"[BUILD]     - {f}")
    else:
        print(f"[BUILD]   (无法获取具体文件列表)")

    # ========== 第三步: 连接 Milvus 并构建向量索引 ==========
    print("[BUILD] 连接 Milvus 向量数据库...")
    print(f"[BUILD]   地址: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
    print(f"[BUILD]   集合名: {config.MILVUS_COLLECTION}")

    store = VectorStore(
        milvus_host=config.MILVUS_HOST,
        milvus_port=config.MILVUS_PORT,
        collection_name=config.MILVUS_COLLECTION,
        embedding_model_path=config.EMBEDDING_MODEL_PATH,
        embedding_batch_size=config.EMBEDDING_BATCH_SIZE,
        insert_batch_size=config.MILVUS_INSERT_BATCH_SIZE,
    )

    # ========== 索引构建: 根据参数选择增量或全量模式 ==========
    index_start = time.time()
    if args.incremental and store.is_ready:
        # 增量更新模式: 仅索引新增的文档块
        print("[BUILD] [2/2] 开始构建索引 (模式: 增量更新)...")
        store.build_index_incremental(chunks)
    else:
        # 全量构建模式: 重建整个索引
        print("[BUILD] [2/2] 开始构建索引 (模式: 全量构建)...")
        if args.incremental and not store.is_ready:
            print("[BUILD]   注意: 增量模式要求集合已存在，当前集合不存在，自动切换为全量构建")
        store.build_index(chunks)
    index_done = time.time()
    index_elapsed = index_done - index_start

    # ========== 索引构建完成日志 ==========
    print(f"[BUILD] 索引构建完成 (耗时 {index_elapsed:.2f}s)")

    # ========== 第四步: 可选构建 Neo4j 图谱 ==========
    graph_elapsed = 0.0
    graph_stats = {"chunks": 0, "triples": 0, "inserted": 0}
    should_build_graph = (config.ENABLE_NEO4J or args.graph) and not args.skip_graph
    if should_build_graph:
        graph_start = time.time()
        print("[BUILD] [3/3] 开始构建 Neo4j 图谱 ...")
        graph_store = Neo4jGraphStore(
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
        )
        if graph_store.is_ready:
            graph_builder = GraphBuilder(graph_store)
            graph_stats = graph_builder.build_from_chunks(
                chunks,
                clear=args.clear_graph or args.rebuild,
                max_chunks=args.graph_max_chunks or None,
            )
            graph_store.close()
        else:
            print("[BUILD] Neo4j 未就绪，跳过图谱构建")
        graph_elapsed = time.time() - graph_start
        print(f"[BUILD] Neo4j 图谱构建完成 (耗时 {graph_elapsed:.2f}s)")
    else:
        print("[BUILD] Neo4j 图谱构建跳过 (ENABLE_NEO4J=0 或 --skip-graph)")

    # ========== 最终汇总信息 ==========
    total_elapsed = time.time() - start_time
    print("=" * 50)
    print("[BUILD] 最终汇总:")
    print(f"[BUILD]   文档处理耗时: {process_elapsed:.2f}s")
    print(f"[BUILD]   索引构建耗时: {index_elapsed:.2f}s")
    print(f"[BUILD]   图谱构建耗时: {graph_elapsed:.2f}s")
    print(f"[BUILD]   总耗时:       {total_elapsed:.2f}s")
    print(f"[BUILD]   文本块数:     {len(chunks)}")
    print(f"[BUILD]   图谱三元组数: {graph_stats['triples']}")
    print(f"[BUILD]   图谱写入数:   {graph_stats['inserted']}")
    print(f"[BUILD]   索引存储位置: {config.VECTOR_STORE_DIR}")
    print(f"[BUILD]   运行 python scripts/app.py 启动服务")
    print("=" * 50)


if __name__ == "__main__":
    main()
