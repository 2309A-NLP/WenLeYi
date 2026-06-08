#!/usr/bin/env python3
"""RAG 性能基准测试脚本

功能：
  1. 内置测试问题（覆盖不同类型）
  2. 启动阶段预热本地模型和向量库
  3. 每次计时前清除该问题的进程内缓存，测试真实完整回答耗时
  4. 输出汇总报告
  5. 保存结果到 benchmark_results.json

用法：
  python scripts/benchmark.py                    # 默认3轮
  python scripts/benchmark.py --rounds 5         # 指定轮数
  python scripts/benchmark.py --save before      # 保存为优化前基准
  python scripts/benchmark.py --save after       # 保存为优化后基准
"""
import sys
import os
import json
import time
import argparse
from datetime import datetime

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================================
# 测试问题集
# =====================================================================
TEST_QUESTIONS = [
    "公司的组织架构是什么？",
    "销售部有哪些部门？",
    "大客户销售部由哪些销售处构成？",
    "公司的主要业务有哪些？",
    "2024年的营业收入是多少？",
    "公司的核心竞争力是什么？",
    "公司的风险管理措施有哪些？",
    "公司的研发投入情况如何？",
    "公司的客户群体主要有哪些？",
    "公司的未来发展战略是什么？",
]


def run_benchmark(rounds=3):
    """执行基准测试"""
    from core.config import Config
    from core.vector_store import VectorStore
    from core.retriever import HybridRetriever
    from core.llm import LLMClient
    from core.reranker import Reranker
    from core.rag import RAGPipeline

    config = Config()

    print("=" * 60)
    print("RAG 性能基准测试")
    print("=" * 60)
    print(f"LLM模型: {config.LLM_MODEL}")
    print(f"检索模式: {config.SEARCH_MODE}")
    print(f"测试问题数: {len(TEST_QUESTIONS)}")
    print(f"每问题测试轮数: {rounds}")
    print("=" * 60)

    # 初始化组件
    print("\n[初始化] 加载向量存储 ...")
    vector_store = VectorStore(
        milvus_host=config.MILVUS_HOST,
        milvus_port=config.MILVUS_PORT,
        collection_name=config.MILVUS_COLLECTION,
        embedding_model_path=config.EMBEDDING_MODEL_PATH,
    )
    vector_store.load_chunks()
    chunks = vector_store._chunks if hasattr(vector_store, "_chunks") else []
    print(f"[初始化] 向量存储加载完成, {len(chunks)} 个文档块")

    print("[初始化] 预热向量检索 ...")
    warmup_stats = vector_store.warmup(TEST_QUESTIONS[0])
    print(f"[初始化] 向量检索预热完成: {warmup_stats}")

    print("[初始化] 加载检索器 ...")
    retriever = HybridRetriever(vector_store, chunks, config)
    print("[初始化] 检索器加载完成")

    print("[初始化] 加载 LLM 客户端 ...")
    llm = LLMClient(config.LLM_API_KEY, config.LLM_API_BASE, config.LLM_MODEL, config.LLM_TIMEOUT)
    print("[初始化] LLM 客户端加载完成")

    reranker = None
    if config.ENABLE_RERANKER and config.RERANKER_MODEL_PATH:
        print("[初始化] 加载 Reranker ...")
        try:
            reranker = Reranker(config.RERANKER_MODEL_PATH)
            warmup_stats = reranker.warmup()
            print(f"[初始化] Reranker 预热完成: {warmup_stats}")
        except ImportError:
            print("[初始化] sentence_transformers 未安装, 跳过 Reranker")
            reranker = None
        except Exception as e:
            print(f"[初始化] Reranker 加载失败: {e}, 跳过 Reranker")
            reranker = None

    rag = RAGPipeline(retriever, llm, config, reranker)
    print("[初始化] RAG Pipeline 加载完成")
    print("=" * 60)

    # 执行测试
    all_results = []
    for q_idx, question in enumerate(TEST_QUESTIONS):
        print(f"\n--- 问题 {q_idx + 1}/{len(TEST_QUESTIONS)}: {question} ---")
        question_results = []

        for r in range(rounds):
            print(f"  轮次 {r + 1}/{rounds} ... ", end="", flush=True)
            # benchmark 测试真实完整回答耗时，不测缓存命中耗时。
            rag._answer_cache.pop(question.strip(), None)
            start = time.time()
            answer = rag.query(question, use_history=False, stream=False)
            elapsed = time.time() - start
            retrieve_time = rag._last_retrieve_time  # 获取检索耗时
            question_results.append({
                "question": question,
                "total": round(elapsed, 3),
                "retrieve_time": round(retrieve_time, 3),  # 记录检索耗时
                "answer_length": len(answer) if answer else 0,
            })
            if r + 1 < rounds:
                time.sleep(0.5)
            print(f"完成 (检索{retrieve_time:.2f}s, 总耗时{elapsed:.2f}s)")

        all_results.append(question_results)

    # 汇总报告
    print("\n" + "=" * 60)
    print("性能测试报告")
    print("=" * 60)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_model": config.LLM_MODEL,
            "search_mode": config.SEARCH_MODE,
            "reranker_enabled": config.ENABLE_RERANKER,
            "neo4j_enabled": config.ENABLE_NEO4J,
            "top_k": config.TOP_K,
            "rerank_top_k": config.RERANK_TOP_K,
            "faiss_top_k": config.FAISS_TOP_K,
            "bm25_top_k": config.BM25_TOP_K,
        },
        "questions": [],
        "overall": {},
    }

    totals = []
    retrieve_times = []
    for i, question in enumerate(TEST_QUESTIONS):
        q_rounds = all_results[i]
        avg_seconds = sum(item["total"] for item in q_rounds) / len(q_rounds)
        avg_retrieve = sum(item["retrieve_time"] for item in q_rounds) / len(q_rounds)
        answer_length = q_rounds[-1]["answer_length"]
        totals.append(avg_seconds)
        retrieve_times.append(avg_retrieve)
        summary["questions"].append({
            "question": question,
            "total_seconds": round(avg_seconds, 3),
            "retrieve_seconds": round(avg_retrieve, 3),
            "rounds": q_rounds,
            "answer_length": answer_length,
        })
        status = "OK" if avg_retrieve <= 1 else "SLOW"
        print(f"{status} {question[:30]:<30s} 检索{avg_retrieve:.2f}s | 总耗时{avg_seconds:.2f}s")

    avg_total = sum(totals) / len(totals)
    min_total = min(totals)
    max_total = max(totals)
    avg_retrieve = sum(retrieve_times) / len(retrieve_times)
    min_retrieve = min(retrieve_times)
    max_retrieve = max(retrieve_times)
    passed = sum(1 for t in retrieve_times if t <= 1)

    summary["overall"] = {
        "avg_seconds": round(avg_total, 3),
        "min_seconds": round(min_total, 3),
        "max_seconds": round(max_total, 3),
        "avg_retrieve": round(avg_retrieve, 3),
        "min_retrieve": round(min_retrieve, 3),
        "max_retrieve": round(max_retrieve, 3),
        "pass_rate": f"{passed}/{len(retrieve_times)}",
        "target_seconds": 1,
    }

    print("-" * 60)
    print(f"检索平均耗时: {avg_retrieve:.2f}s")
    print(f"检索最快: {min_retrieve:.2f}s")
    print(f"检索最慢: {max_retrieve:.2f}s")
    print(f"总平均耗时: {avg_total:.2f}s")
    print(f"达标率: {passed}/{len(retrieve_times)} (检索目标≤1s)")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="RAG 性能基准测试")
    parser.add_argument("--rounds", type=int, default=3, help="每个问题测试轮数 (默认3)")
    parser.add_argument("--save", type=str, choices=["before", "after"], help="保存结果标签 (before=优化前, after=优化后)")
    args = parser.parse_args()

    summary = run_benchmark(rounds=args.rounds)

    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark")
    os.makedirs(output_dir, exist_ok=True)

    if args.save:
        filename = f"benchmark_{args.save}.json"
    else:
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {filepath}")


if __name__ == "__main__":
    main()
