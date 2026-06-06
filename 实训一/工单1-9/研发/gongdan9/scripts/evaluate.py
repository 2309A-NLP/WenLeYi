"""评估脚本 -- 从 sample_questions.pdf 提取问题，计算 context precision / context recall / 响应时间。

用法:
    python scripts/evaluate.py                  # 运行评估并输出报告
    python scripts/evaluate.py --before         # 标记为优化前（结果保存到 results/before/）
    python scripts/evaluate.py --after          # 标记为优化后（结果保存到 results/after/）
    python scripts/evaluate.py --tag xxx        # 自定义标签
"""
import os
import sys
import json
import time
import re
import argparse
from pathlib import Path

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.vector_store import VectorStore, QuestionStore
from core.document_processor import process_documents
from core.retriever import HybridRetriever
from core.llm import LLMClient
from core.reranker import Reranker
from core.rag import RAGPipeline
from core.graph_store import Neo4jGraphStore
import pymupdf


# =====================================================================
# 1. 从 sample_questions.pdf 提取问题和参考答案
# =====================================================================
def extract_questions(pdf_path: str) -> list:
    """解析 sample_questions.pdf，提取问题和参考答案列表。

    返回: [{"question": str, "reference": str, "source_doc": str}, ...]
    """
    print(f"[EVAL] 正在解析 {pdf_path} ...")
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # 按 "问题：" 分割文本
    parts = re.split(r'问题[：:]', full_text)
    questions = []

    for part in parts[1:]:  # 跳过第一个空段
        # 提取问题文本（到 "答案" 或 "参考答案" 之前）
        q_match = re.split(r'(?:参考)?答案[：:]', part)
        if len(q_match) < 2:
            continue

        question_text = q_match[0].strip()
        # 清理问题文本中的换行和多余空格
        question_text = re.sub(r'\s+', ' ', question_text).strip()

        answer_text = q_match[1].strip()
        # 答案文本保留原始格式（用于评估）
        answer_text = re.sub(r'\n{3,}', '\n\n', answer_text).strip()

        # 提取来源文档（答案后面通常有文档名）
        source_doc = ""
        doc_match = re.search(r'(.+?\.pdf)', answer_text)
        if doc_match:
            source_doc = doc_match.group(1)
            # 从答案中移除文档名行
            answer_text = re.sub(r'\s*' + re.escape(source_doc) + r'\s*$', '', answer_text).strip()

        if question_text and answer_text:
            questions.append({
                "question": question_text,
                "reference": answer_text,
                "source_doc": source_doc,
            })

    print(f"[EVAL] 共提取 {len(questions)} 个问题")
    for i, q in enumerate(questions):
        print(f"[EVAL]   Q{i+1}: {q['question'][:80]}...")
    return questions


# =====================================================================
# 2. 初始化 RAG 系统
# =====================================================================
def init_rag_system(config):
    """初始化 RAG 系统的所有组件，返回 (rag_pipeline, retriever)"""
    print("[EVAL] 正在初始化 RAG 系统 ...")

    # 向量存储
    vector_store = VectorStore(
        milvus_host=config.MILVUS_HOST,
        milvus_port=config.MILVUS_PORT,
        collection_name=config.MILVUS_COLLECTION,
        embedding_model_path=config.EMBEDDING_MODEL_PATH,
        embedding_batch_size=config.EMBEDDING_BATCH_SIZE,
        insert_batch_size=config.MILVUS_INSERT_BATCH_SIZE,
    )

    if not vector_store.is_ready:
        print("[EVAL] [ERROR] 向量索引未就绪，请先运行 build_index.py")
        sys.exit(1)

    # 加载 chunks
    if hasattr(vector_store, "load_chunks"):
        vector_store.load_chunks()
    chunks = vector_store._chunks if hasattr(vector_store, "_chunks") else []
    print(f"[EVAL] 加载 {len(chunks)} 个文档块")

    # Neo4j 图谱
    graph_store = None
    if config.ENABLE_NEO4J:
        graph_store = Neo4jGraphStore(
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
        )
        if not graph_store.is_ready:
            graph_store = None

    # 混合检索器
    retriever = HybridRetriever(vector_store, chunks, config, graph_store=graph_store)

    # LLM
    llm = LLMClient(config.LLM_API_KEY, config.LLM_API_BASE, config.LLM_MODEL, config.LLM_TIMEOUT)

    # Reranker
    reranker = None
    if config.ENABLE_RERANKER and config.RERANKER_MODEL_PATH:
        reranker = Reranker(config.RERANKER_MODEL_PATH)

    # QA 缓存（评估时禁用，避免缓存干扰）
    question_store = None

    # RAG Pipeline
    rag = RAGPipeline(retriever, llm, config, reranker, question_store=question_store)

    print("[EVAL] RAG 系统初始化完成")
    return rag, retriever


# =====================================================================
# 3. 评估单个问题
# =====================================================================
def evaluate_single(rag, retriever, llm, question: dict, index: int) -> dict:
    """评估单个问题，返回评估结果字典。"""
    q = question["question"]
    ref = question["reference"]
    print(f"\n[EVAL] ========== 问题 {index + 1} ==========")
    print(f"[EVAL] 问题: {q[:100]}")

    # --- 检索 ---
    search_start = time.time()
    results = retriever.retrieve(q, top_k=10)
    search_time = time.time() - search_start
    print(f"[EVAL] 检索完成: {len(results)} 条结果, 耗时={search_time:.2f}s")

    # --- Reranker ---
    rerank_time = 0
    if rag.reranker and rag.config.ENABLE_RERANKER and results:
        rerank_start = time.time()
        results = rag.reranker.rerank(q, results, rag.config.RERANK_TOP_K)
        rerank_time = time.time() - rerank_start
        print(f"[EVAL] Reranker 完成: {len(results)} 条结果, 耗时={rerank_time:.2f}s")

    # 收集检索到的上下文
    retrieved_contexts = [chunk["text"] for chunk, score in results]
    context_text = "\n\n---\n\n".join(retrieved_contexts)

    # --- LLM 生成答案 ---
    # 构建 prompt
    context_for_prompt = "\n\n".join([
        f"【来源: {chunk.get('source', '未知')}】\n{chunk['text']}"
        for chunk, _ in results
    ])
    prompt = f"参考资料：\n{context_for_prompt}\n\n问题：{q}"

    gen_start = time.time()
    answer = llm.chat(
        prompt,
        system_prompt=rag.SYSTEM_PROMPT if hasattr(rag, 'SYSTEM_PROMPT') else "你是一个专业的文档问答助手。",
        temperature=0.3,
        max_tokens=2048,
    )
    gen_time = time.time() - gen_start
    total_time = search_time + rerank_time + gen_time

    print(f"[EVAL] LLM 生成完成: 长度={len(answer)}, 耗时={gen_time:.2f}s")
    print(f"[EVAL] 总响应时间: {total_time:.2f}s")
    print(f"[EVAL] 生成答案前200字: {answer[:200]}")

    return {
        "question": q,
        "reference": ref,
        "source_doc": question.get("source_doc", ""),
        "generated_answer": answer,
        "retrieved_contexts": retrieved_contexts,
        "context_text": context_text,
        "search_time": round(search_time, 3),
        "rerank_time": round(rerank_time, 3),
        "gen_time": round(gen_time, 3),
        "total_time": round(total_time, 3),
    }


# =====================================================================
# 通用分数解析：从 mimo-v2.5 的返回中提取 0~1 的分数
# =====================================================================
def _parse_score(result: str, label: str) -> float:
    """从 LLM 返回文本中提取 0~1 之间的小数分数。

    mimo-v2.5 会在 reasoning 过程中输出多个数字（如 0.0、0.5），
    最终结论通常在文本最后几行。策略：取最后 3 行中出现的数字。
    """
    # 按行分割，取最后 3 行（结论区域）
    lines = result.strip().split('\n')
    tail_text = '\n'.join(lines[-3:]) if len(lines) >= 3 else result

    # 在尾部文本中找所有 0~1 之间的小数
    numbers = re.findall(r'(?<!\d)(0?\.\d+|1\.0+)(?!\d)', tail_text)
    if numbers:
        val = float(numbers[-1])
        val = max(0.0, min(1.0, val))
        print(f"[EVAL-JUDGE] {label} 解析成功: {val}")
        return val

    # 兜底：全文找最后一个数字
    all_numbers = re.findall(r'(?<!\d)(0?\.\d+|1\.0+)(?!\d)', result)
    if all_numbers:
        val = float(all_numbers[-1])
        val = max(0.0, min(1.0, val))
        print(f"[EVAL-JUDGE] {label} 全文兜底解析: {val}")
        return val

    print(f"[EVAL-JUDGE] [WARN] {label}: 无法从返回中提取分数")
    return 0.0


# =====================================================================
# 4. LLM-as-Judge 评估 precision / recall
# =====================================================================
def judge_precision(llm, question: str, reference: str, contexts: list) -> float:
    """用 LLM 评估上下文精度（context precision）。

    precision = 检索上下文中与参考答案相关的内容比例
    """
    if not contexts:
        return 0.0

    # 将上下文编号，限制总长度避免 prompt 过长
    numbered_contexts = "\n\n".join([
        f"[片段{i+1}]: {ctx[:300]}" for i, ctx in enumerate(contexts[:8])
    ])

    prompt = f"""请评估以下检索片段与参考答案的相关性。

问题：{question}

参考答案（前500字）：{reference[:500]}

检索到的片段：
{numbered_contexts}

判断每个片段是否与参考答案内容相关，然后计算精度。
精度 = 相关片段数 / 总片段数

请只输出一个数字（0到1之间的小数），例如：0.6"""

    result = llm.chat(prompt, temperature=0.1, max_tokens=4096)
    if not result or result.strip() == "":
        print("[EVAL-JUDGE] [WARN] Precision: LLM 返回空内容")
        return 0.0
    print(f"[EVAL-JUDGE] Precision 原始返回: {result[:300]}")

    return _parse_score(result, "Precision")


def judge_recall(llm, question: str, reference: str, contexts: list) -> float:
    """用 LLM 评估上下文召回（context recall）。

    recall = 参考答案中有多少信息被检索上下文覆盖
    """
    if not contexts:
        return 0.0

    context_text = "\n\n".join([f"[片段{i+1}]: {ctx[:300]}" for i, ctx in enumerate(contexts[:8])])

    prompt = f"""请评估以下检索片段对参考答案的覆盖程度。

问题：{question}

参考答案（前500字）：{reference[:500]}

检索到的片段：
{context_text}

参考答案中有多少比例的信息点被这些检索片段覆盖了？
请只输出一个数字（0到1之间的小数），例如：0.7"""

    result = llm.chat(prompt, temperature=0.1, max_tokens=4096)
    if not result or result.strip() == "":
        print("[EVAL-JUDGE] [WARN] Recall: LLM 返回空内容")
        return 0.0
    print(f"[EVAL-JUDGE] Recall 原始返回: {result[:300]}")

    return _parse_score(result, "Recall")


# =====================================================================
# 5. 主评估流程
# =====================================================================
def run_evaluation(tag: str = ""):
    """执行完整评估流程。"""
    start_time = time.time()

    # 加载配置
    config = Config()
    config.log_config()

    # 初始化 RAG 系统
    rag, retriever = init_rag_system(config)

    # 提取问题
    sample_pdf = os.path.join(config.DOCS_DIR, "sample_questions.pdf")
    if not os.path.exists(sample_pdf):
        print(f"[EVAL] [ERROR] sample_questions.pdf 不存在: {sample_pdf}")
        sys.exit(1)
    questions = extract_questions(sample_pdf)
    if not questions:
        print("[EVAL] [ERROR] 未提取到任何问题")
        sys.exit(1)

    # 评估每个问题
    results = []
    all_search_times = []
    all_gen_times = []
    all_total_times = []
    all_precisions = []
    all_recalls = []

    for i, q in enumerate(questions):
        eval_result = evaluate_single(rag, retriever, rag.llm, q, i)

        # LLM-as-Judge 评估
        print(f"\n[EVAL-JUDGE] 评估问题 {i+1} 的 precision 和 recall ...")
        precision = judge_precision(rag.llm, q["question"], q["reference"], eval_result["retrieved_contexts"])
        recall = judge_recall(rag.llm, q["question"], q["reference"], eval_result["retrieved_contexts"])

        eval_result["context_precision"] = round(precision, 4)
        eval_result["context_recall"] = round(recall, 4)

        print(f"[EVAL] 问题 {i+1} 评估结果:")
        print(f"[EVAL]   Precision = {precision:.4f}")
        print(f"[EVAL]   Recall    = {recall:.4f}")
        print(f"[EVAL]   响应时间  = {eval_result['total_time']:.2f}s")

        results.append(eval_result)
        all_search_times.append(eval_result["search_time"])
        all_gen_times.append(eval_result["gen_time"])
        all_total_times.append(eval_result["total_time"])
        all_precisions.append(precision)
        all_recalls.append(recall)

    # 汇总统计
    total_elapsed = time.time() - start_time
    summary = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(questions),
        "avg_context_precision": round(sum(all_precisions) / len(all_precisions), 4),
        "avg_context_recall": round(sum(all_recalls) / len(all_recalls), 4),
        "avg_search_time": round(sum(all_search_times) / len(all_search_times), 3),
        "avg_gen_time": round(sum(all_gen_times) / len(all_gen_times), 3),
        "avg_total_time": round(sum(all_total_times) / len(all_total_times), 3),
        "max_total_time": round(max(all_total_times), 3),
        "precision_meets_target": sum(all_precisions) / len(all_precisions) >= 0.8,
        "recall_meets_target": sum(all_recalls) / len(all_recalls) >= 0.9,
        "time_meets_target": max(all_total_times) <= 3.0,
        "eval_duration_seconds": round(total_elapsed, 1),
        "details": results,
    }

    # 保存结果
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    filename = f"eval_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json" if tag else f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[EVAL] 结果已保存到: {filepath}")

    # 打印汇总报告
    print("\n" + "=" * 60)
    print("[EVAL] 评估汇总报告")
    print("=" * 60)
    print(f"标签: {tag or '(无)'}")
    print(f"问题数: {summary['total_questions']}")
    print(f"平均 Context Precision: {summary['avg_context_precision']:.4f} {'✅' if summary['precision_meets_target'] else '❌ (目标≥0.8)'}")
    print(f"平均 Context Recall:    {summary['avg_context_recall']:.4f} {'✅' if summary['recall_meets_target'] else '❌ (目标≥0.9)'}")
    print(f"平均响应时间:           {summary['avg_total_time']:.3f}s {'✅' if summary['time_meets_target'] else '❌ (目标≤3s)'}")
    print(f"最大响应时间:           {summary['max_total_time']:.3f}s")
    print(f"评估耗时:               {summary['eval_duration_seconds']:.1f}s")
    print("=" * 60)

    # 逐题结果
    print("\n逐题详情:")
    for i, r in enumerate(results):
        print(f"\nQ{i+1}: {r['question'][:60]}...")
        print(f"  Precision: {r['context_precision']:.4f} | Recall: {r['context_recall']:.4f} | 时间: {r['total_time']:.2f}s")

    return summary


# =====================================================================
# 6. 对比两份评估报告
# =====================================================================
def compare_reports(before_path: str, after_path: str):
    """对比优化前后的评估报告。"""
    with open(before_path, "r", encoding="utf-8") as f:
        before = json.load(f)
    with open(after_path, "r", encoding="utf-8") as f:
        after = json.load(f)

    print("\n" + "=" * 60)
    print("[COMPARE] 优化前后对比")
    print("=" * 60)
    print(f"{'指标':<20} {'优化前':>10} {'优化后':>10} {'变化':>10}")
    print("-" * 60)

    metrics = [
        ("Context Precision", "avg_context_precision"),
        ("Context Recall", "avg_context_recall"),
        ("平均响应时间(s)", "avg_total_time"),
        ("最大响应时间(s)", "max_total_time"),
    ]

    for name, key in metrics:
        b = before.get(key, 0)
        a = after.get(key, 0)
        diff = a - b
        sign = "+" if diff > 0 else ""
        # 响应时间越小越好，precision/recall 越大越好
        if "时间" in name:
            indicator = "✅" if diff <= 0 else "❌"
        else:
            indicator = "✅" if diff >= 0 else "❌"
        print(f"{name:<20} {b:>10.4f} {a:>10.4f} {sign}{diff:>9.4f} {indicator}")

    print("-" * 60)
    print(f"{'达标情况':}")
    print(f"  Precision ≥ 0.8: {'✅ 达标' if after.get('precision_meets_target') else '❌ 未达标'}")
    print(f"  Recall ≥ 0.9:    {'✅ 达标' if after.get('recall_meets_target') else '❌ 未达标'}")
    print(f"  响应时间 ≤ 3s:   {'✅ 达标' if after.get('time_meets_target') else '❌ 未达标'}")
    print("=" * 60)


# =====================================================================
# 入口
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph RAG 评估脚本")
    parser.add_argument("--before", action="store_true", help="标记为优化前")
    parser.add_argument("--after", action="store_true", help="标记为优化后")
    parser.add_argument("--tag", type=str, default="", help="自定义标签")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="对比两份评估报告")
    args = parser.parse_args()

    if args.compare:
        compare_reports(args.compare[0], args.compare[1])
    else:
        tag = args.tag
        if args.before:
            tag = tag or "before"
        elif args.after:
            tag = tag or "after"
        run_evaluation(tag=tag)
