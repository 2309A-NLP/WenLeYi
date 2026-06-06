"""RAG vs LightRAG 对比评估脚本（RAGAS 指标版）。

实现 RAGAS 四大核心指标:
  - Faithfulness（忠实度）：回答是否基于检索到的上下文
  - Answer Relevancy（答案相关性）：回答是否切题
  - Context Precision（上下文精度）：检索到的上下文是否与问题相关
  - Context Recall（上下文召回率）：是否检索到了回答所需的全部信息

使用方法:
    python scripts/compare_rag_lightrag.py
    python scripts/compare_rag_lightrag.py --build-index
    python scripts/compare_rag_lightrag.py --questions 3
"""

import sys
import os
import re
import json
import time
import argparse
import shutil
from pathlib import Path
from typing import List, Dict

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.config import Config
from core.lightrag_engine import LightRAGEngine


# ======================================================================
# RAGAS 指标评估引擎（LLM-as-Judge）
# ======================================================================

class RAGASEvaluator:
    """RAGAS 四大核心指标的 LLM 评估实现。"""

    def __init__(self, api_key: str, base_url: str, model: str):
        import requests
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.requests = requests

    def _llm_judge(self, prompt: str) -> float:
        """调用 LLM 评分，返回 0-1 分数。"""
        try:
            resp = self.requests.post(
                f"{self.base_url}/chat/completions",
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.0, "max_tokens": 500},
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r'(\d+\.?\d*)', text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score / 100.0 if score > 1 else score))
            return 0.0
        except Exception as e:
            print(f"    [评估] LLM调用失败: {e}")
            return 0.0

    def _overlap_score(self, left: str, right: str) -> float:
        """LLM judge 返回 0 时的轻量兜底，按中文二/三元词项重合估分。"""
        def terms(text: str):
            compact = re.sub(r"\s+", "", text or "")
            output = set()
            output.update(compact[i:i + 2] for i in range(max(1, len(compact) - 1)))
            output.update(compact[i:i + 3] for i in range(max(1, len(compact) - 2)))
            return {item for item in output if item}

        a, b = terms(left), terms(right)
        if not a or not b:
            return 0.0
        return max(0.0, min(1.0, len(a & b) / max(1, min(len(a), len(b)))))

    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Faithfulness: 回答中有多少内容能从上下文中找到依据。"""
        if not answer or not contexts:
            return 0.0
        ctx_text = "\n---\n".join(c[:800] for c in contexts[:5])
        prompt = f"""请逐句分析以下回答，判断每个句子是否有上下文依据支持。

上下文：
{ctx_text}

回答：
{answer}

评分标准：
- 1.0: 回答的每个句子都能在上下文中找到依据
- 0.8: 大部分句子有依据，少数是合理推断
- 0.6: 一半左右的句子有依据
- 0.4: 少量句子有依据
- 0.2: 极少句子有依据
- 0.0: 回答完全是编造的

请只返回一个 0-1 之间的数字。"""
        score = self._llm_judge(prompt)
        return score or self._overlap_score(answer, "\n".join(contexts))

    def answer_relevancy(self, question: str, answer: str) -> float:
        """Answer Relevancy: 回答是否直接针对问题。"""
        if not answer:
            return 0.0
        prompt = f"""请评估以下回答与问题的相关性。

问题：{question}

回答：{answer}

评分标准：
- 1.0: 完全切题，直接回答了问题的所有方面
- 0.8: 基本切题，回答了主要方面
- 0.6: 部分切题，回答了一些相关内容
- 0.4: 偏题，只涉及小部分相关内容
- 0.2: 基本偏题
- 0.0: 完全不相关

请只返回一个 0-1 之间的数字。"""
        score = self._llm_judge(prompt)
        return score or self._overlap_score(question, answer)

    def context_precision(self, question: str, contexts: List[str]) -> float:
        """Context Precision: 检索到的上下文中有多少与问题相关。"""
        if not contexts:
            return 0.0
        ctx_text = "\n---\n".join(c[:500] for c in contexts[:5])
        prompt = f"""请评估以下检索到的上下文与问题的相关性。

问题：{question}

检索到的上下文：
{ctx_text}

评分标准：
- 1.0: 所有上下文都与问题直接相关
- 0.8: 大部分上下文与问题相关
- 0.6: 一半上下文与问题相关
- 0.4: 少量上下文与问题相关
- 0.2: 几乎没有相关上下文
- 0.0: 完全不相关

请只返回一个 0-1 之间的数字。"""
        score = self._llm_judge(prompt)
        return score or self._overlap_score(question, "\n".join(contexts))

    def context_recall(self, question: str, answer: str, contexts: List[str]) -> float:
        """Context Recall: 上下文是否涵盖了回答所需的信息。"""
        if not answer or not contexts:
            return 0.0
        ctx_text = "\n---\n".join(c[:500] for c in contexts[:5])
        prompt = f"""请评估检索到的上下文是否包含了生成以下回答所需的全部信息。

问题：{question}

回答：
{answer}

检索到的上下文：
{ctx_text}

评分标准：
- 1.0: 上下文完全覆盖了回答中提到的所有信息点
- 0.8: 上下文覆盖了大部分信息点
- 0.6: 上下文覆盖了一半信息点
- 0.4: 上下文只覆盖了少量信息点
- 0.2: 上下文几乎没有覆盖回答所需的信息
- 0.0: 完全没有覆盖

请只返回一个 0-1 之间的数字。"""
        score = self._llm_judge(prompt)
        return score or self._overlap_score(answer, "\n".join(contexts))

    def evaluate_all(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """一次性计算四个指标。"""
        f = self.faithfulness(answer, contexts)
        ar = self.answer_relevancy(question, answer)
        cp = self.context_precision(question, contexts)
        cr = self.context_recall(question, answer, contexts)
        return {
            "faithfulness": f,
            "answer_relevancy": ar,
            "context_precision": cp,
            "context_recall": cr,
        }


# ======================================================================
# 工具函数
# ======================================================================

def load_test_questions(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_lightrag_index(engine: LightRAGEngine, docs_dir: str):
    pdf_files = list(Path(docs_dir).glob("*.pdf"))
    print(f"\n{'='*60}")
    print(f"[索引] 发现 {len(pdf_files)} 个 PDF 文件")
    print(f"{'='*60}")
    for pdf_file in pdf_files:
        if "sample_questions" in pdf_file.name:
            continue
        result = engine.insert_pdf(str(pdf_file))
        print(f"[索引] {pdf_file.name}: {result['status']}")
    engine.save_graph_to_file(os.path.join(engine.working_dir, "knowledge_graph.json"))


def reset_lightrag_storage(working_dir: str):
    resolved = os.path.abspath(working_dir)
    project_resolved = os.path.abspath(str(PROJECT_DIR))
    if not resolved.startswith(project_resolved):
        raise RuntimeError(f"拒绝清理项目目录外的路径: {resolved}")
    if os.path.isdir(resolved):
        print(f"[LightRAG] 清空旧存储: {resolved}")
        shutil.rmtree(resolved)


def init_rag_components(config: Config) -> dict:
    """初始化 RAG 组件（带自动建索引）。"""
    components = {}
    try:
        from core.vector_store import VectorStore
        from core.retriever import HybridRetriever
        from core.rag import RAGPipeline
        from core.llm import LLMClient

        vector_store = VectorStore(
            milvus_host=config.MILVUS_HOST,
            milvus_port=config.MILVUS_PORT,
            collection_name=config.MILVUS_COLLECTION,
            embedding_model_path=config.EMBEDDING_MODEL_PATH,
        )

        chunks = vector_store.load_chunks()
        if not chunks:
            print("[RAG] WARN 向量库为空，从文档构建索引...")
            from core.document_processor import process_documents
            chunks = process_documents(config.DOCS_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            if chunks:
                vector_store.build_index(chunks)
                print(f"[RAG] OK 构建完成: {len(chunks)} chunks")
            else:
                print("[RAG] ERROR 无法构建 chunks")
                return components

        retriever = HybridRetriever(vector_store, chunks, config)
        llm_client = LLMClient(config.LLM_API_KEY, config.LLM_API_BASE, config.LLM_MODEL, config.LLM_TIMEOUT)
        rag = RAGPipeline(retriever, llm_client, config)

        components = {"retriever": retriever, "rag": rag}
        print(f"[RAG] OK 初始化完成, chunks={len(chunks)}")
    except Exception as e:
        print(f"[RAG] ERROR 初始化失败: {e}")
        import traceback
        traceback.print_exc()
    return components


def run_rag_query(question: str, config: Config, components: dict) -> dict:
    try:
        retriever = components.get("retriever")
        rag = components.get("rag")
        if not retriever or not rag:
            return {"answer": "RAG组件未初始化", "context": "", "error": "no components"}

        results = retriever.retrieve(question)
        if not results:
            return {"answer": "未检索到相关内容", "context": "", "error": "no results"}

        context = "\n".join([chunk.get("text", "")[:500] for chunk, _ in results[:5]])
        answer = rag.query(question, stream=False)
        return {"answer": answer, "context": context}
    except Exception as e:
        return {"answer": f"RAG查询失败: {e}", "context": "", "error": str(e)}


def run_lightrag_query(question: str, engine: LightRAGEngine) -> dict:
    return engine.query_with_context(question, mode="mix")


def generate_report(
    rag_results: List[Dict], lightrag_results: List[Dict],
    rag_metrics: dict, lightrag_metrics: dict, output_dir: str,
) -> str:
    """生成 Markdown 对比报告。"""
    lines = []
    lines.append("# RAG vs LightRAG 检索结果对比报告\n")
    lines.append(f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 评估指标
    lines.append("## 一、RAGAS 评估指标对比\n")
    lines.append("| 指标 | 传统RAG | LightRAG | 差异 |")
    lines.append("|:-----|:-------:|:--------:|:----:|")

    metric_names = {
        "faithfulness": "Faithfulness（忠实度）",
        "answer_relevancy": "Answer Relevancy（答案相关性）",
        "context_precision": "Context Precision（上下文精度）",
        "context_recall": "Context Recall（上下文召回率）",
    }
    rag_vals, ltrag_vals = [], []
    for key, name in metric_names.items():
        rv = rag_metrics.get(key, 0.0)
        lv = lightrag_metrics.get(key, 0.0)
        d = lv - rv
        rag_vals.append(rv)
        ltrag_vals.append(lv)
        lines.append(f"| {name} | {rv:.4f} | {lv:.4f} | {'+' if d>=0 else ''}{d:.4f} |")

    rag_avg = sum(rag_vals)/len(rag_vals) if rag_vals else 0
    ltrag_avg = sum(ltrag_vals)/len(ltrag_vals) if ltrag_vals else 0
    d = ltrag_avg - rag_avg
    lines.append(f"| **综合平均分** | **{rag_avg:.4f}** | **{ltrag_avg:.4f}** | **{'+' if d>=0 else ''}{d:.4f}** |")
    lines.append("")

    # 逐题对比
    lines.append("## 二、逐题检索结果对比\n")
    max_len = max(len(rag_results), len(lightrag_results))
    for i in range(max_len):
        ri = rag_results[i] if i < len(rag_results) else {}
        li = lightrag_results[i] if i < len(lightrag_results) else {}
        qid = ri.get("id", li.get("id", "N/A"))
        q = ri.get("question", li.get("question", ""))
        ra = ri.get("answer", "N/A")
        la = li.get("answer", "N/A")
        rt = ri.get("time", 0)
        lt = li.get("time", 0)

        lines.append(f"### 问题 {i+1} (ID: {qid})\n")
        lines.append(f"**问题**: {q}\n")
        lines.append(f"<table><tr><th></th><th>传统RAG</th><th>LightRAG</th></tr>")
        lines.append(f"<tr><td><b>回答</b></td><td>{ra[:300]}{'...' if len(ra)>300 else ''}</td><td>{la[:300]}{'...' if len(la)>300 else ''}</td></tr>")
        lines.append(f"<tr><td><b>耗时</b></td><td>{rt:.2f}s</td><td>{lt:.2f}s</td></tr></table>\n")

    # 总结
    lines.append("## 三、总结\n")
    if rag_vals and ltrag_vals:
        winner = "LightRAG" if ltrag_avg > rag_avg else "传统RAG"
        lines.append(f"- 综合评估: **{winner}** 表现更优")
        lines.append(f"- 传统RAG 综合分: {rag_avg:.4f}")
        lines.append(f"- LightRAG 综合分: {ltrag_avg:.4f}")
    lines.append(f"- 共评估 {max_len} 个问题")

    md = "\n".join(lines)
    with open(os.path.join(output_dir, "comparison_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


def main():
    parser = argparse.ArgumentParser(description="RAG vs LightRAG 对比评估")
    parser.add_argument("--build-index", action="store_true", help="先构建LightRAG索引")
    parser.add_argument("--reset-lightrag", action="store_true", help="先清空 LightRAG 存储再构建")
    parser.add_argument("--questions", type=int, default=0, help="只测试前N个问题")
    parser.add_argument("--skip-rag", action="store_true", help="跳过传统RAG")
    parser.add_argument("--skip-lightrag", action="store_true", help="跳过LightRAG")
    args = parser.parse_args()

    config = Config()
    config.SEARCH_MODE = os.getenv("EVAL_SEARCH_MODE", "bm25")
    print(f"\n{'='*60}")
    print("RAG vs LightRAG 对比评估（RAGAS 指标版）")
    print(f"{'='*60}")

    questions = load_test_questions(os.path.join(config.DOCS_DIR, "test_questions.json"))
    if args.questions > 0:
        questions = questions[:args.questions]
    print(f"[测试] 共 {len(questions)} 个问题")

    # LightRAG
    lightrag_engine = None
    if not args.skip_lightrag:
        working_dir = os.path.join(str(PROJECT_DIR), "lightrag_storage")
        if args.reset_lightrag:
            reset_lightrag_storage(working_dir)
        lightrag_engine = LightRAGEngine(
            api_key=config.LLM_API_KEY, base_url=config.LLM_API_BASE,
            model=config.LLM_MODEL,
            working_dir=working_dir,
            embedding_model_path=config.EMBEDDING_MODEL_PATH,
        )
        index_file = os.path.join(lightrag_engine.working_dir, "vdb_entities.json")
        if args.build_index or not os.path.exists(index_file):
            build_lightrag_index(lightrag_engine, config.DOCS_DIR)

    # RAG
    rag_components = {}
    if not args.skip_rag:
        print("\n[RAG] 初始化...")
        rag_components = init_rag_components(config)

    # 查询
    rag_results, lightrag_results = [], []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['question'][:50]}...")

        if not args.skip_rag:
            start = time.time()
            r = run_rag_query(q["question"], config, rag_components)
            r["id"], r["question"], r["time"] = q["id"], q["question"], time.time()-start
            rag_results.append(r)
            print(f"  RAG: {r['time']:.2f}s, {len(r.get('answer',''))}字")

        if not args.skip_lightrag and lightrag_engine:
            start = time.time()
            l = run_lightrag_query(q["question"], lightrag_engine)
            l["id"], l["question"], l["time"] = q["id"], q["question"], time.time()-start
            lightrag_results.append(l)
            print(f"  LightRAG: {l['time']:.2f}s, {len(l.get('answer',''))}字")

    # 评估
    print(f"\n{'='*60}")
    print("RAGAS 指标评估...")
    print(f"{'='*60}")

    evaluator = RAGASEvaluator(config.LLM_API_KEY, config.LLM_API_BASE, config.LLM_MODEL)

    def eval_results(results):
        all_m = {"faithfulness":[], "answer_relevancy":[], "context_precision":[], "context_recall":[]}
        for i, item in enumerate(results):
            ctx = [item.get("context","")] if item.get("context") else []
            m = evaluator.evaluate_all(item["question"], item.get("answer",""), ctx)
            for k,v in m.items():
                all_m[k].append(v)
            print(f"  [{i+1}] F={m['faithfulness']:.2f} AR={m['answer_relevancy']:.2f} CP={m['context_precision']:.2f} CR={m['context_recall']:.2f}")
        return {k: sum(v)/len(v) if v else 0 for k,v in all_m.items()}

    rag_metrics = eval_results(rag_results) if rag_results else {}
    lightrag_metrics = eval_results(lightrag_results) if lightrag_results else {}

    # 保存
    output_dir = os.path.join(str(PROJECT_DIR), "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "rag_results.json"), "w", encoding="utf-8") as f:
        json.dump(rag_results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "lightrag_results.json"), "w", encoding="utf-8") as f:
        json.dump(lightrag_results, f, ensure_ascii=False, indent=2)
    comp = {"rag_metrics": rag_metrics, "lightrag_metrics": lightrag_metrics}
    if rag_metrics and lightrag_metrics:
        comp["improvement"] = {k: lightrag_metrics.get(k,0)-rag_metrics.get(k,0) for k in rag_metrics}
    with open(os.path.join(output_dir, "metrics_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comp, f, ensure_ascii=False, indent=2)

    # 生成报告
    md = generate_report(rag_results, lightrag_results, rag_metrics, lightrag_metrics, output_dir)

    # 终端打印
    print(f"\n{'='*60}")
    print("评估指标对比")
    print(f"{'='*60}")
    if rag_metrics:
        print(f"{'指标':<22} {'传统RAG':<10} {'LightRAG':<10}")
        print("-"*42)
        for k in rag_metrics:
            print(f"{k:<22} {rag_metrics[k]:<10.4f} {lightrag_metrics.get(k,0):<10.4f}")
    print(f"\n报告已保存: {output_dir}/comparison_report.md")


if __name__ == "__main__":
    main()
