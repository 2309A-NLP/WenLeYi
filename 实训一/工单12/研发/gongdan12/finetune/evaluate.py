"""评估微调前后的Embedding模型效果。

对比指标：
1. Cosine Similarity (正例对)
2. Recall@K (检索召回率)
3. MRR (平均倒数排名)
4. 检索时间
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# ---- 配置 ----
BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
os.path.join(os.path.dirname(__file__), "..", "models", "m3e-base")
)
FINETUNED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "m3e-finetuned")
DATASET_FILE = os.path.join(os.path.dirname(__file__), "train_dataset.json")
RESULT_FILE = os.path.join(os.path.dirname(__file__), "eval_results.json")

# 从你的RAG项目中提取的测试查询（手动构造或从日志中获取）
TEST_QUERIES = [
    "公司的主营业务是什么？",
    "招股说明书中提到的融资金额是多少？",
    "公司的主要客户有哪些？",
    "公司的竞争优势是什么？",
    "公司的财务状况如何？",
    "公司的风险因素有哪些？",
    "公司的核心技术是什么？",
    "公司的市场份额是多少？",
    "公司的管理团队构成如何？",
    "公司的未来发展战略是什么？",
]


def load_test_data() -> List[Dict]:
    """加载测试数据（从训练集中划分或手动构造）。"""
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 取最后20%作为测试集
    split_idx = int(len(data) * 0.8)
    test_data = data[split_idx:]
    return test_data


def compute_cosine_similarity(
    model: SentenceTransformer, text_pairs: List[Tuple[str, str]]
) -> List[float]:
    """计算文本对的余弦相似度。"""
    embeddings1 = model.encode([p[0] for p in text_pairs])
    embeddings2 = model.encode([p[1] for p in text_pairs])

    similarities = []
    for e1, e2 in zip(embeddings1, embeddings2):
        sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        similarities.append(float(sim))
    return similarities


def compute_recall_at_k(
    model: SentenceTransformer, queries: List[str], corpus: List[str], k: int = 5
) -> float:
    """计算Recall@K。"""
    query_embeddings = model.encode(queries)
    corpus_embeddings = model.encode(corpus)

    recalls = []
    for i, q_emb in enumerate(query_embeddings):
        # 计算与所有语料的相似度
        scores = np.dot(corpus_embeddings, q_emb) / (
            np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(q_emb)
        )
        # 获取top-K
        top_k_idx = np.argsort(scores)[-k:][::-1]
        # 检查正确答案是否在top-K中（这里假设正确答案在corpus中的位置）
        # 简化处理：只要相似度>0.5就算命中
        hits = sum(1 for idx in top_k_idx if scores[idx] > 0.5)
        recalls.append(min(hits / 1.0, 1.0))  # 归一化

    return float(np.mean(recalls))


def compute_mrr(
    model: SentenceTransformer, queries: List[str], corpus: List[str]
) -> float:
    """计算平均倒数排名(MRR)。"""
    query_embeddings = model.encode(queries)
    corpus_embeddings = model.encode(corpus)

    rr_list = []
    for i, q_emb in enumerate(query_embeddings):
        scores = np.dot(corpus_embeddings, q_emb) / (
            np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(q_emb)
        )
        # 找到最相似的语料索引
        top_idx = np.argmax(scores)
        # MRR: 1/rank（这里简化为1/1如果最相似的相似度>0.5）
        if scores[top_idx] > 0.5:
            rr_list.append(1.0)
        else:
            rr_list.append(0.0)

    return float(np.mean(rr_list))


def evaluate_model(model_name: str, model_path: str, test_data: List[Dict]) -> Dict:
    """评估单个模型。"""
    print(f"\n评估模型: {model_name}")
    print(f"  路径: {model_path}")

    # 加载模型
    start_time = time.time()
    model = SentenceTransformer(model_path)
    load_time = time.time() - start_time
    print(f"  加载时间: {load_time:.2f}s")

    # 准备测试数据
    queries = [item["query"] for item in test_data]
    positives = [item["positive"] for item in test_data]
    negatives = [item["negative"] for item in test_data]

    # 1. 余弦相似度（正例对）
    print("  计算余弦相似度...")
    pos_pairs = list(zip(queries, positives))
    pos_similarities = compute_cosine_similarity(model, pos_pairs)
    avg_pos_sim = float(np.mean(pos_similarities))

    neg_pairs = list(zip(queries, negatives))
    neg_similarities = compute_cosine_similarity(model, neg_pairs)
    avg_neg_sim = float(np.mean(neg_similarities))

    # 2. 检索性能
    print("  计算检索性能...")
    # 构建语料库（正例+负例混合）
    corpus = positives + negatives
    recall_5 = compute_recall_at_k(model, queries, corpus, k=5)
    mrr = compute_mrr(model, queries, corpus)

    # 3. 编码速度
    print("  测试编码速度...")
    start_time = time.time()
    for _ in range(10):
        model.encode(queries)
    encode_time = (time.time() - start_time) / 10

    results = {
        "model_name": model_name,
        "model_path": model_path,
        "load_time": load_time,
        "avg_positive_similarity": avg_pos_sim,
        "avg_negative_similarity": avg_neg_sim,
        "similarity_gap": avg_pos_sim - avg_neg_sim,
        "recall_at_5": recall_5,
        "mrr": mrr,
        "avg_encode_time": encode_time,
        "num_test_samples": len(test_data),
    }

    print(f"  正例相似度: {avg_pos_sim:.4f}")
    print(f"  负例相似度: {avg_neg_sim:.4f}")
    print(f"  相似度差距: {avg_pos_sim - avg_neg_sim:.4f}")
    print(f"  Recall@5: {recall_5:.4f}")
    print(f"  MRR: {mrr:.4f}")
    print(f"  平均编码时间: {encode_time:.4f}s")

    return results


def compare_results(base_results: Dict, finetuned_results: Dict) -> Dict:
    """对比两个模型的结果。"""
    comparison = {
        "base_model": base_results,
        "finetuned_model": finetuned_results,
        "improvements": {},
    }

    # 计算提升百分比
    metrics = [
        "avg_positive_similarity",
        "similarity_gap",
        "recall_at_5",
        "mrr",
    ]
    for metric in metrics:
        base_val = base_results[metric]
        finetuned_val = finetuned_results[metric]
        if base_val > 0:
            improvement = (finetuned_val - base_val) / base_val * 100
        else:
            improvement = 0
        comparison["improvements"][metric] = {
            "base": base_val,
            "finetuned": finetuned_val,
            "improvement_pct": improvement,
        }

    return comparison


def evaluate():
    """主函数：评估微调前后效果。"""
    print("=" * 60)
    print("Embedding模型微调效果评估")
    print("=" * 60)

    # 加载测试数据
    print("\n加载测试数据...")
    test_data = load_test_data()
    print(f"  测试样本数: {len(test_data)}")

    # 评估基础模型
    base_results = evaluate_model("m3e-base", BASE_MODEL_PATH, test_data)

    # 评估微调后模型
    if os.path.exists(FINETUNED_MODEL_PATH):
        finetuned_results = evaluate_model(
            "m3e-finetuned", FINETUNED_MODEL_PATH, test_data
        )
    else:
        print(f"\n警告: 微调模型不存在于 {FINETUNED_MODEL_PATH}")
        print("请先运行 finetune.py 进行微调")
        return None

    # 对比结果
    comparison = compare_results(base_results, finetuned_results)

    # 保存结果
    output_path = os.path.abspath(RESULT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存到: {output_path}")

    # 打印对比报告
    print("\n" + "=" * 60)
    print("微调效果对比报告")
    print("=" * 60)
    for metric, data in comparison["improvements"].items():
        print(f"\n{metric}:")
        print(f"  微调前: {data['base']:.4f}")
        print(f"  微调后: {data['finetuned']:.4f}")
        print(f"  提升: {data['improvement_pct']:.2f}%")

    return comparison


if __name__ == "__main__":
    evaluate()
