"""微调Embedding模型。

使用sentence-transformers对m3e-base进行领域微调。
支持对比学习（MultipleNegativesRankingLoss）和三元组损失。
"""

import os
import json
import random
import torch
from pathlib import Path
from typing import List, Dict
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from sentence_transformers.readers import (
    TripletReader,
)

# ---- 配置 ----
BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    r"D:\桌面\模型\m3e-base"
)
DATASET_FILE = os.path.join(os.path.dirname(__file__), "train_dataset.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "m3e-finetuned")
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
EVAL_STEPS = 50
SAVE_STEPS = 100
LOGGING_STEPS = 10


def load_dataset() -> List[InputExample]:
    """加载训练数据集。"""
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        # 三元组格式: (query, positive, negative)
        examples.append(
            InputExample(
                texts=[item["query"], item["positive"]],
                label=1.0,  # 正例对
            )
        )
        examples.append(
            InputExample(
                texts=[item["query"], item["negative"]],
                label=0.0,  # 负例对
            )
        )

    print(f"加载 {len(examples)} 条训练样本（{len(data)} 个三元组）")
    return examples


def create_eval_data() -> tuple:
    """创建评估数据（从训练集中划分10%）。"""
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 随机打乱
    random.seed(42)
    random.shuffle(data)

    # 取前10%作为评估
    split_idx = max(1, len(data) // 10)
    eval_data = data[:split_idx]

    queries = [item["query"] for item in eval_data]
    positives = [item["positive"] for item in eval_data]
    negatives = [item["negative"] for item in eval_data]

    return queries, positives, negatives


def finetune():
    """执行微调。"""
    print("=" * 50)
    print("Embedding模型微调")
    print("=" * 50)

    # 1. 加载基础模型
    print(f"\n[1/5] 加载基础模型: {BASE_MODEL_PATH}")
    model = SentenceTransformer(BASE_MODEL_PATH)
    print(f"  -> 模型维度: {model.get_sentence_embedding_dimension()}")

    # 2. 加载训练数据
    print(f"\n[2/5] 加载训练数据...")
    train_examples = load_dataset()
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=BATCH_SIZE
    )

    # 3. 定义损失函数
    print(f"\n[3/5] 定义损失函数: MultipleNegativesRankingLoss")
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 4. 创建评估器
    print(f"\n[4/5] 创建评估器...")
    queries, positives, negatives = create_eval_data()
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1=queries,
        sentences2=positives,
        scores=[1.0] * len(queries),  # 全部正例
        main_similarity="cosine",
        name="eval",
    )

    # 5. 开始训练
    print(f"\n[5/5] 开始训练...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Warmup Steps: {WARMUP_STEPS}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        evaluation_steps=EVAL_STEPS,
        output_path=OUTPUT_DIR,
        optimizer_params={"lr": LEARNING_RATE},
        show_progress_bar=True,
    )

    print(f"\n训练完成！模型已保存到: {OUTPUT_DIR}")

    # 最终评估
    print("\n执行最终评估...")
    final_score = evaluator(model)
    print(f"最终相似度评分: {final_score:.4f}")

    return model, final_score


if __name__ == "__main__":
    finetune()
