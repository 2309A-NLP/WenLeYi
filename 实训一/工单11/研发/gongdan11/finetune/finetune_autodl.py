"""微调Embedding模型（AutoDL版）。

使用sentence-transformers对m3e-base进行领域微调。
AutoDL环境优化：支持GPU加速、混合精度训练。
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

# ---- AutoDL配置 ----
# AutoDL数据盘路径
AUTODL_DATA_DIR = "/root/autodl-tmp"

# 基础模型路径（AutoDL上需要先下载或上传模型）
BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    os.path.join(AUTODL_DATA_DIR, "models", "m3e-base")
)

# 训练数据路径
DATASET_FILE = os.path.join(os.path.dirname(__file__), "train_dataset.json")

# 输出目录
OUTPUT_DIR = os.path.join(AUTODL_DATA_DIR, "finetune", "m3e-finetuned")

# ---- 训练参数（GPU优化） ----
BATCH_SIZE = 32  # GPU显存足够可以加大
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
EVAL_STEPS = 50
SAVE_STEPS = 100
LOGGING_STEPS = 10

# 混合精度训练（A100/3090支持）
USE_FP16 = torch.cuda.is_available()


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
    """执行微调（AutoDL优化版）。"""
    print("=" * 50)
    print("Embedding模型微调 (AutoDL版)")
    print("=" * 50)

    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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
    print(f"  FP16: {USE_FP16}")
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
