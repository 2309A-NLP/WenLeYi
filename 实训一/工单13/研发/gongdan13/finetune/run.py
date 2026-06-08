"""一键运行Embedding模型微调全流程。

使用方法:
    D:\an10-1\envs\nlp_1\python.exe finetune/run.py

流程:
    1. 生成数据集（从PDF提取问答对）
    2. 微调模型（用sentence-transformers训练）
    3. 评估对比（微调前后效果对比）
"""

import os
import sys

# 确保在正确的目录下运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dataset_generator import generate_dataset
from finetune import finetune
from evaluate import evaluate


def main():
    print("=" * 60)
    print("Embedding模型微调全流程")
    print("=" * 60)
    print()

    # 第一步：生成数据集
    print("\n" + "=" * 40)
    print("步骤1：生成训练数据集")
    print("=" * 40)
    try:
        dataset = generate_dataset()
        print(f"数据集生成完成，共 {len(dataset)} 条数据")
    except Exception as e:
        print(f"数据集生成失败: {e}")
        print("请检查 MIMO_SK_KEY 环境变量是否设置")
        return

    # 第二步：微调模型
    print("\n" + "=" * 40)
    print("步骤2：微调Embedding模型")
    print("=" * 40)
    try:
        model, score = finetune()
        print(f"微调完成，评估得分: {score:.4f}")
    except Exception as e:
        print(f"微调失败: {e}")
        return

    # 第三步：评估对比
    print("\n" + "=" * 40)
    print("步骤3：评估微调效果")
    print("=" * 40)
    try:
        results = evaluate()
        if results:
            print("\n全流程完成！")
            print("评估结果已保存到 finetune/eval_results.json")
    except Exception as e:
        print(f"评估失败: {e}")
        return


if __name__ == "__main__":
    main()
