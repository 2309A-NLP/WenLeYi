"""一键运行Embedding模型微调全流程（AutoDL版）。

AutoDL环境：
- GPU: A100/3090/4090等
- 系统: Ubuntu
- Python: conda环境
- 数据盘: /root/autodl-tmp/

使用方法:
    python run.py
"""

import os
import sys

# AutoDL数据盘路径
AUTODL_DATA_DIR = "/root/autodl-tmp"

# 确保在正确的目录下运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dataset_generator import generate_dataset
from finetune import finetune
from evaluate import evaluate


def main():
    print("=" * 60)
    print("Embedding模型微调全流程 (AutoDL版)")
    print("=" * 60)
    print()

    # 检查GPU
    import torch
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: GPU不可用，将使用CPU训练（速度很慢）")

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
            print("评估结果已保存到 eval_results.json")
            print("微调模型已保存到 m3e-finetuned/")
    except Exception as e:
        print(f"评估失败: {e}")
        return


if __name__ == "__main__":
    main()
