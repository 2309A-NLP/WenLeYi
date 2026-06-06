# Embedding模型微调工具

从PDF文档生成训练数据，微调m3e-base模型，提升RAG检索效果。

## 快速开始

### 1. 安装依赖
```bash
D:\an10-1\envs\nlp_1\python.exe -m pip install sentence-transformers datasets
```

### 2. 设置API密钥
```bash
set MIMO_SK_KEY=你的sk-开头的密钥
```

### 3. 一键运行全流程
```bash
D:\an10-1\envs\nlp_1\python.exe finetune/run.py
```

或者分步运行：

```bash
# 步骤1：生成数据集
D:\an10-1\envs\nlp_1\python.exe finetune/dataset_generator.py

# 步骤2：微调模型
D:\an10-1\envs\nlp_1\python.exe finetune/finetune.py

# 步骤3：评估效果
D:\an10-1\envs\nlp_1\python.exe finetune/evaluate.py
```

## 文件说明

- `run.py` - 一键运行全流程
- `dataset_generator.py` - 从PDF生成问答对数据集
- `finetune.py` - 微调Embedding模型
- `evaluate.py` - 评估微调前后效果
- `train_dataset.json` - 生成的训练数据集（运行后生成）
- `m3e-finetuned/` - 微调后的模型（运行后生成）
- `eval_results.json` - 评估结果（运行后生成）

## 输出说明

微调后的模型保存在 `finetune/m3e-finetuned/` 目录下，可以直接替换项目中的 `m3e-base` 使用。

评估报告会生成 `eval_results.json`，包含：
- 正例/负例余弦相似度
- Recall@5（检索召回率）
- MRR（平均倒数排名）
- 编码速度对比

## 配置说明

在 `finetune.py` 中可以修改以下参数：

```python
BATCH_SIZE = 16        # 批量大小
EPOCHS = 3             # 训练轮数
LEARNING_RATE = 2e-5   # 学习率
WARMUP_STEPS = 100     # 预热步数
```

## 常见问题

Q: 报错 `No module named 'sentence_transformers'`
A: 需要安装依赖: `pip install sentence-transformers`

Q: 报错 `LLM调用失败`
A: 检查 MIMO_SK_KEY 环境变量是否正确设置

Q: 微调后效果没有提升
A: 尝试增加训练数据量或调整训练参数
