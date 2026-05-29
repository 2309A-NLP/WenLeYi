"""
RAGAS 评测脚本。

用途：
- 读取金融问答测试集
- 调用项目的检索与生成链路
- 计算 context_relevancy / context_recall 指标并导出结果
"""

import nest_asyncio
nest_asyncio.apply()

import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, Value, Sequence
from ragas import evaluate
from ragas.metrics import context_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("RAGAS 评测脚本")
print("=" * 60)

# 导入 main1 中的必要函数
from main1 import get_embedding_model, hybrid_search, get_llm_client, init_milvus

def load_test_dataset():
    """加载评测集 CSV，默认字段包含 question 与 ground_truth。"""
    print("\n📂 加载测试数据集...")
    df = pd.read_csv("data/financial_qa.csv")
    print(f"✅ 成功加载 {len(df)} 个测试样本")
    print(f"   数据列: {list(df.columns)}")
    return df

def get_answer_and_context(question: str, idx: int, total: int):
    """执行一次 RAG 问答，返回模型答案与检索上下文。"""
    print(f"\n[{idx+1}/{total}] 处理问题: {question[:50]}...")
    client = get_llm_client()
    retrieved_docs = hybrid_search(question, role_id=1)

    if not retrieved_docs:
        context = "无相关金融知识参考。"
    else:
        context = "\n\n".join(
            [doc.get("text", "") for doc in retrieved_docs if isinstance(doc, dict)]
        )
        print(f"   检索到 {len(retrieved_docs)} 条相关文档")

    try:
        print("   正在调用 LLM 生成回答...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位金融理财师，请根据上下文和用户问题提供专业回答。"},
                {"role": "user", "content": f"【金融知识参考】：{context}\n客户的问题：{question}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        print(f"   回答生成成功 (长度: {len(answer)} 字符)")
    except Exception as e:
        print(f"   ❌ 调用 LLM API 时出错: {e}")
        answer = "抱歉，我现在无法回答，请稍后再试。"

    if not isinstance(retrieved_docs, list):
        retrieved_docs = []

    return answer, retrieved_docs

def prepare_ragas_dataset(test_df: pd.DataFrame) -> Dataset:
    """将测试 DataFrame 转换为 RAGAS 所需 Dataset 结构。"""
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("\n" + "=" * 60)
    print("开始执行 RAG 流程")
    print("=" * 60)

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="执行 RAG 流程"):
        question = row['question']
        ground_truth = row['ground_truth']
        answer, retrieved_contexts = get_answer_and_context(question, idx, len(test_df))

        questions.append(question)
        answers.append(answer)
        contexts.append([str(c) if c else "" for c in retrieved_contexts])
        ground_truths.append(ground_truth)

    print("\n" + "=" * 60)
    print("构建 RAGAS 数据集")
    print("=" * 60)

    data = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths,
    }

    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("contexts", Sequence(Value("string")))

    print(f"✅ RAGAS 数据集构建完成，包含 {len(dataset)} 条样本")

    return dataset

def main():
    print("\n" + "=" * 60)
    print("步骤 1: 初始化 Milvus 和 Embedding 模型")
    print("=" * 60)

    print("初始化 Milvus 连接和模型...")
    init_milvus()
    get_embedding_model()
    print("✅ 初始化完成")

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        print("❌ 错误：请在 .env 文件中设置 DEEPSEEK_API_KEY")
        return

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    print(f"使用 API Base URL: {base_url}")

    class CustomChatOpenAI(ChatOpenAI):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs['n'] = 1
            kwargs.pop('logit_bias', None)
            kwargs.pop('frequency_penalty', None)
            kwargs.pop('presence_penalty', None)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    print("\n" + "=" * 60)
    print("步骤 2: 加载 LLM 评估器")
    print("=" * 60)

    llm = CustomChatOpenAI(
        model="deepseek-chat",
        openai_api_key=deepseek_key,
        openai_api_base=base_url,
        temperature=0,
        max_tokens=512,
        request_timeout=30,
        max_retries=1,
        n=1,
    )
    evaluator_llm = LangchainLLMWrapper(llm)
    print("✅ LLM 评估器加载完成")

    print("\n" + "=" * 60)
    print("步骤 3: 加载测试数据集")
    print("=" * 60)

    test_df = load_test_dataset()

    print("\n" + "=" * 60)
    print("步骤 4: 执行 RAG 流程并准备评估数据")
    print("=" * 60)

    ragas_dataset = prepare_ragas_dataset(test_df)

    print("\n" + "=" * 60)
    print("步骤 5: 执行 RAGAS 评估")
    print("=" * 60)
    print("⚠️ 注意：评估过程可能需要几分钟，请耐心等待...")

    result = evaluate(
        ragas_dataset,
        metrics=[context_relevancy, context_recall],
        llm=evaluator_llm,
    )

    print("\n" + "=" * 60)
    print("📊 RAGAS 评估结果")
    print("=" * 60)

    print(result)

    df_result = result.to_pandas()

    print("\n===== 详细评估结果 =====")
    print(df_result.to_string())

    output_dir = "evaluation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "ragas_evaluation_results.csv")
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 评估结果已保存到: {os.path.abspath(output_path)}")
    print("=" * 60)
    print("评测完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
