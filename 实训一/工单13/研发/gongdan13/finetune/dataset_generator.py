"""从PDF文档生成Embedding微调数据集。

从招股说明书中提取文本段落，调用LLM生成问答对。
输出格式：三元组 (query, positive, negative)
"""

import os
import json
import random
import requests
import fitz  # PyMuPDF
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple


# ---- 配置 ----
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "documents")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "train_dataset.json")
LLM_API_URL = "https://api.xiaomimimo.com/v1/chat/completions"
LLM_API_KEY = os.environ.get("MIMO_SK_KEY", "")
LLM_MODEL = "mimo-v2.5"
CHUNK_SIZE = 512  # 每段文本最大字符数
NUM_QUESTIONS_PER_CHUNK = 3  # 每段生成几个问题
MAX_CHUNKS = 200  # 最多处理的段落数（减少调用次数）
MAX_WORKERS = 5  # 并发线程数


def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """从PDF提取文本段落。"""
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text()
    doc.close()

    # 按段落分割，合并短段落
    paragraphs = [p.strip() for p in all_text.split("\n") if len(p.strip()) > 20]
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) > CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = p
        else:
            current_chunk = current_chunk + "\n" + p if current_chunk else p
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def call_llm(prompt: str) -> str:
    """调用LLM生成内容。"""
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    try:
        resp = requests.post(LLM_API_URL, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [LLM调用失败] {e}")
        return ""


def generate_qa_pairs(text_chunk: str) -> List[Dict]:
    """对一段文本生成问答对三元组。"""
    prompt = f"""你是一个数据标注专家。请基于以下文本段落，生成{NUM_QUESTIONS_PER_CHUNK}个高质量的问答对三元组。

要求：
1. query（问题）：针对文本内容提出具体问题
2. positive（正例答案）：从文本中可以找到正确答案的回答
3. negative（负例答案）：与问题相关但答案错误或不够准确的回答

文本段落：
{text_chunk}

请严格按以下JSON格式输出（只输出JSON数组，不要其他文字）：
[
  {{"query": "问题1", "positive": "正确答案1", "negative": "错误答案1"}},
  {{"query": "问题2", "positive": "正确答案2", "negative": "错误答案2"}}
]"""

    response = call_llm(prompt)
    if not response:
        return []

    # 解析JSON
    valid = []
    try:
        # 尝试提取JSON部分
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            qa_list = json.loads(response[start:end])
            # 验证格式
            for qa in qa_list:
                if all(k in qa for k in ("query", "positive", "negative")):
                    valid.append(qa)
    except json.JSONDecodeError:
        pass
    return valid


def process_chunk(i, chunk):
    """处理单个段落（用于并发）。"""
    qa_pairs = generate_qa_pairs(chunk)
    return qa_pairs


def generate_dataset():
    """主函数：从所有PDF生成数据集。"""
    print("=" * 50)
    print("Embedding微调数据集生成器")
    print("=" * 50)

    pdf_dir = os.path.abspath(PDF_DIR)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    print(f"发现 {len(pdf_files)} 个PDF文件")

    all_chunks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"提取文本: {pdf_file}")
        chunks = extract_text_from_pdf(pdf_path)
        print(f"  -> {len(chunks)} 个段落")
        all_chunks.extend(chunks)

    print(f"\n共 {len(all_chunks)} 个段落，开始生成问答对...")
    # 只取前MAX_CHUNKS个段落
    all_chunks = all_chunks[:MAX_CHUNKS]
    print(f"取前 {MAX_CHUNKS} 个段落，并发 {MAX_WORKERS} 线程")
    print(f"预计生成约 {len(all_chunks) * NUM_QUESTIONS_PER_CHUNK} 条数据\n")

    dataset = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(all_chunks)}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            done_count += 1
            i = futures[future]
            try:
                qa_pairs = future.result()
                if qa_pairs:
                    dataset.extend(qa_pairs)
                    print(f"[{done_count}/{len(all_chunks)}] 获得 {len(qa_pairs)} 条")
                else:
                    print(f"[{done_count}/{len(all_chunks)}] 失败，跳过")
            except Exception as e:
                print(f"[{done_count}/{len(all_chunks)}] 异常: {e}")

    # 去重
    seen = set()
    unique_dataset = []
    for item in dataset:
        key = item["query"]
        if key not in seen:
            seen.add(key)
            unique_dataset.append(item)

    print(f"\n去重后共 {len(unique_dataset)} 条训练数据")

    # 保存
    output_path = os.path.abspath(OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique_dataset, f, ensure_ascii=False, indent=2)
    print(f"数据集已保存到: {output_path}")

    return unique_dataset


if __name__ == "__main__":
    generate_dataset()
