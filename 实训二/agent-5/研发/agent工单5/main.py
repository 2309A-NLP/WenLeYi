# 工单编号：人工智能NLP-Agent数字人项目-招股书数据问答智能体任务
"""
主程序 - 批量处理问题，生成答案并输出JSONL文件
功能：
1. 读取question.json中的所有问题
2. 逐条调用检索+生成流程
3. 输出answer.jsonl（保留原始id，补全answer字段）
4. 支持断点续跑：已有答案的问题会跳过
"""
import os
import json
import time
from config import QUESTION_FILE, OUTPUT_DIR
from query import query_answer


def load_questions(filepath):
    """
    读取question.jsonl文件（JSONL格式，每行一个JSON对象）
    返回：问题列表 [{"id": ..., "question": ...}, ...]
    """
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                questions.append(item)
            except json.JSONDecodeError as e:
                print(f"[警告] 第{line_num}行JSON解析失败: {e}, 跳过该行")
    print(f"共读取 {len(questions)} 道问题")
    return questions


def load_existing_answers(output_path):
    """
    加载已有的答案文件，用于断点续跑
    返回：{id: answer} 字典
    """
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # 只保留有实际答案的（排除旧的"未找到相关信息"）
                    if item.get("answer") and item["answer"] != "未找到相关信息":
                        existing[item["id"]] = item["answer"]
                except json.JSONDecodeError:
                    pass
        print(f"已有有效答案: {len(existing)} 条")
    return existing


def save_results(results, output_path):
    """
    将结果保存为JSONL文件（每行一个JSON对象）
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + "\n")
    print(f"结果已保存: {output_path} ({len(results)} 条)")


def main():
    """主函数：读取问题 -> 批量处理 -> 保存结果（支持断点续跑）"""
    print("=" * 60)
    print("招股书数据问答智能体 - 批量处理")
    print("工单编号：人工智能NLP-Agent数字人项目-招股书数据问答智能体任务")
    print("=" * 60)

    # 检查问题文件
    if not os.path.exists(QUESTION_FILE):
        print(f"[错误] 问题文件不存在: {QUESTION_FILE}")
        return

    # 读取问题
    print("\n[步骤1] 读取问题文件...")
    questions = load_questions(QUESTION_FILE)
    if not questions:
        print("[错误] 没有读取到任何问题")
        return

    # 加载已有答案（断点续跑）
    output_path = os.path.join(OUTPUT_DIR, "answer.jsonl")
    existing_answers = load_existing_answers(output_path)

    # 统计
    total = len(questions)
    skipped = 0
    processed = 0
    failed = 0
    results = []

    print(f"\n[步骤2] 开始处理 {total} 道问题（已有答案会跳过）...\n")

    for i, item in enumerate(questions, 1):
        qid = item["id"]
        question = item["question"]

        # 断点续跑：已有有效答案则跳过
        if qid in existing_answers:
            results.append({
                "id": qid,
                "question": question,
                "answer": existing_answers[qid],
            })
            skipped += 1
            print(f"[{i}/{total}] [跳过] id={qid} (已有答案)")
            continue

        print(f"\n[{i}/{total}] 处理问题 id={qid}")
        print(f"  问题: {question[:80]}...")

        # 调用检索+生成
        start_time = time.time()
        answer = query_answer(question)
        elapsed = time.time() - start_time

        # 判断是否有效
        is_valid = answer and answer != "未找到相关信息"
        status = "OK" if is_valid else "无答案"

        print(f"  答案: {answer[:80]}...")
        print(f"  耗时: {elapsed:.1f}秒 [{status}]")

        results.append({
            "id": qid,
            "question": question,
            "answer": answer,
        })

        if is_valid:
            processed += 1
        else:
            failed += 1

        # 每10题保存一次（防中断丢失）
        if i % 10 == 0:
            save_results(results, output_path)
            print(f"  [自动保存] 已保存 {len(results)} 条")

        # 请求间隔，避免429限流（首次无需等待）
        if i < total:
            time.sleep(5)

    # 最终保存
    save_results(results, output_path)

    # 统计
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"  总问题数: {total}")
    print(f"  跳过(已有答案): {skipped}")
    print(f"  本次新答对: {processed}")
    print(f"  未找到答案: {failed}")
    print(f"  输出文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
