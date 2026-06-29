# 工单编号：人工智能NLP-Agent数字人项目-基金问答智能体任务
"""
重新处理失败的题目
"""
import json
import os
from config import QUESTION_FILE, OUTPUT_FILE, OUTPUT_DIR
from prompt_builder import build_system_prompt
from main import is_prospectus_question
from sql_generator import generate_sql, generate_sql_with_retry
from sql_executor import execute_sql, format_answer


def load_questions():
    """加载问题"""
    questions = []
    with open(QUESTION_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line.strip()))
    return {q['id']: q for q in questions}


def load_results():
    """加载已有结果"""
    results = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line.strip())
                    results[r['id']] = r
    return results


def save_results(results):
    """保存结果"""
    sorted_results = sorted(results.values(), key=lambda x: x['id'])
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for r in sorted_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def process_question(q, system_prompt):
    """处理单个问题"""
    question = q['question']
    result = {'id': q['id'], 'question': question, 'answer': '', 'sql': '', 'error': None}
    
    # 招股书问题
    if is_prospectus_question(question):
        from pdf_parser import answer_prospectus_question
        from sql_generator import generate_prospectus_answer
        
        content = answer_prospectus_question(question)
        if content:
            answer = generate_prospectus_answer(question, content)
            if answer:
                result['answer'] = answer
                return result
        
        result['answer'] = '[招股书问题，未找到相关数据]'
        result['error'] = 'prospectus_no_data'
        return result
    
    # 数据库问题
    sql = generate_sql(question, system_prompt)
    if not sql:
        result['error'] = 'SQL生成失败'
        result['answer'] = 'SQL生成失败'
        return result
    
    result['sql'] = sql
    
    for attempt in range(2):
        success, query_result, error_msg = execute_sql(sql)
        if success:
            result['answer'] = format_answer(question, sql, query_result)
            return result
        if attempt == 0:
            sql = generate_sql_with_retry(question, system_prompt, error_msg)
            if sql:
                result['sql'] = sql
    
    result['error'] = error_msg
    result['answer'] = f"查询失败: {error_msg}"
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    questions = load_questions()
    results = load_results()
    
    # 找出失败的题目
    failed_ids = [qid for qid, r in results.items() if r.get('error')]
    print(f"总题数: {len(results)}")
    print(f"失败题数: {len(failed_ids)}")
    print(f"准备重新处理...")
    
    system_prompt = build_system_prompt()
    
    improved = 0
    still_failed = 0
    
    for i, qid in enumerate(failed_ids):
        q = questions[qid]
        print(f"\r[{i+1}/{len(failed_ids)}] ID:{qid} 处理中...", end='', flush=True)
        
        try:
            new_result = process_question(q, system_prompt)
            
            if not new_result.get('error'):
                # 成功了！
                results[qid] = new_result
                improved += 1
                print(f"\r[{i+1}/{len(failed_ids)}] ID:{qid} ✅ 恢复成功!")
            else:
                still_failed += 1
        
        except Exception as e:
            still_failed += 1
    
    # 保存
    save_results(results)
    
    total = len(results)
    success = sum(1 for r in results.values() if not r.get('error'))
    
    print(f"\n\n=== 处理完成 ===")
    print(f"原失败: {len(failed_ids)} 道")
    print(f"恢复成功: {improved} 道")
    print(f"仍然失败: {still_failed} 道")
    print(f"当前成功率: {success}/{total} ({success/total*100:.1f}%)")


if __name__ == '__main__':
    main()
