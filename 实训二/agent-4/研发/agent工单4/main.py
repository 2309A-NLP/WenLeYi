# 工单编号：人工智能NLP-Agent数字人项目-基金问答智能体任务
"""
主流程模块 - 支持断点续跑
"""
import json
import os
from config import (
    QUESTION_FILE, OUTPUT_FILE, ERROR_LOG,
    OUTPUT_DIR, MAX_RETRY
)
from prompt_builder import build_system_prompt
from sql_generator import generate_sql, generate_sql_with_retry
from sql_executor import execute_sql, format_answer


def load_questions(filepath):
    """加载问题文件"""
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_existing_results():
    """加载已有的结果"""
    results = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    results[r['id']] = r
    return results


def save_results(results):
    """保存所有结果"""
    sorted_results = sorted(results.values(), key=lambda x: x['id'])
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for r in sorted_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def save_error(entry):
    """记录错误日志"""
    with open(ERROR_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def is_prospectus_question(question):
    """判断是否是招股书问题"""
    keywords = ['招股', '公司简介', '经营范围', '竞争优势', '专利', '员工人数',
                '控股股东', '发起人', '总资产周转率', '存货', '流动资产', '研发部门',
                '发行前', '零售价格', '经营模式', '军用领域', '募集资金', '投资项目',
                '市场占有率', '机器设备', '销售费用', '法定代表人', '经营业绩',
                '政府补助', '高新技术产品', '本科以上', '业务包括', '出口退税',
                '收入比重', '营业收入', '净利润', '财务指标', '财务表',
                '部门', '产品及服务', '成新率', '固定资产', '补助计入',
                '成立时', '变更设立', '负责产品研发', '哪个部门',
                '优势', '核心产品', '主要业务', '经营下降', '增长幅度',
                '占比多少', '比例是多少', '金额为多少', '持有股份',
                '总股本', '注册资本', '注册地址', '办公地址']
    return any(kw in question for kw in keywords)


def process_single_question(question_data, system_prompt):
    """处理单个问题"""
    q_id = question_data['id']
    question = question_data['question']
    
    result = {
        'id': q_id,
        'question': question,
        'answer': '',
        'sql': '',
        'error': None
    }
    
    # 招股书问题 - 从PDF中查找答案
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
    
    # 生成SQL
    sql = generate_sql(question, system_prompt)
    if not sql:
        result['error'] = 'SQL生成失败'
        result['answer'] = 'SQL生成失败'
        return result
    
    result['sql'] = sql
    
    # 执行SQL（带重试）
    for attempt in range(MAX_RETRY + 1):
        success, query_result, error_msg = execute_sql(sql)
        
        if success:
            result['answer'] = format_answer(question, sql, query_result)
            return result
        
        if attempt < MAX_RETRY:
            sql = generate_sql_with_retry(question, system_prompt, error_msg)
            if sql:
                result['sql'] = sql
    
    result['error'] = error_msg
    result['answer'] = f"查询失败: {error_msg}"
    return result


def main(start_idx=0, end_idx=None, batch_size=10):
    """主函数 - 支持断点续跑"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载问题
    print("加载问题文件...")
    questions = load_questions(QUESTION_FILE)
    print(f"共 {len(questions)} 道题")
    
    # 加载已有结果（断点续跑）
    existing_results = load_existing_results()
    print(f"已有结果: {len(existing_results)} 条")
    
    if end_idx is None:
        end_idx = len(questions)
    
    # 构建系统提示词
    print("构建Prompt...")
    system_prompt = build_system_prompt()
    print(f"Prompt长度: {len(system_prompt)} 字符")
    
    # 处理问题
    processed = 0
    skipped = 0
    
    for i in range(start_idx, min(end_idx, len(questions))):
        q = questions[i]
        q_id = q['id']
        
        # 跳过已处理的
        if q_id in existing_results:
            skipped += 1
            continue
        
        # 跳过招股书问题
        if is_prospectus_question(q['question']):
            skip_result = {
                'id': q_id,
                'question': q['question'],
                'answer': '[招股书问题，需PDF数据]',
                'sql': '',
                'error': 'prospectus_question'
            }
            existing_results[q_id] = skip_result
            skipped += 1
            print(f"\r[{i+1}/{end_idx}] ID:{q_id} - 跳过", end='', flush=True)
            continue
        
        print(f"\r[{i+1}/{end_idx}] ID:{q_id} 处理中...", end='', flush=True)
        
        try:
            result = process_single_question(q, system_prompt)
            existing_results[q_id] = result
            processed += 1
            
            if result['error']:
                save_error(result)
            
        except Exception as e:
            error_result = {
                'id': q_id,
                'question': q['question'],
                'answer': f"处理异常: {str(e)}",
                'sql': '',
                'error': str(e)
            }
            existing_results[q_id] = error_result
            processed += 1
            save_error(error_result)
        
        # 每批保存一次
        if processed % batch_size == 0:
            save_results(existing_results)
            print(f"\n  已保存 {len(existing_results)} 条结果")
    
    # 最终保存
    save_results(existing_results)
    
    # 统计
    total = len(existing_results)
    errors = sum(1 for r in existing_results.values() if r.get('error'))
    print(f"\n\n=== 处理完成 ===")
    print(f"总计: {total} 条结果")
    print(f"本次处理: {processed} 道题")
    print(f"跳过: {skipped} 道题")
    print(f"成功: {total - errors} 道")
    print(f"失败: {errors} 道")
    print(f"结果已保存到: {OUTPUT_FILE}")
    
    return existing_results


if __name__ == '__main__':
    import sys
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(start_idx=start, end_idx=end)
