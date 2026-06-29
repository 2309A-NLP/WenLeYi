# 工单编号：人工智能NLP-Agent数字人项目-基金问答智能体任务
"""
交互式问答界面 - 输入问题，实时返回答案
"""
import sys
import os
import difflib

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_builder import build_system_prompt
from sql_generator import generate_sql, generate_sql_with_retry
from sql_executor import execute_sql, format_answer
from config import MAX_RETRY

# ========== SQL缓存 ==========
# 相似问题复用SQL，避免重复调LLM生成SQL
_sql_cache = {}  # {normalized_question: sql}
_SQL_CACHE_SIMILARITY_THRESHOLD = 0.8  # 相似度阈值（0-1，越高越严格）


def _normalize_question(question):
    """标准化问题用于缓存匹配：去空白、统一标点、转小写"""
    import re
    q = question.strip().lower()
    q = re.sub(r'\s+', ' ', q)          # 多个空白合并为一个空格
    q = q.replace('？', '?').replace('，', ',').replace('。', '.')
    return q


def _find_cached_sql(question):
    """从缓存中查找相似问题的SQL，返回(sql, matched_question)或(None, None)"""
    normalized = _normalize_question(question)
    # 1. 精确匹配
    if normalized in _sql_cache:
        return _sql_cache[normalized], normalized
    # 2. 模糊匹配（difflib序列匹配）
    if _sql_cache:
        cached_keys = list(_sql_cache.keys())
        matches = difflib.get_close_matches(
            normalized, cached_keys, n=1, cutoff=_SQL_CACHE_SIMILARITY_THRESHOLD
        )
        if matches:
            return _sql_cache[matches[0]], matches[0]
    return None, None


def _cache_sql(question, sql):
    """将SQL存入缓存"""
    normalized = _normalize_question(question)
    _sql_cache[normalized] = sql


def ask(question):
    """
    问答函数 - 输入问题，返回答案
    
    Args:
        question: 自然语言问题
    
    Returns:
        dict: {'question': str, 'answer': str, 'sql': str, 'error': str|None}
    """
    from main import is_prospectus_question
    
    result = {
        'question': question,
        'answer': '',
        'sql': '',
        'error': None
    }
    
    # 检查是否是招股书问题 - 从PDF中查找答案
    if is_prospectus_question(question):
        from pdf_parser import answer_prospectus_question
        from sql_generator import generate_prospectus_answer
        
        content = answer_prospectus_question(question)
        if content:
            answer = generate_prospectus_answer(question, content)
            if answer:
                result['answer'] = answer
                return result
        
        result['answer'] = '抱歉，无法从数据库和招股书中找到此问题的答案。'
        result['error'] = 'prospectus_no_data'
        return result
    
    system_prompt = build_system_prompt()
    
    # 先查SQL缓存，相似问题直接复用，避免重复调LLM
    cached_sql, matched_q = _find_cached_sql(question)
    if cached_sql:
        sql = cached_sql
        print(f"[SQL缓存命中] 匹配问题: {matched_q}")
    else:
        # 缓存未命中，调用LLM生成SQL
        sql = generate_sql(question, system_prompt)
        if not sql:
            result['error'] = 'SQL生成失败'
            result['answer'] = '抱歉，无法理解您的问题，请换个说法试试。'
            return result
        # 将新生成的SQL存入缓存
        _cache_sql(question, sql)
    
    result['sql'] = sql
    
    # 执行SQL（带重试）
    for attempt in range(MAX_RETRY + 1):
        success, query_result, error_msg = execute_sql(sql)
        
        if success:
            # 检查是否有实际数据返回
            has_data = query_result and query_result.get('rows') and len(query_result['rows']) > 0
            if has_data:
                result['answer'] = format_answer(question, sql, query_result)
                return result
            elif attempt < MAX_RETRY:
                # 空结果也触发重试，提示LLM用LIKE模糊匹配
                sql = generate_sql_with_retry(question, system_prompt, "查询结果为空，请尝试使用LIKE模糊匹配基金名称")
                if sql:
                    result['sql'] = sql
                continue
            else:
                result['answer'] = format_answer(question, sql, query_result)
                return result
        
        if attempt < MAX_RETRY:
            sql = generate_sql_with_retry(question, system_prompt, error_msg)
            if sql:
                result['sql'] = sql
    
    result['error'] = error_msg
    result['answer'] = f'查询失败: {error_msg}'
    return result


def interactive_mode():
    """交互式问答模式"""
    print("=" * 60)
    print("  基金数据问答系统")
    print("  工单编号: 人工智能NLP-Agent数字人项目-基金问答智能体任务")
    print("=" * 60)
    print("输入问题即可查询，输入 quit/exit 退出")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        
        if not question:
            continue
        
        if question.lower() in ['quit', 'exit', 'q', '退出']:
            print("再见！")
            break
        
        print("正在查询...", end='', flush=True)
        result = ask(question)
        print("\r" + " " * 20 + "\r", end='')  # 清除"正在查询..."
        
        print(f"\n答案: {result['answer']}")
        
        # 显示SQL（调试用）
        if '--debug' in sys.argv:
            print(f"SQL: {result['sql']}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] != '--debug':
        # 命令行单次问答模式
        question = ' '.join(sys.argv[1:])
        result = ask(question)
        print(f"问题: {result['question']}")
        print(f"答案: {result['answer']}")
        if '--debug' in sys.argv:
            print(f"SQL: {result['sql']}")
    else:
        # 交互式模式
        interactive_mode()
