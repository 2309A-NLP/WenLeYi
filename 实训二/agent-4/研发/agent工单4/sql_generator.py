# 工单编号：人工智能NLP-Agent数字人项目-基金问答智能体任务
"""
SQL生成模块 - 简化版，防止卡住
"""
import re
import requests
from config import (
    XIAOMI_API_KEY, XIAOMI_BASE_URL, XIAOMI_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
)


def call_llm(system_prompt, user_prompt):
    """调用DeepSeek API"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }
    data = {
        'model': DEEPSEEK_MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'temperature': 0.1,
        'max_tokens': 512
    }
    
    try:
        response = requests.post(
            f'{DEEPSEEK_BASE_URL}/chat/completions',
            headers=headers,
            json=data,
            timeout=15
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"\n  API错误: {e}")
        return None


def extract_sql(text):
    """从返回文本中提取SQL"""
    if not text:
        return None
    
    # 清理文本
    text = text.strip()
    
    # 去除首尾引号
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    
    # 尝试提取代码块
    match = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        if sql.upper().startswith('SELECT'):
            return clean_sql(sql)
    
    # 尝试提取完整的SQL语句（支持多行）
    match = re.search(r'(SELECT\s+.+?)(?:;\s*$|\Z)', text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        if sql.upper().startswith('SELECT') and len(sql) > 20:
            return clean_sql(sql)
    
    # 尝试提取以SELECT开头的内容，遇到中文字符停止
    lines = text.split('\n')
    sql_lines = []
    in_sql = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith('SELECT'):
            in_sql = True
        if in_sql:
            # 遇到纯中文行则停止
            if re.match(r'^[\u4e00-\u9fa5]+$', line):
                break
            # 跳过包含中文的行
            if re.search(r'[\u4e00-\u9fa5]{3,}', line) and not line.upper().startswith('SELECT'):
                continue
            sql_lines.append(line)
    
    if sql_lines:
        sql = ' '.join(sql_lines).strip()
        if sql.upper().startswith('SELECT') and len(sql) > 20:
            return clean_sql(sql)
    
    # 如果原文以SELECT开头，直接返回
    if text.upper().startswith('SELECT'):
        return clean_sql(text)
    
    return None


def clean_sql(sql):
    """清理SQL，去除末尾多余内容"""
    if not sql:
        return sql
    
    # 去除末尾的分号和逗号
    sql = sql.rstrip(';').rstrip(',').strip()
    
    # 去除末尾的中文注释（不在引号内的中文文本）
    # 策略：找到最后一个单引号，如果其后的文本以中文开头，则认为是注释并去掉
    # 这样不会误删SQL中的中文列名/表名
    last_quote = sql.rfind("'")
    if last_quote >= 0:
        after_quote = sql[last_quote + 1:].strip()
        if after_quote and re.match(r'[\u4e00-\u9fa5\uff0c\u3001\u3002\uff01\uff1f\u300a\u300b]', after_quote):
            sql = sql[:last_quote + 1].strip()
    
    return sql


def generate_sql(question, system_prompt):
    """生成SQL"""
    from prompt_builder import build_user_prompt
    user_prompt = build_user_prompt(question)
    
    response = call_llm(system_prompt, user_prompt)
    sql = extract_sql(response)
    return sql


def generate_sql_with_retry(question, system_prompt, error_msg):
    """带错误反馈的重试"""
    from prompt_builder import build_user_prompt
    user_prompt = build_user_prompt(question) + f"\n\n上次SQL报错: {error_msg}\n请修正后重新生成SQL:"
    
    response = call_llm(system_prompt, user_prompt)
    sql = extract_sql(response)
    return sql


def generate_prospectus_answer(question, content):
    """根据招股书内容生成答案"""
    system_prompt = """你是金融文档分析专家。根据提供的招股书内容回答问题。

规则:
1. 只根据提供的内容回答，不要编造
2. 如果内容中没有相关信息，回答"未找到相关信息"
3. 直接给出答案，不要解释"""
    
    # 限制内容长度，避免API超时
    content = content[:2000]
    
    user_prompt = f"""招股书内容:
{content}

问题: {question}

答案:"""
    
    response = call_llm(system_prompt, user_prompt)
    return response
