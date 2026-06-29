# 工单编号：人工智能NLP-Agent数字人项目-基金问答智能体任务
"""
PDF招股书解析模块 - 从PDF文本中提取答案
"""
import os
import re
import subprocess

# PDF文本目录
TXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pdf_txt_file')

# 缓存：公司名 -> 文件路径
_company_files = {}


def load_company_index():
    """建立公司名到文件的索引（读取整个文件）"""
    global _company_files
    if _company_files:
        return
    
    if not os.path.exists(TXT_DIR):
        return
    
    for filename in os.listdir(TXT_DIR):
        if not filename.endswith('.txt'):
            continue
        
        filepath = os.path.join(TXT_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            matches = re.findall(r'([\u4e00-\u9fa5]{2,20}(?:有限公司|股份有限公司))', content)
            for match in matches:
                if len(match) >= 4 and match not in _company_files:
                    _company_files[match] = filepath
        except:
            continue


def find_company_file(company_name):
    """根据公司名查找对应的文本文件"""
    load_company_index()
    
    # 精确匹配
    if company_name in _company_files:
        return _company_files[company_name]
    
    # 模糊匹配
    for name, path in _company_files.items():
        if company_name in name or name in company_name:
            return path
    
    # grep搜索
    try:
        result = subprocess.run(
            ['grep', '-rl', company_name, TXT_DIR],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except:
        pass
    
    return None


def extract_relevant_content(filepath, question, max_chars=3000):
    """从文件中提取与问题相关的内容"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # 根据问题类型确定搜索关键词
        search_keywords = []
        
        if '法定代表人' in question:
            search_keywords = ['法定代表人', '代表人']
        elif '发起人' in question:
            search_keywords = ['发起人', '发起设立', '设立时']
        elif '部门' in question or '研发' in question:
            search_keywords = ['部门', '研发', '技术中心', '研发机构']
        elif '专利' in question:
            search_keywords = ['专利', '发明专利', '实用新型']
        elif '注册资本' in question or '股本' in question:
            search_keywords = ['注册资本', '股本', '万元']
        elif '控股股东' in question or '持有' in question:
            search_keywords = ['控股股东', '持股', '持有股份']
        elif '营业收入' in question or '净利润' in question:
            search_keywords = ['营业收入', '净利润', '利润总额']
        elif '总资产' in question or '周转率' in question:
            search_keywords = ['总资产', '周转率']
        elif '存货' in question:
            search_keywords = ['存货']
        elif '市场占有率' in question:
            search_keywords = ['市场占有率', '市场份额']
        elif '经营模式' in question:
            search_keywords = ['经营模式', '业务模式']
        elif '竞争优势' in question:
            search_keywords = ['竞争优势', '核心竞争力']
        elif '募集资金' in question or '投资项目' in question:
            search_keywords = ['募集资金', '投资项目', '募投项目']
        elif '员工' in question:
            search_keywords = ['员工', '人员', '职工']
        else:
            search_keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', question)
        
        # 搜索包含关键词的行
        relevant = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if len(line_stripped) < 3:
                continue
            
            if any(kw in line_stripped for kw in search_keywords if len(kw) >= 2):
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                context = '\n'.join(l.strip() for l in lines[start:end])
                relevant.append(context)
        
        if relevant:
            result = '\n\n'.join(relevant[:10])
            return result[:max_chars]
        
        return content[:max_chars]
    
    except:
        return None


def answer_prospectus_question(question):
    """从招股书中查找答案"""
    matches = re.findall(r'([\u4e00-\u9fa5]{2,20}(?:有限公司|股份有限公司))', question)
    
    company_name = None
    for match in matches:
        if len(match) >= 4:
            company_name = match
            break
    
    if not company_name:
        return None
    
    filepath = find_company_file(company_name)
    if not filepath:
        return None
    
    content = extract_relevant_content(filepath, question)
    return content


if __name__ == '__main__':
    load_company_index()
    print(f"已索引 {len(_company_files)} 家公司")
    
    test_questions = [
        '华瑞电器股份有限公司的法定代表人是谁',
        '湖南长远锂科股份有限公司变更设立时作为发起人的法人有哪些',
    ]
    
    for q in test_questions:
        content = answer_prospectus_question(q)
        print(f"\n问题: {q}")
        if content:
            print(f"找到内容: {len(content)}字")
            for line in content.split('\n'):
                for kw in ['法定代表人', '发起人']:
                    if kw in line:
                        print(f"  → {line.strip()[:60]}")
                        break
        else:
            print("未找到内容")
