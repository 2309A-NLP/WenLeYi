# 工单编号：人工智能NLP-Agent数字人项目-基金问答智能体任务
"""
PDF表格提取脚本 - 提取PDF中的表格内容并更新到txt文件
"""
import os
import re
import pymupdf

PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pdf')
TXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pdf_txt_file')


def extract_tables_from_pdf(pdf_path):
    """从PDF中提取表格内容"""
    tables = []
    try:
        doc = pymupdf.open(pdf_path)
        for page_num, page in enumerate(doc):
            # 尝试提取表格
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for tab in tabs.tables:
                    # 提取表格数据
                    data = tab.extract()
                    if data:
                        # 转换为文本格式
                        table_text = []
                        for row in data:
                            # 过滤None值
                            row_text = [str(cell) if cell else '' for cell in row]
                            table_text.append(' | '.join(row_text))
                        tables.append('\n'.join(table_text))
        doc.close()
    except Exception as e:
        print(f"  提取表格失败: {e}")
    return tables


def update_txt_with_tables(txt_path, tables):
    """将表格内容更新到txt文件"""
    if not tables:
        return False
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找表格占位符并替换
        table_pattern = re.compile(r'<\|TABLE_\d+_\d+\.xlsx\|>')
        
        table_index = 0
        def replace_table(match):
            nonlocal table_index
            if table_index < len(tables):
                result = f"\n[表格内容]:\n{tables[table_index]}\n"
                table_index += 1
                return result
            return match.group()
        
        new_content = table_pattern.sub(replace_table, content)
        
        # 如果有新内容，写回文件
        if new_content != content:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
    except Exception as e:
        print(f"  更新文件失败: {e}")
    
    return False


def process_all_files():
    """处理所有PDF和txt文件"""
    if not os.path.exists(PDF_DIR) or not os.path.exists(TXT_DIR):
        print("PDF或TXT目录不存在")
        return
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.PDF') or f.endswith('.pdf')]
    txt_files = [f for f in os.listdir(TXT_DIR) if f.endswith('.txt')]
    
    print(f"PDF文件: {len(pdf_files)}个")
    print(f"TXT文件: {len(txt_files)}个")
    
    updated = 0
    for pdf_name in pdf_files:
        # 找对应的txt文件
        base_name = os.path.splitext(pdf_name)[0]
        txt_name = None
        for txt in txt_files:
            if base_name in txt:
                txt_name = txt
                break
        
        if not txt_name:
            continue
        
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        txt_path = os.path.join(TXT_DIR, txt_name)
        
        print(f"\n处理: {pdf_name[:30]}...")
        
        # 提取表格
        tables = extract_tables_from_pdf(pdf_path)
        print(f"  提取到 {len(tables)} 个表格")
        
        # 更新txt文件
        if tables:
            if update_txt_with_tables(txt_path, tables):
                updated += 1
                print(f"  已更新txt文件")
    
    print(f"\n完成! 更新了 {updated} 个文件")


if __name__ == '__main__':
    process_all_files()
