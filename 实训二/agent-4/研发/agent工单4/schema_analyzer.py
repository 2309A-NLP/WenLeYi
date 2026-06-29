# 工单编号：人工智能NLP-Agent数字人项目-基金问答智能体任务
"""
数据库结构分析模块 - 提取表结构、字段信息、样例数据
"""
import sqlite3
from config import DB_PATH


def get_all_tables():
    """获取所有表名"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [t[0] for t in cursor.fetchall()]
    conn.close()
    return tables


def get_table_schema(table_name):
    """获取单张表的字段信息"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    cursor.execute(f'PRAGMA table_info([{table_name}])')
    columns = cursor.fetchall()
    conn.close()
    return [{'name': c[1], 'type': c[2], 'notnull': c[3], 'pk': c[5]} for c in columns]


def get_table_sample(table_name, limit=3):
    """获取表的样例数据"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM [{table_name}] LIMIT {limit}')
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]
    conn.close()
    return col_names, rows


def get_table_count(table_name):
    """获取表的行数"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    cursor.execute(f'SELECT COUNT(*) FROM [{table_name}]')
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_full_schema_text():
    """生成完整的数据库schema文本，用于prompt"""
    tables = get_all_tables()
    schema_parts = []
    
    for table_name in tables:
        columns = get_table_schema(table_name)
        col_names, sample_rows = get_table_sample(table_name, limit=2)
        
        # 构建CREATE TABLE语句
        col_defs = []
        for col in columns:
            pk_str = ' PRIMARY KEY' if col['pk'] else ''
            col_defs.append(f'    "{col["name"]}" {col["type"]}{pk_str}')
        
        create_sql = f"CREATE TABLE [{table_name}] (\n" + ",\n".join(col_defs) + "\n);"
        
        # 样例数据
        sample_text = ""
        if sample_rows:
            sample_text = f"\n-- 样例数据:\n"
            for row in sample_rows[:2]:
                vals = [f"'{v}'" if isinstance(v, str) else str(v) for v in row]
                sample_text += f"-- {', '.join(vals)}\n"
        
        schema_parts.append(f"{create_sql}{sample_text}")
    
    return "\n\n".join(schema_parts)


def get_relationship_text():
    """生成表关联关系文本"""
    return """
关联关系说明:
1. 基金基本信息.基金代码 → 基金日行情表.基金代码
2. 基金基本信息.基金代码 → 基金股票持仓明细.基金代码
3. 基金基本信息.基金代码 → 基金债券持仓明细.基金代码
4. 基金基本信息.基金代码 → 基金可转债持仓明细.基金代码
5. 基金基本信息.基金代码 → 基金规模变动表.基金代码
6. 基金基本信息.基金代码 → 基金份额持有人结构.基金代码
7. 基金股票持仓明细.股票代码 → A股票日行情表.股票代码
8. 基金股票持仓明细.股票代码 → A股公司行业划分表.股票代码
9. 基金可转债持仓明细.对应股票代码 → A股票日行情表.股票代码

日期格式注意:
- 基金日行情表/A股票日行情表/港股票日行情表/A股公司行业划分表: YYYYMMDD (如'20210105')
- 持仓明细表: YYYYMMDD (如'20201231')
- 基金规模变动表/基金份额持有人结构: YYYY-MM-DD HH:MM:SS (如'2019-03-31 00:00:00')

涨跌幅计算公式:
股票涨跌幅 = (收盘价 - 前一日收盘价) / 前一日收盘价 * 100%
"""


if __name__ == '__main__':
    # 测试
    tables = get_all_tables()
    print(f"数据库共 {len(tables)} 张表:")
    for t in tables:
        count = get_table_count(t)
        print(f"  {t}: {count}行")
    
    print("\n完整Schema:")
    print(get_full_schema_text()[:2000])
