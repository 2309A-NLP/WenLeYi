# -*- coding: utf-8 -*-
"""
博金杯数据库查询模块
功能：让LLM根据用户问题自动生成SQL，然后在SQLite数据库中执行查询
例如：用户问"基金007484的重仓股" → LLM生成SQL → 查询数据库 → 返回结果
包含SQL安全验证，只允许SELECT查询，防止恶意操作
"""

import sqlite3
import os
import re
import logging
import json

logger = logging.getLogger(__name__)

# SQLite数据库文件路径（博金杯比赛数据）
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "数据", "博金杯比赛数据.db")

# SQL安全验证规则
ALLOWED_SQL_KEYWORDS = ['SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'ORDER', 'BY', 'GROUP', 'HAVING', 'LIMIT', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'ON', 'AS', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'LIKE', 'IN', 'BETWEEN', 'IS', 'NULL', 'NOT', 'ASC', 'DESC', 'UNION', 'ALL', 'EXISTS']
FORBIDDEN_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE']

# 数据库表结构描述（给LLM看的，让它知道怎么写SQL）
SCHEMA_DESC = """
博金杯比赛数据.db 包含以下表：

1. 基金基本信息 (fund_basic)
   - 基金代码(TEXT), 基金简称(TEXT), 基金全称(TEXT), 管理人(TEXT), 托管人(TEXT),
     基金类型(TEXT), 成立日期(TEXT), 管理费率(TEXT), 托管费率(TEXT)

2. 基金股票持仓明细 (fund_stock_holdings)
   - 基金代码(TEXT), 基金简称(TEXT), 持仓日期(TEXT), 股票代码(TEXT), 股票名称(TEXT),
     数量(REAL), 市值(REAL), 市值占基金资产净值比(REAL), 第N大重仓股(INTEGER),
     所在证券市场(TEXT), 所属国家(地区)(TEXT), 报告类型(TEXT)

3. 基金债券持仓明细 (fund_bond_holdings)
   - 基金代码(TEXT), 基金简称(TEXT), 持仓日期(TEXT), 债券类型(TEXT), 债券名称(TEXT),
     持债数量(REAL), 持债市值(REAL), 持债市值占基金资产净值比(REAL)

4. 基金可转债持仓明细 (fund_convertible_holdings)
   - 基金代码(TEXT), 基金简称(TEXT), 持仓日期(TEXT), 对应股票代码(TEXT), 债券名称(TEXT)

5. 基金日行情表 (fund_daily_prices)
   - 基金代码(TEXT), 交易日期(TEXT), 单位净值(REAL), 资产净值(REAL)

6. A股票日行情表 (a_stock_daily)
   - 股票代码(TEXT), 交易日(TEXT), 收盘价(元)(REAL), 成交量(股)(REAL)

7. A股公司行业划分表 (stock_industry)
   - 股票代码(TEXT), 一级行业名称(TEXT), 二级行业名称(TEXT)

8. 基金规模变动表 (fund_scale)
   - 基金代码(TEXT), 报告期期末基金总份额(REAL)

9. 基金份额持有人结构 (fund_holders)
   - 基金代码(TEXT), 机构投资者持有的基金份额(REAL), 个人投资者持有的基金份额(REAL)
"""


def query_db(sql, params=None, limit=50):
    """
    执行SQL查询并返回结果
    参数：
        sql: SQL查询语句（只允许SELECT）
        params: SQL参数（防SQL注入）
        limit: 最多返回多少条结果
    返回：(结果列表, 错误信息)
    """
    if not os.path.exists(DB_PATH):
        return None, f"数据库文件不存在: {DB_PATH}"

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # 让结果可以用列名访问
        cur = conn.cursor()
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        rows = cur.fetchmany(limit)  # 最多取limit条
        # 把Row对象转成普通字典
        col_names = [desc[0] for desc in cur.description] if cur.description else []
        result = [dict(zip(col_names, row)) for row in rows]
        conn.close()
        return result, None
    except Exception as e:
        return None, str(e)


def get_llm_generated_sql(question, llm_client):
    """
    让LLM根据用户问题自动生成SQL查询
    流程：用户问题 → 构造prompt → 调用LLM → 提取SQL → 安全验证 → 返回
    """
    prompt = f"""你是数据库专家，根据用户的问题和数据库结构生成SQLite SQL查询。

数据库表结构：
{SCHEMA_DESC}

注意事项：
- 使用标准的SQLite语法
- 实际表名是中文的（基金债券持仓明细），不要用英文别名
- 返回的SQL只包含SELECT查询，不能修改数据
- LIMIT结果最多50条
- 如果用户问"前N个"、"top N"、"前三"等，SQL必须加上 LIMIT N
- 股票代码和基金代码是TEXT类型
- 日期格式为YYYYMMDD，如20210105
- 查询基金持仓时，先用 基金简称 LIKE '%关键词%' 在基金基本信息表中找到基金代码，再用基金代码去持仓表查

示例：
用户问"基金007484的重仓股" → SELECT 股票名称, 市值, 市值占基金资产净值比 FROM 基金股票持仓明细 WHERE 基金代码='007484' ORDER BY 第N大重仓股 LIMIT 10

用户问题：{question}

请只返回SQL语句，不要其他解释。"""

    try:
        from utils import call_llm
        sql = call_llm([
            {"role": "system", "content": "你是一个SQL专家。根据用户问题生成SQLite查询语句。只返回SQL，不要其他内容。"},
            {"role": "user", "content": prompt}
        ], temperature=0.1, max_tokens=500)  # temperature=0.1让LLM尽量稳定输出
    except Exception as e:
        return None, f"LLM生成SQL失败: {e}"

    # 提取SQL：去掉可能的```sql代码块标记
    sql_clean = re.sub(r'```sql|```', '', sql).strip()

    # 安全验证：确保生成的SQL不会破坏数据库
    is_valid, error_msg = validate_sql(sql_clean)
    if not is_valid:
        return None, f"SQL验证失败: {error_msg}"

    # 确认是SELECT查询（不能是INSERT/DELETE等）
    if not sql_clean.upper().startswith('SELECT'):
        return None, f"生成的不是查询语句: {sql_clean[:100]}"

    return sql_clean, None


def validate_sql(sql):
    """
    【安全函数】验证SQL查询的安全性
    检查规则：
    1. 必须是SELECT语句
    2. 不能包含INSERT/UPDATE/DELETE/DROP等危险操作
    3. 不能包含多条语句（防止SQL注入）
    4. 不能包含SQL注释
    """
    if not sql or not sql.strip():
        return False, "SQL语句为空"

    sql_upper = sql.upper().strip()

    # 检查是否是SELECT语句
    if not sql_upper.startswith('SELECT'):
        return False, "只允许SELECT查询语句"

    # 检查是否包含危险操作关键字
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(r'\b' + keyword + r'\b', sql_upper):
            return False, f"禁止使用{keyword}操作"

    # 检查是否有分号（可能的多语句注入攻击）
    if ';' in sql and sql.index(';') < len(sql) - 1:
        return False, "不允许执行多条SQL语句"

    # 检查是否包含注释
    if '--' in sql or '/*' in sql:
        return False, "不允许包含SQL注释"

    return True, None


def format_db_results(rows, question):
    """
    将数据库查询结果格式化为自然语言
    例如：把 [{股票名称: 贵州茅台, 市值: 1000}] 转成 "查询结果如下：1. 股票名称: 贵州茅台 | 市值: 1000"
    """
    if not rows:
        return f"关于「{question}」，数据库中没有找到相关数据。"
    if isinstance(rows, str):
        return rows

    result = f"查询结果如下：\n"
    if len(rows) == 1 and len(rows[0]) <= 2:
        # 单行少字段：直接列出
        for k, v in rows[0].items():
            if v is not None:
                result += f"{k}: {v}\n"
        return result

    # 多行数据：编号列出
    for i, row in enumerate(rows[:10], 1):
        vals = [f"{k}: {v}" for k, v in row.items() if v is not None]
        result += f"\n{i}. {' | '.join(vals[:3])}"

    if len(rows) > 10:
        result += f"\n\n...共 {len(rows)} 条结果，仅显示前10条"

    return result
