# 工单编号：人工智能NLP-Agent数字人项目-基金问答智能体任务
"""
SQL执行模块 - 加了查询超时限制
索引优化：首次连接时自动为常用查询字段创建索引，加速基金查询
"""
import sqlite3
import threading
from config import DB_PATH, SQL_TIMEOUT as QUERY_TIMEOUT


_indexes_initialized = False


def ensure_indexes():
    """为常用查询字段创建索引，加速基金相关查询（仅首次执行）"""
    global _indexes_initialized
    if _indexes_initialized:
        return

    # 定义每张表需要建索引的字段
    index_definitions = [
        # 基金基本信息
        ("idx_fund_basic_code",         "基金基本信息", "基金代码"),
        ("idx_fund_basic_name",         "基金基本信息", "基金简称"),
        ("idx_fund_basic_type",         "基金基本信息", "基金类型"),
        ("idx_fund_basic_company",      "基金基本信息", "基金管理人"),
        # 基金日行情表
        ("idx_fund_daily_code",         "基金日行情表", "基金代码"),
        ("idx_fund_daily_date",         "基金日行情表", "交易日"),
        ("idx_fund_daily_code_date",    "基金日行情表", "基金代码, 交易日"),
        # 基金股票持仓明细
        ("idx_stock_hold_code",         "基金股票持仓明细", "基金代码"),
        ("idx_stock_hold_stock_code",   "基金股票持仓明细", "股票代码"),
        ("idx_stock_hold_stock_name",   "基金股票持仓明细", "股票名称"),
        ("idx_stock_hold_date",         "基金股票持仓明细", "持仓日期"),
        ("idx_stock_hold_code_date",    "基金股票持仓明细", "基金代码, 持仓日期"),
        # 基金债券持仓明细
        ("idx_bond_hold_code",          "基金债券持仓明细", "基金代码"),
        ("idx_bond_hold_date",          "基金债券持仓明细", "持仓日期"),
        # 基金可转债持仓明细
        ("idx_cb_hold_code",            "基金可转债持仓明细", "基金代码"),
        ("idx_cb_hold_date",            "基金可转债持仓明细", "持仓日期"),
        ("idx_cb_hold_stock_code",      "基金可转债持仓明细", "对应股票代码"),
        # 基金规模变动表
        ("idx_scale_code",              "基金规模变动表", "基金代码"),
        ("idx_scale_date",              "基金规模变动表", "统计截止日期"),
        # 基金份额持有人结构
        ("idx_holder_code",             "基金份额持有人结构", "基金代码"),
        ("idx_holder_date",             "基金份额持有人结构", "统计截止日期"),
        # A股票日行情表
        ("idx_a_stock_code",            "A股票日行情表", "股票代码"),
        ("idx_a_stock_date",            "A股票日行情表", "交易日"),
        ("idx_a_stock_code_date",       "A股票日行情表", "股票代码, 交易日"),
        ("idx_a_stock_name",            "A股票日行情表", "股票名称"),
        # A股公司行业划分表
        ("idx_industry_code",           "A股公司行业划分表", "股票代码"),
        ("idx_industry_name",           "A股公司行业划分表", "股票名称"),
        ("idx_industry_sector",         "A股公司行业划分表", "行业名称"),
        # 港股票日行情表
        ("idx_hk_stock_code",           "港股票日行情表", "股票代码"),
        ("idx_hk_stock_date",           "港股票日行情表", "交易日"),
    ]

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()

        # 获取已有索引，避免重复创建
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        existing = {row[0] for row in cursor.fetchall()}

        created = 0
        for idx_name, table, columns in index_definitions:
            if idx_name in existing:
                continue
            try:
                cursor.execute(
                    f'CREATE INDEX IF NOT EXISTS [{idx_name}] ON [{table}] ({columns})'
                )
                created += 1
            except sqlite3.OperationalError:
                # 表或字段不存在时静默跳过（如数据库尚未导入数据）
                pass

        if created > 0:
            conn.commit()

        _indexes_initialized = True
    except Exception:
        # 索引创建失败不应影响正常查询
        pass
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def execute_sql(sql):
    """执行SQL查询，带超时限制"""
    ensure_indexes()

    if not sql or not sql.strip():
        return False, None, "SQL为空"
    
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith('SELECT'):
        return False, None, "只允许SELECT查询"
    
    dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE']
    for kw in dangerous:
        if kw in sql_upper:
            return False, None, f"禁止{kw}操作"
    
    result = [None]
    error = [None]
    conn_ref = [None]  # 追踪连接引用，防止超时时连接泄漏
    
    def run_query():
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5)
            conn_ref[0] = conn  # 保存连接引用，供主线程超时清理
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            result[0] = {'columns': columns, 'rows': rows}
        except sqlite3.OperationalError as e:
            error[0] = f"SQL执行错误: {str(e)}"
        except Exception as e:
            error[0] = f"未知错误: {str(e)}"
        finally:
            # 确保连接总是被关闭，防止资源泄漏
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    
    t = threading.Thread(target=run_query)
    t.daemon = True
    t.start()
    t.join(timeout=QUERY_TIMEOUT)
    
    if t.is_alive():
        # 超时后尝试从主线程关闭连接，防止连接泄漏
        if conn_ref[0]:
            try:
                conn_ref[0].close()
            except Exception:
                pass
        return False, None, f"查询超时(>{QUERY_TIMEOUT}秒)"
    
    if error[0]:
        return False, None, error[0]
    
    if result[0]:
        return True, result[0], None
    
    return False, None, "查询无结果"


def format_result(result, question=''):
    """格式化查询结果"""
    if not result or not result.get('rows'):
        return "未找到相关数据，请尝试换个说法或使用更完整的基金名称"
    
    columns = result['columns']
    rows = result['rows']
    
    # 去重
    seen = set()
    unique_rows = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            unique_rows.append(row)
    rows = unique_rows
    
    # 单行单列：直接返回值
    if len(rows) == 1 and len(columns) == 1:
        return format_value(rows[0][0], question)
    
    # 单行多列：返回所有字段
    if len(rows) == 1:
        parts = [f"{col}: {format_value(val, question)}" for col, val in zip(columns, rows[0])]
        return ", ".join(parts)
    
    # 多行：全部分点显示，每个结果换行
    parts = []
    for i, row in enumerate(rows[:20], 1):
        if len(columns) == 1:
            parts.append(f"{i}. {format_value(row[0], question)}")
        elif len(columns) == 2:
            parts.append(f"{i}. {format_value(row[0], question)} | {columns[1]}: {format_value(row[1], question)}")
        else:
            row_parts = [f"{col}: {format_value(val, question)}" for col, val in zip(columns, row)]
            parts.append(f"{i}. {' | '.join(row_parts)}")
    
    result_text = "\n".join(parts)
    
    if len(rows) > 20:
        result_text += f"\n... (共{len(rows)}条，仅显示前20条)"
    
    return result_text

def format_value(val, question=''):
    """格式化单个值"""
    if val is None:
        return "无"
    if isinstance(val, float):
        if any(kw in question for kw in ['百分', '%', '比例', '占比']):
            return f"{val:.2f}%"
        if abs(val) >= 10000:
            return f"{val:,.2f}"
        if val == int(val):
            return str(int(val))
        return f"{val:.2f}"
    if isinstance(val, int):
        if abs(val) >= 10000:
            return f"{val:,}"
    return str(val)


def format_answer(question, sql, result, error_msg=None):
    """格式化最终答案"""
    if error_msg:
        return f"查询失败: {error_msg}"
    if result is None:
        return "查询无结果"
    return format_result(result, question)
