# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
"""
数据库操作层：MySQL增删查改（每次查询新建连接，避免连接断开）
"""

import pymysql
from config import DB_CONFIG, TABLE_NAME, CHAT_TABLE


def _get_conn():
    """每次创建新连接，避免连接断开问题"""
    return pymysql.connect(
        **DB_CONFIG,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def add(member, date_str, type_, category, description, amount):
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        sql = f"""
            INSERT INTO {TABLE_NAME}
            (member, date, type, category, description, amount)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (member, date_str, type_, category, description, amount))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def query(member=None, date_str=None, month=None,
          category=None, description=None, type_=None):
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        conds, params = [], []

        if member:
            conds.append("member = %s"); params.append(member)
        if type_:
            conds.append("type = %s"); params.append(type_)
        if category:
            conds.append("category LIKE %s"); params.append(f"%{category}%")
        if description:
            conds.append("description LIKE %s"); params.append(f"%{description}%")
        if date_str:
            conds.append("date = %s"); params.append(date_str)
        if month:
            conds.append("DATE_FORMAT(date, '%%Y-%%m') = %s"); params.append(month)

        where = " AND ".join(conds) if conds else "1=1"
        sql = f"SELECT * FROM {TABLE_NAME} WHERE {where} ORDER BY date DESC, id DESC"
        cursor.execute(sql, params)
        return cursor.fetchall()
    finally:
        conn.close()


def find_by_description(description, member=None):
    conds = ["description LIKE %s"]
    params = [f"%{description}%"]
    if member:
        conds.append("member = %s"); params.append(member)
    where = " AND ".join(conds)
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        sql = f"SELECT * FROM {TABLE_NAME} WHERE {where} ORDER BY date DESC"
        cursor.execute(sql, params)
        return cursor.fetchall()
    finally:
        conn.close()


def get_by_id(record_id):
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {TABLE_NAME} WHERE id = %s", (record_id,))
        return cursor.fetchone()
    finally:
        conn.close()


def delete(record_id):
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {TABLE_NAME} WHERE id = %s", (record_id,))
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def update(record_id, **kwargs):
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        parts, vals = [], []
        # 字段白名单：只允许更新这些字段，防止SQL注入
        ALLOWED_FIELDS = {"member", "date", "type", "category", "description", "amount"}
        for k, v in kwargs.items():
            if v is not None and k in ALLOWED_FIELDS:
                parts.append(f"{k} = %s"); vals.append(v)
        if not parts:
            return 0
        vals.append(record_id)
        sql = f"UPDATE {TABLE_NAME} SET {', '.join(parts)} WHERE id = %s"
        cursor.execute(sql, vals)
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def get_monthly_summary(member=None, month=None):
    if not month:
        from datetime import datetime
        month = datetime.now().strftime("%Y-%m")
    records = query(member=member, month=month)
    total = sum(float(r['amount']) for r in records)
    return {"month": month, "member": member or "全家", "total": total, "records": records, "count": len(records)}


def today_summary(date_str=None):
    """指定日期或今日收支汇总"""
    from datetime import datetime
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    records = query(date_str=date_str)
    expense = sum(float(r['amount']) for r in records if r['type'] == '支出')
    income = sum(float(r['amount']) for r in records if r['type'] == '收入')
    return {"date": date_str, "expense": expense, "income": income, "count": len(records)}


def monthly_category_summary(month=None):
    from datetime import datetime
    if not month:
        month = datetime.now().strftime("%Y-%m")
    records = query(month=month, type_="支出")
    cat_map = {}
    for r in records:
        cat = r.get("category") or "其他"
        cat_map[cat] = cat_map.get(cat, 0) + float(r["amount"])
    total = sum(cat_map.values())
    categories = []
    for cat, amount in sorted(cat_map.items(), key=lambda x: -x[1]):
        pct = round(amount / total * 100, 1) if total > 0 else 0
        categories.append({"name": cat, "amount": amount, "percent": pct})
    return {"month": month, "total": total, "count": len(records), "categories": categories}


# ---- 聊天记录 ----
def save_chat(role, content):
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"INSERT INTO {CHAT_TABLE} (role, content) VALUES (%s, %s)",
            (role, content)
        )
        conn.commit()
    finally:
        conn.close()


def get_chats(limit=50):
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT role, content, created_at FROM {CHAT_TABLE} ORDER BY id DESC LIMIT %s",
            (limit,)
        )
        rows = list(cursor.fetchall())
        rows.reverse()
        return rows
    finally:
        conn.close()


def clear_chats():
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {CHAT_TABLE}")
        conn.commit()
    finally:
        conn.close()
