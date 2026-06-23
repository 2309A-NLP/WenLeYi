# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-Agent 数字人项目-日程提醒智能体任务
数据库操作封装：schedule_remind_info / schedule_remind_log
每次操作独立连接（connection-per-query）
"""
import pymysql
import config


def _get_conn():
    """每次获取新连接，不用singleton"""
    return pymysql.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        database=config.DB_NAME,
        charset=config.DB_CHARSET,
        cursorclass=pymysql.cursors.DictCursor,
    )


# ==================== schedule_remind_info 表 ====================

def add_schedule(schedule_date, schedule_time, content, repeat_rule=None):
    """新增日程，返回新插入的id"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = """INSERT INTO schedule_remind_info
                     (schedule_date, schedule_time, content, repeat_rule)
                     VALUES (%s, %s, %s, %s)"""
            cur.execute(sql, (schedule_date, schedule_time, content, repeat_rule))
            conn.commit()
            return cur.lastrowid
    finally:
        conn.close()


def get_today_schedules(date_str):
    """查询当天有效日程，按时间排序"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = """SELECT id, schedule_date, schedule_time, content, repeat_rule
                     FROM schedule_remind_info
                     WHERE schedule_date = %s AND status = 1
                     ORDER BY schedule_time ASC"""
            cur.execute(sql, (date_str,))
            return cur.fetchall()
    finally:
        conn.close()


def get_schedule_by_id(schedule_id):
    """按ID查询单条日程"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = """SELECT id, schedule_date, schedule_time, content, repeat_rule, status
                     FROM schedule_remind_info WHERE id = %s"""
            cur.execute(sql, (schedule_id,))
            return cur.fetchone()
    finally:
        conn.close()


def delete_schedule(schedule_id):
    """软删除日程（status置0），返回删除前的记录"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            # 先查再删
            cur.execute("SELECT * FROM schedule_remind_info WHERE id = %s AND status = 1",
                        (schedule_id,))
            record = cur.fetchone()
            if record:
                cur.execute("UPDATE schedule_remind_info SET status = 0 WHERE id = %s",
                            (schedule_id,))
                conn.commit()
            return record
    finally:
        conn.close()


def update_schedule(schedule_id, field, value):
    """修改日程字段，返回修改前的记录"""
    allowed = {"content", "schedule_date", "schedule_time", "repeat_rule"}
    if field not in allowed:
        return None
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schedule_remind_info WHERE id = %s AND status = 1",
                        (schedule_id,))
            record = cur.fetchone()
            if record:
                sql = "UPDATE schedule_remind_info SET {} = %s WHERE id = %s".format(field)
                cur.execute(sql, (value, schedule_id))
                conn.commit()
            return record
    finally:
        conn.close()


def get_pending_reminders(date_str, time_str):
    """查询到时需提醒的日程（status=1, remind_status=1, 时间已到）"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = """SELECT id, schedule_date, schedule_time, content, repeat_rule
                     FROM schedule_remind_info
                     WHERE status = 1 AND remind_status = 1
                       AND schedule_date <= %s AND schedule_time <= %s
                     ORDER BY schedule_time ASC"""
            cur.execute(sql, (date_str, time_str))
            return cur.fetchall()
    finally:
        conn.close()


def mark_reminded(schedule_id):
    """标记日程已提醒"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE schedule_remind_info SET remind_status = 0 WHERE id = %s",
                        (schedule_id,))
            conn.commit()
    finally:
        conn.close()


def create_next_cycle(schedule_record):
    """为循环日程创建下一次日程"""
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta

    rule = schedule_record.get("repeat_rule")
    current_date = schedule_record["schedule_date"]
    current_time = schedule_record["schedule_time"]
    content = schedule_record["content"]

    if rule == "daily":
        next_date = current_date + timedelta(days=1)
    elif rule == "weekly":
        next_date = current_date + timedelta(weeks=1)
    elif rule == "monthly":
        next_date = current_date + relativedelta(months=1)
    else:
        return None

    return add_schedule(next_date, current_time, content, rule)


# ==================== schedule_remind_log 表 ====================

def add_remind_log(schedule_id, remind_text):
    """记录提醒日志"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = """INSERT INTO schedule_remind_log (schedule_id, remind_time, remind_text)
                     VALUES (%s, NOW(), %s)"""
            cur.execute(sql, (schedule_id, remind_text))
            conn.commit()
    finally:
        conn.close()


# ==================== 新增：分类查询/搜索/清空 ====================

def format_time_str(t):
    """将timedelta或字符串格式化为HH:MM"""
    from datetime import timedelta
    if isinstance(t, timedelta):
        total = int(t.total_seconds())
        return "{:02d}:{:02d}".format(total // 3600, (total % 3600) // 60)
    return str(t)[:5]


def get_schedules_by_category(category, today_str):
    """按分类查询日程"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if category == "today":
                sql = """SELECT id, schedule_date, schedule_time, content, repeat_rule, remind_status
                         FROM schedule_remind_info
                         WHERE schedule_date = %s AND status = 1 AND remind_status = 1
                         ORDER BY schedule_time ASC"""
                cur.execute(sql, (today_str,))
            elif category == "completed":
                sql = """SELECT id, schedule_date, schedule_time, content, repeat_rule
                         FROM schedule_remind_info
                         WHERE schedule_date = %s AND remind_status = 0 AND status = 1
                         ORDER BY schedule_time ASC"""
                cur.execute(sql, (today_str,))
            elif category == "repeating":
                sql = """SELECT id, schedule_date, schedule_time, content, repeat_rule
                         FROM schedule_remind_info
                         WHERE repeat_rule IS NOT NULL AND status = 1
                         ORDER BY schedule_date ASC, schedule_time ASC"""
                cur.execute(sql)
            elif category == "history":
                sql = """SELECT id, schedule_date, schedule_time, content, repeat_rule, status
                         FROM schedule_remind_info
                         WHERE schedule_date < %s OR status = 0
                         ORDER BY schedule_date DESC, schedule_time DESC
                         LIMIT 50"""
                cur.execute(sql, (today_str,))
            else:
                return []
            rows = cur.fetchall()
            # 格式化时间
            for r in rows:
                r["time_str"] = format_time_str(r.pop("schedule_time"))
                r["date_str"] = str(r.pop("schedule_date", ""))
            return rows
    finally:
        conn.close()


def search_schedules(keyword):
    """全局搜索日程（按内容关键词）"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = """SELECT id, schedule_date, schedule_time, content, repeat_rule, status
                     FROM schedule_remind_info
                     WHERE content LIKE %s
                     ORDER BY schedule_date DESC, schedule_time DESC
                     LIMIT 30"""
            cur.execute(sql, ("%" + keyword + "%",))
            rows = cur.fetchall()
            for r in rows:
                r["time_str"] = format_time_str(r.pop("schedule_time"))
                r["date_str"] = str(r.pop("schedule_date", ""))
            return rows
    finally:
        conn.close()


def clear_all_schedules():
    """清空所有日程数据"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM schedule_remind_info")
            cur.execute("DELETE FROM schedule_remind_log")
            cur.execute("ALTER TABLE schedule_remind_info AUTO_INCREMENT = 1")
            cur.execute("ALTER TABLE schedule_remind_log AUTO_INCREMENT = 1")
            conn.commit()
            return True
    finally:
        conn.close()


def get_category_counts(today_str):
    """获取各分类数量"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            counts = {}
            cur.execute("SELECT COUNT(*) as c FROM schedule_remind_info WHERE schedule_date=%s AND status=1 AND remind_status=1", (today_str,))
            counts["today"] = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) as c FROM schedule_remind_info WHERE schedule_date=%s AND remind_status=0 AND status=1", (today_str,))
            counts["completed"] = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) as c FROM schedule_remind_info WHERE repeat_rule IS NOT NULL AND status=1")
            counts["repeating"] = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) as c FROM schedule_remind_info WHERE schedule_date < %s OR status=0", (today_str,))
            counts["history"] = cur.fetchone()["c"]
            return counts
    finally:
        conn.close()


def get_schedules_by_date(date_str):
    """查询指定日期的所有日程"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = """SELECT id, schedule_date, schedule_time, content, repeat_rule, remind_status, status
                     FROM schedule_remind_info
                     WHERE schedule_date = %s AND status = 1
                     ORDER BY schedule_time ASC"""
            cur.execute(sql, (date_str,))
            rows = cur.fetchall()
            for r in rows:
                r["time_str"] = format_time_str(r.pop("schedule_time"))
                r["date_str"] = str(r.pop("schedule_date"))
            return rows
    finally:
        conn.close()


def get_month_schedule_counts(year, month):
    """获取某月每天的日程数量"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = """SELECT schedule_date, COUNT(*) as cnt
                     FROM schedule_remind_info
                     WHERE YEAR(schedule_date) = %s AND MONTH(schedule_date) = %s AND status = 1
                     GROUP BY schedule_date"""
            cur.execute(sql, (year, month))
            rows = cur.fetchall()
            result = {}
            for r in rows:
                result[str(r["schedule_date"])] = r["cnt"]
            return result
    finally:
        conn.close()


def update_schedule_field(schedule_id, field, value):
    """编辑日程字段（直接更新，用于侧边栏编辑）"""
    allowed = {"content", "schedule_date", "schedule_time", "repeat_rule"}
    if field not in allowed:
        return None
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schedule_remind_info WHERE id = %s AND status = 1", (schedule_id,))
            record = cur.fetchone()
            if record:
                sql = "UPDATE schedule_remind_info SET {} = %s WHERE id = %s".format(field)
                cur.execute(sql, (value, schedule_id))
                conn.commit()
            return record
    finally:
        conn.close()


def update_full_schedule(schedule_id, schedule_date, schedule_time, content, repeat_rule):
    """完整编辑日程（同时更新所有字段）"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schedule_remind_info WHERE id = %s AND status = 1", (schedule_id,))
            record = cur.fetchone()
            if record:
                sql = """UPDATE schedule_remind_info
                         SET schedule_date=%s, schedule_time=%s, content=%s, repeat_rule=%s
                         WHERE id = %s"""
                cur.execute(sql, (schedule_date, schedule_time, content, repeat_rule, schedule_id))
                conn.commit()
            return record
    finally:
        conn.close()


# ==================== V3：回收站/操作日志/批量/冲突/热力图 ====================

def soft_delete_to_recycle(schedule_id):
    """移入回收站（status=2）"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schedule_remind_info WHERE id = %s AND status = 1", (schedule_id,))
            record = cur.fetchone()
            if record:
                cur.execute("UPDATE schedule_remind_info SET status = 2, deleted_at = NOW() WHERE id = %s", (schedule_id,))
                conn.commit()
            return record
    finally:
        conn.close()


def restore_from_recycle(schedule_id):
    """从回收站恢复"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schedule_remind_info WHERE id = %s AND status = 2", (schedule_id,))
            record = cur.fetchone()
            if record:
                cur.execute("UPDATE schedule_remind_info SET status = 1, deleted_at = NULL WHERE id = %s", (schedule_id,))
                conn.commit()
            return record
    finally:
        conn.close()


def get_recycle_bin():
    """获取回收站日程"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT id, schedule_date, schedule_time, content, deleted_at
                         FROM schedule_remind_info WHERE status = 2
                         ORDER BY deleted_at DESC""")
            rows = cur.fetchall()
            for r in rows:
                r["time_str"] = format_time_str(r.pop("schedule_time"))
                r["date_str"] = str(r.pop("schedule_date"))
                r["deleted_str"] = str(r.pop("deleted_at", ""))[:16]
            return rows
    finally:
        conn.close()


def clear_recycle_bin():
    """永久清空回收站"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM schedule_remind_info WHERE status = 2")
            conn.commit()
            return cur.rowcount
    finally:
        conn.close()


def cleanup_old_recycle(days=30):
    """清除超过N天的回收站数据"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM schedule_remind_info WHERE status = 2 AND deleted_at < DATE_SUB(NOW(), INTERVAL %s DAY)", (days,))
            conn.commit()
            return cur.rowcount
    finally:
        conn.close()


def add_operation_log(action_type, schedule_id=None, schedule_content=None, detail=None, source="chat"):
    """记录操作日志"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO schedule_operation_log
                         (action_type, schedule_id, schedule_content, detail, source)
                         VALUES (%s, %s, %s, %s, %s)""",
                        (action_type, schedule_id, schedule_content, detail, source))
            conn.commit()
    finally:
        conn.close()


def get_operation_logs(limit=50, action_type=None):
    """查询操作日志"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if action_type:
                cur.execute("""SELECT id, action_type, schedule_id, schedule_content, detail, source, created_at
                             FROM schedule_operation_log WHERE action_type = %s
                             ORDER BY created_at DESC LIMIT %s""", (action_type, limit))
            else:
                cur.execute("""SELECT id, action_type, schedule_id, schedule_content, detail, source, created_at
                             FROM schedule_operation_log ORDER BY created_at DESC LIMIT %s""", (limit,))
            rows = cur.fetchall()
            for r in rows:
                r["time_str"] = str(r["created_at"])[:16]
            return rows
    finally:
        conn.close()


def batch_delete_schedules(ids):
    """批量删除（移入回收站）"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(ids))
            cur.execute("SELECT id, content FROM schedule_remind_info WHERE id IN ({}) AND status=1".format(placeholders), ids)
            records = cur.fetchall()
            cur.execute("UPDATE schedule_remind_info SET status=2, deleted_at=NOW() WHERE id IN ({}) AND status=1".format(placeholders), ids)
            conn.commit()
            return records
    finally:
        conn.close()


def batch_complete_schedules(ids):
    """批量标记完成"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(ids))
            cur.execute("UPDATE schedule_remind_info SET remind_status=0 WHERE id IN ({}) AND status=1".format(placeholders), ids)
            conn.commit()
            return cur.rowcount
    finally:
        conn.close()


def check_time_conflict(schedule_date, schedule_time, content=None, exclude_id=None):
    """检查时间冲突：同一日期、同一时间、同一内容才算冲突"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if content:
                # 精确匹配：日期+时间+内容完全相同
                sql = """SELECT id, schedule_time, content FROM schedule_remind_info
                         WHERE schedule_date = %s AND status = 1"""
                params = [schedule_date]
                if exclude_id:
                    sql += " AND id != %s"
                    params.append(exclude_id)
                cur.execute(sql, params)
                rows = cur.fetchall()
                conflicts = []
                input_time = str(schedule_time)[:5] if ":" in str(schedule_time) else str(schedule_time)
                input_content = content.strip().lower()
                for r in rows:
                    t = format_time_str(r["schedule_time"])
                    c = r["content"].strip().lower()
                    if t == input_time and c == input_content:
                        conflicts.append({"id": r["id"], "time": t, "content": r["content"]})
                return conflicts
            else:
                # 仅匹配时间
                sql = """SELECT id, schedule_time, content FROM schedule_remind_info
                         WHERE schedule_date = %s AND status = 1"""
                params = [schedule_date]
                if exclude_id:
                    sql += " AND id != %s"
                    params.append(exclude_id)
                cur.execute(sql, params)
                rows = cur.fetchall()
                conflicts = []
                input_h = int(str(schedule_time).split(":")[0]) if ":" in str(schedule_time) else -1
                for r in rows:
                    t = format_time_str(r["schedule_time"])
                    exist_h = int(t.split(":")[0])
                    if input_h == exist_h:
                        conflicts.append({"id": r["id"], "time": t, "content": r["content"]})
                return conflicts
    finally:
        conn.close()


def get_heatmap_data(year, month):
    """获取热力图数据（每天日程数量）"""
    return get_month_schedule_counts(year, month)


# ==================== 建表 ====================

def init_tables():
    """初始化数据库表（如果不存在则创建）"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schedule_remind_info (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    schedule_date DATE NOT NULL COMMENT '日程日期',
                    schedule_time TIME NOT NULL COMMENT '日程时间',
                    content VARCHAR(500) NOT NULL COMMENT '事项内容',
                    repeat_rule VARCHAR(50) DEFAULT NULL COMMENT '循环规则 daily/weekly/monthly',
                    remind_status TINYINT DEFAULT 1 COMMENT '1=待提醒 0=已提醒',
                    status TINYINT DEFAULT 1 COMMENT '1=有效 0=已删除 2=回收站',
                    deleted_at DATETIME DEFAULT NULL COMMENT '移入回收站时间',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='日程提醒信息表'
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schedule_remind_log (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    schedule_id INT NOT NULL COMMENT '关联schedule_remind_info.id',
                    remind_time DATETIME NOT NULL COMMENT '实际提醒时间',
                    remind_text VARCHAR(500) COMMENT '提醒话术',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='提醒记录表'
            """)
            # 操作日志表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schedule_operation_log (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    action_type VARCHAR(20) NOT NULL COMMENT '操作类型 add/delete/update/restore/batch_delete/batch_complete',
                    schedule_id INT COMMENT '关联schedule_remind_info.id',
                    schedule_content VARCHAR(500) COMMENT '日程内容',
                    detail TEXT COMMENT '操作详情',
                    source VARCHAR(20) DEFAULT 'chat' COMMENT '操作来源 chat/sidebar/batch/voice',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '操作时间'
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='操作日志表'
            """)
            # 尝试添加 deleted_at 字段（如果表已存在）
            try:
                cur.execute("ALTER TABLE schedule_remind_info ADD COLUMN deleted_at DATETIME DEFAULT NULL COMMENT '移入回收站时间' AFTER status")
            except Exception:
                pass  # 字段已存在
            conn.commit()
            print("数据库表初始化完成")
    finally:
        conn.close()
