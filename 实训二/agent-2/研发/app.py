# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-Agent 数字人项目-日程提醒智能体任务
Flask主程序：路由 + 前端轮询 + 启动调度器
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import date
import database
import agent
from scheduler import start_scheduler, get_new_reminders, schedule_reminder, cancel_reminder
import config

app = Flask(__name__)
# 启用 CORS 跨域支持，允许所有来源访问
CORS(app)


@app.route("/")
def index():
    """渲染聊天页面"""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """接收用户消息，调agent处理"""
    data = request.get_json(force=True)
    user_text = data.get("message", "")
    session_id = data.get("session_id", "default")

    if not user_text.strip():
        return jsonify({"reply": "请输入您的日程需求。"})

    reply = agent.handle_user_input(user_text, session_id)
    return jsonify({"reply": reply})


@app.route("/reminders")
def reminders():
    """前端轮询：获取新提醒"""
    items = get_new_reminders()
    return jsonify({"reminders": items})


@app.route("/schedules")
def schedules():
    """查询当天日程列表"""
    today = date.today().strftime("%Y-%m-%d")
    try:
        rows = database.get_today_schedules(today)
        result = []
        for r in rows:
            t = r["schedule_time"]
            from datetime import timedelta
            if isinstance(t, timedelta):
                total = int(t.total_seconds())
                t = "{:02d}:{:02d}".format(total // 3600, (total % 3600) // 60)
            else:
                t = str(t)[:5]
            result.append({
                "id": r["id"],
                "time": t,
                "content": r["content"],
                "repeat_rule": r.get("repeat_rule"),
            })
        return jsonify({"date": today, "schedules": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/init_db", methods=["POST"])
def init_db():
    """初始化数据库表（仅允许POST，防止爬虫/预加载误触发）"""
    try:
        database.init_tables()
        return jsonify({"status": "ok", "message": "数据库表初始化完成"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==================== 新增API ====================

@app.route("/api/schedules_category")
def schedules_category():
    """按分类查询日程"""
    category = request.args.get("category", "today")
    today = date.today().strftime("%Y-%m-%d")
    try:
        rows = database.get_schedules_by_category(category, today)
        return jsonify({"category": category, "schedules": _safe_json(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
def api_search():
    """全局搜索日程"""
    keyword = request.args.get("q", "").strip()
    if not keyword:
        return jsonify({"schedules": []})
    try:
        rows = database.search_schedules(keyword)
        return jsonify({"keyword": keyword, "schedules": _safe_json(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/counts")
def api_counts():
    """获取各分类数量"""
    today = date.today().strftime("%Y-%m-%d")
    try:
        counts = database.get_category_counts(today)
        return jsonify(counts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear_all", methods=["POST"])
def api_clear_all():
    """清空所有日程数据"""
    try:
        database.clear_all_schedules()
        return jsonify({"status": "ok", "message": "已清空所有日程"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete_schedule", methods=["POST"])
def api_delete_schedule():
    """直接删除指定日程（侧边栏用）→ 移入回收站"""
    data = request.get_json(force=True)
    schedule_id = data.get("id")
    if not schedule_id:
        return jsonify({"error": "缺少id"}), 400
    try:
        from scheduler import cancel_reminder
        cancel_reminder(schedule_id)
        record = database.soft_delete_to_recycle(schedule_id)
        if record:
            database.add_operation_log("delete", schedule_id, record.get("content"), source="sidebar")
            return jsonify({"status": "ok", "message": "已移入回收站"})
        return jsonify({"error": "未找到日程"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/schedules_by_date")
def api_schedules_by_date():
    """查询指定日期的日程"""
    date_str = request.args.get("date", date.today().strftime("%Y-%m-%d"))
    try:
        rows = database.get_schedules_by_date(date_str)
        return jsonify({"date": date_str, "schedules": _safe_json(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/month_counts")
def api_month_counts():
    """获取某月每天的日程数量"""
    try:
        year = int(request.args.get("year", date.today().year))
    except (ValueError, TypeError):
        year = date.today().year
    try:
        month = int(request.args.get("month", date.today().month))
    except (ValueError, TypeError):
        month = date.today().month
    try:
        counts = database.get_month_schedule_counts(year, month)
        return jsonify({"year": year, "month": month, "counts": counts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/update_schedule", methods=["POST"])
def api_update_schedule():
    """完整编辑日程"""
    data = request.get_json(force=True)
    schedule_id = data.get("id")
    schedule_date = data.get("schedule_date")
    schedule_time = data.get("schedule_time")
    content = data.get("content")
    repeat_rule = data.get("repeat_rule")
    if not schedule_id:
        return jsonify({"error": "缺少id"}), 400
    try:
        if repeat_rule in ("", "null", "none"):
            repeat_rule = None
        record = database.update_full_schedule(schedule_id, schedule_date, schedule_time, content, repeat_rule)
        if record:
            from scheduler import cancel_reminder, schedule_reminder
            cancel_reminder(schedule_id)
            schedule_reminder(schedule_id, schedule_date, schedule_time, repeat_rule)
            database.add_operation_log("update", schedule_id, content, source="sidebar")
            return jsonify({"status": "ok", "message": "日程已更新"})
        return jsonify({"error": "未找到日程"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== V3 API ====================

@app.route("/api/recycle_bin")
def api_recycle_bin():
    """获取回收站"""
    try:
        rows = database.get_recycle_bin()
        return jsonify({"schedules": _safe_json(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recycle_restore", methods=["POST"])
def api_recycle_restore():
    """从回收站恢复"""
    data = request.get_json(force=True)
    schedule_id = data.get("id")
    try:
        record = database.restore_from_recycle(schedule_id)
        if record:
            from scheduler import schedule_reminder
            schedule_reminder(schedule_id, record["schedule_date"], record["schedule_time"], record.get("repeat_rule"))
            database.add_operation_log("restore", schedule_id, record.get("content"), source="recycle")
            return jsonify({"status": "ok", "message": "已恢复"})
        return jsonify({"error": "未找到"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recycle_clear", methods=["POST"])
def api_recycle_clear():
    """清空回收站"""
    try:
        count = database.clear_recycle_bin()
        database.add_operation_log("batch_delete", detail="清空回收站{}条".format(count), source="recycle")
        return jsonify({"status": "ok", "count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch_delete", methods=["POST"])
def api_batch_delete():
    """批量删除（移入回收站）"""
    data = request.get_json(force=True)
    ids = data.get("ids", [])
    if not ids:
        return jsonify({"error": "未选择日程"}), 400
    try:
        from scheduler import cancel_reminder
        for sid in ids:
            cancel_reminder(sid)
        records = database.batch_delete_schedules(ids)
        for r in records:
            database.add_operation_log("batch_delete", r["id"], r.get("content"), source="batch")
        return jsonify({"status": "ok", "count": len(records)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch_complete", methods=["POST"])
def api_batch_complete():
    """批量标记完成"""
    data = request.get_json(force=True)
    ids = data.get("ids", [])
    if not ids:
        return jsonify({"error": "未选择日程"}), 400
    try:
        count = database.batch_complete_schedules(ids)
        database.add_operation_log("batch_complete", detail="批量完成{}条".format(count), source="batch")
        return jsonify({"status": "ok", "count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/check_conflict")
def api_check_conflict():
    """检查时间冲突"""
    d = request.args.get("date", "")
    t = request.args.get("time", "")
    if not d or not t:
        return jsonify({"conflicts": []})
    try:
        conflicts = database.check_time_conflict(d, t)
        return jsonify({"conflicts": conflicts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/operation_logs")
def api_operation_logs():
    """查询操作日志"""
    try:
        limit = int(request.args.get("limit", 50))
    except (ValueError, TypeError):
        limit = 50
    action_type = request.args.get("type", None)
    try:
        rows = database.get_operation_logs(limit, action_type)
        return jsonify({"logs": _safe_json(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 健康检查 ====================

@app.route("/health")
def health():
    """健康检查接口，返回服务状态"""
    return jsonify({"code": 0, "message": "success", "data": {"status": "running"}})


# ==================== V1 统一API（CRUD + 统一响应格式） ====================

def _safe_json(obj):
    """递归转换不可序列化的对象为字符串"""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_json(i) for i in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)

@app.route("/api/v1/schedule/add", methods=["POST"])
def api_v1_schedule_add():
    """新增日程（V1统一接口）"""
    data = request.get_json(force=True)
    schedule_date = data.get("schedule_date", "")
    schedule_time = data.get("schedule_time", "")
    content = data.get("content", "")
    repeat_rule = data.get("repeat_rule")

    if not schedule_date or not schedule_time or not content:
        return jsonify({"code": -1, "message": "缺少必要参数（schedule_date/schedule_time/content）", "data": None})

    try:
        new_id = database.add_schedule(schedule_date, schedule_time, content, repeat_rule or None)
        # 创建提醒调度
        schedule_reminder(new_id, schedule_date, schedule_time, repeat_rule)
        database.add_operation_log("add", new_id, content, source="api")
        record = database.get_schedule_by_id(new_id)
        return jsonify({"code": 0, "message": "success", "data": _safe_json(record)})
    except Exception as e:
        return jsonify({"code": -1, "message": str(e), "data": None}), 500


@app.route("/api/v1/schedule/query", methods=["GET"])
def api_v1_schedule_query():
    """查询日程（V1统一接口），支持按id或按日期查询"""
    schedule_id = request.args.get("id")
    schedule_date = request.args.get("date")
    keyword = request.args.get("keyword")

    try:
        if schedule_id:
            # 按ID查询单条
            record = database.get_schedule_by_id(schedule_id)
            if not record:
                return jsonify({"code": -1, "message": "未找到日程", "data": None})
            # 格式化时间
            if record.get("schedule_time"):
                record["time_str"] = database.format_time_str(record["schedule_time"])
                record["date_str"] = str(record.get("schedule_date", ""))
            return jsonify({"code": 0, "message": "success", "data": record})
        elif keyword:
            # 关键词搜索
            rows = database.search_schedules(keyword)
            return jsonify({"code": 0, "message": "success", "data": {"total": len(rows), "items": _safe_json(rows)}})
        else:
            # 按日期查询（默认今天）
            date_str = schedule_date or date.today().strftime("%Y-%m-%d")
            rows = database.get_schedules_by_date(date_str)
            return jsonify({"code": 0, "message": "success", "data": {"date": date_str, "total": len(rows), "items": _safe_json(rows)}})
    except Exception as e:
        return jsonify({"code": -1, "message": str(e), "data": None}), 500


@app.route("/api/v1/schedule/update", methods=["POST"])
def api_v1_schedule_update():
    """更新日程（V1统一接口）"""
    data = request.get_json(force=True)
    schedule_id = data.get("id")
    schedule_date = data.get("schedule_date")
    schedule_time = data.get("schedule_time")
    content = data.get("content")
    repeat_rule = data.get("repeat_rule")

    if not schedule_id:
        return jsonify({"code": -1, "message": "缺少id参数", "data": None})

    try:
        if repeat_rule in ("", "null", "none"):
            repeat_rule = None
        record = database.update_full_schedule(schedule_id, schedule_date, schedule_time, content, repeat_rule)
        if record:
            # 更新提醒调度
            cancel_reminder(schedule_id)
            if schedule_date and schedule_time:
                schedule_reminder(schedule_id, schedule_date, schedule_time, repeat_rule)
            database.add_operation_log("update", schedule_id, content, source="api")
            updated = database.get_schedule_by_id(schedule_id)
            return jsonify({"code": 0, "message": "success", "data": updated})
        return jsonify({"code": -1, "message": "未找到日程", "data": None})
    except Exception as e:
        return jsonify({"code": -1, "message": str(e), "data": None}), 500


@app.route("/api/v1/schedule/delete", methods=["POST"])
def api_v1_schedule_delete():
    """删除日程（V1统一接口），移入回收站"""
    data = request.get_json(force=True)
    schedule_id = data.get("id")

    if not schedule_id:
        return jsonify({"code": -1, "message": "缺少id参数", "data": None})

    try:
        cancel_reminder(schedule_id)
        record = database.soft_delete_to_recycle(schedule_id)
        if record:
            database.add_operation_log("delete", schedule_id, record.get("content"), source="api")
            return jsonify({"code": 0, "message": "success", "data": {"deleted_id": schedule_id}})
        return jsonify({"code": -1, "message": "未找到日程", "data": None})
    except Exception as e:
        return jsonify({"code": -1, "message": str(e), "data": None}), 500


if __name__ == "__main__":
    # 初始化数据库表
    database.init_tables()
    # 启动定时调度器
    start_scheduler()
    print("=" * 50)
    print("日程提醒智能体已启动")
    print("访问: http://localhost:{}".format(config.FLASK_PORT))
    print("=" * 50)
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
