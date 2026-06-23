# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
"""
Flask应用入口：路由定义 + 启动服务
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from agent import AccountingAgent
import database as db

app = Flask(__name__)
# 启用跨域支持（允许前端跨域请求）
CORS(app)
agent = AccountingAgent()

WELCOME = (
    "您好，欢迎使用咱们小家专属记账本！\n"
    "请按照\"x年x月x日，谁做什么事收入/支出多少钱\"的格式来输入。\n"
    "请告诉我你的账目需求吧~"
)


# ========== 原有路由（保持不变） ==========

@app.route("/")
def index():
    return render_template("index.html", welcome=WELCOME)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data:
        return jsonify({"reply": "请发送有效内容哦~"})
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "请输入内容哦~"})
    try:
        reply = agent.chat(user_input)
    except Exception as e:
        # 捕获所有异常，防止500错误
        reply = "抱歉，处理出错了，请稍后再试~"
    return jsonify({"reply": reply})


@app.route("/view_calls")
def view_calls():
    return jsonify(agent.get_call_log())


@app.route("/view_data")
def view_data():
    return jsonify(db.query())


@app.route("/api/today_summary")
def api_today_summary():
    date_str = request.args.get("date")
    return jsonify(db.today_summary(date_str))


@app.route("/api/date_records")
def api_date_records():
    """查询指定日期的所有记录"""
    date_str = request.args.get("date")
    if not date_str:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
    records = db.query(date_str=date_str)
    return jsonify(records)


@app.route("/api/chats", methods=["GET", "POST"])
def api_chats():
    if request.method == "POST":
        data = request.json
        db.save_chat(data.get("role", "user"), data.get("content", ""))
        return jsonify({"ok": True})
    return jsonify(db.get_chats())


@app.route("/api/chats/clear", methods=["POST"])
def api_clear_chats():
    db.clear_chats()
    return jsonify({"ok": True})


@app.route("/api/monthly_category")
def api_monthly_category():
    return jsonify(db.monthly_category_summary())


# ========== 新增路由 ==========

@app.route("/health")
def health_check():
    """健康检查接口"""
    return jsonify({"code": 0, "message": "success", "data": {"status": "ok"}})


@app.route("/api/v1/record/add", methods=["POST"])
def api_record_add():
    """
    新增账目记录
    请求体: {"member": "爸爸", "date": "2025-06-21", "type": "支出",
             "category": "餐饮", "description": "午餐", "amount": 35.5}
    """
    data = request.json
    if not data:
        return jsonify({"code": 1, "message": "请求体不能为空", "data": None}), 400

    member = data.get("member", "我")
    date_str = data.get("date")
    type_ = data.get("type")
    category = data.get("category", "")
    description = data.get("description", "")
    amount = data.get("amount")

    # 参数校验
    missing = []

    if not date_str:
        missing.append("date")
    if not type_:
        missing.append("type")
    if amount is None:
        missing.append("amount")

    if missing:
        return jsonify({
            "code": 1,
            "message": "缺少必填参数: " + ", ".join(missing),
            "data": None
        }), 400

    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return jsonify({"code": 1, "message": "金额必须为数字", "data": None}), 400

    try:
        record_id = db.add(member, date_str, type_, category, description, amount)
        return jsonify({
            "code": 0,
            "message": "success",
            "data": {"id": record_id, "member": member, "date": date_str,
                     "type": type_, "category": category,
                     "description": description, "amount": amount}
        })
    except Exception as e:
        return jsonify({"code": 2, "message": f"数据库写入失败: {str(e)}", "data": None}), 500


@app.route("/api/v1/record/query", methods=["GET"])
def api_record_query():
    """
    查询账目记录（支持多条件筛选）
    查询参数: member, date, month, category, description, type
    """
    member = request.args.get("member")
    date_str = request.args.get("date")
    month = request.args.get("month")
    category = request.args.get("category")
    description = request.args.get("description")
    type_ = request.args.get("type")

    try:
        records = db.query(member=member, date_str=date_str, month=month,
                           category=category, description=description, type_=type_)
        # 将 datetime 对象转为字符串，确保 JSON 可序列化
        for r in records:
            for k, v in r.items():
                if hasattr(v, 'isoformat'):
                    r[k] = v.isoformat()
        return jsonify({
            "code": 0,
            "message": "success",
            "data": {"records": records, "count": len(records)}
        })
    except Exception as e:
        return jsonify({"code": 2, "message": f"查询失败: {str(e)}", "data": None}), 500


@app.route("/api/v1/record/delete", methods=["POST"])
def api_record_delete():
    """
    删除账目记录
    请求体: {"id": 123}
    """
    data = request.json
    if not data:
        return jsonify({"code": 1, "message": "请求体不能为空", "data": None}), 400

    record_id = data.get("id")
    if record_id is None:
        return jsonify({"code": 1, "message": "缺少必填参数: id", "data": None}), 400

    try:
        rowcount = db.delete(int(record_id))
        if rowcount == 0:
            return jsonify({"code": 1, "message": "记录不存在", "data": None}), 404
        return jsonify({
            "code": 0,
            "message": "success",
            "data": {"deleted_id": int(record_id), "affected_rows": rowcount}
        })
    except Exception as e:
        return jsonify({"code": 2, "message": f"删除失败: {str(e)}", "data": None}), 500


@app.route("/api/v1/record/update", methods=["POST"])
def api_record_update():
    """
    更新账目记录
    请求体: {"id": 123, "amount": 99.0, "category": "交通"}
    至少需要 id + 一个可更新字段
    """
    data = request.json
    if not data:
        return jsonify({"code": 1, "message": "请求体不能为空", "data": None}), 400

    record_id = data.get("id")
    if record_id is None:
        return jsonify({"code": 1, "message": "缺少必填参数: id", "data": None}), 400

    # 提取可更新字段（排除 id）
    allowed_fields = {"member", "date", "type", "category", "description", "amount"}
    updates = {k: v for k, v in data.items() if k in allowed_fields and k != "id"}

    if not updates:
        return jsonify({
            "code": 1,
            "message": "至少需要一个可更新字段: " + ", ".join(allowed_fields),
            "data": None
        }), 400

    try:
        rowcount = db.update(int(record_id), **updates)
        if rowcount == 0:
            return jsonify({"code": 1, "message": "记录不存在或未发生变化", "data": None}), 404
        return jsonify({
            "code": 0,
            "message": "success",
            "data": {"updated_id": int(record_id), "affected_rows": rowcount,
                     "updates": updates}
        })
    except Exception as e:
        return jsonify({"code": 2, "message": f"更新失败: {str(e)}", "data": None}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  小家记账本 启动成功！")
    print("  访问地址: http://localhost:5001")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=True)
