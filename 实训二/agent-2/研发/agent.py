# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-Agent 数字人项目-日程提醒智能体任务
LLM对话核心：DeepSeek API调用 + Prompt工程 + 意图解析
"""
import json
import time
import requests
from datetime import datetime, date, timedelta
import config
import database
from scheduler import schedule_reminder, cancel_reminder


def format_time(t):
    """将timedelta或字符串统一格式化为HH:MM"""
    if isinstance(t, timedelta):
        total = int(t.total_seconds())
        return "{:02d}:{:02d}".format(total // 3600, (total % 3600) // 60)
    return str(t)[:5]


# ==================== 系统Prompt ====================

SYSTEM_PROMPT = """你是日程提醒智能体，负责帮主人管理日程。你必须严格遵守以下规则：

【核心铁律】
在任何情况下，你都要调用数据库，绝不允许跳过数据库直接回复。你必须返回JSON格式的指令。

【输出格式】你只能返回以下JSON格式，不要返回任何自然语言：
- 添加日程：{"action":"add","date":"YYYY-MM-DD","time":"HH:MM","content":"事项内容","repeat":"daily/weekly/monthly/null"}
- 查询日程：{"action":"query","date":"YYYY-MM-DD"}
- 删除日程：{"action":"delete","id":数字}
- 修改日程：{"action":"update","id":数字,"field":"字段名","value":"新值"}
- 引导追问：{"action":"ask","message":"追问的话"}

【日期时间规则】
- "今天" = 当前日期 {today}
- "明天" = 明天日期 {tomorrow}
- "后天" = 后天日期 {day_after_tomorrow}
- "下周一" = 下个周一的日期
- "下午5点" = 17:00
- "上午9点" = 09:00
- "早上" = 08:00（如用户没指定具体时间则追问）
- "晚上" = 20:00（如用户没指定具体时间则追问）
- 如果用户没说具体时间，必须追问

【中文时间理解】你必须能理解以下中文时间表达：
- "十点半" = 10:30
- "十点" = 10:00
- "下午十点半" = 22:30
- "晚上十点" = 22:00
- "凌晨三点" = 03:00
- "中午十二点" = 12:00
- "傍晚六点" = 18:00
- "上午十点半" = 10:30
- "下午两点半" = 14:30
- "两点半" = 14:30（默认下午）
- "三刻钟后" = 当前时间+45分钟
- "半小时后" = 当前时间+30分钟
- "一小时后" = 当前时间+60分钟
- 中文数字对照：一=1,二=2,三=3,四=4,五=5,六=6,七=7,八=8,九=9,十=10,十一=11,十二=12
- "半" = 30分钟
- "一刻" = 15分钟
- "三刻" = 45分钟

【多轮对话】
- 如果用户回复了你之前的追问（比如你问"请问安排在什么时间？"，用户回答"明天下午3点"），你需要结合上下文理解完整意图
- 用户说"确认"/"是"/"好的"等肯定词，表示同意你的建议或确认操作
- 用户说"取消"/"不用了"等否定词，表示取消操作

【完整性引导】
- 如果用户只说了内容没说时间：返回ask，追问"请问这件事安排在什么时间呢？"
- 如果用户只说了时间没说内容：返回ask，追问"请问需要提醒您做什么事情呢？"
- 不允许直接记录残缺信息

【口语化理解】
- "买咖啡" = 购买咖啡
- "开个会" = 开会
- "还信用卡" = 信用卡还款
- "锻炼" = 运动健身
- "吃药" = 服药
- "放学" = 课程结束

【循环规则】
- "每天" → repeat=daily
- "每周" → repeat=weekly
- "每月" → repeat=monthly
- 不提循环 → repeat=null

【删除确认】
收到删除指令后，返回：
{"action":"confirm_delete","id":数字,"content":"该日程的内容","schedule_time":"时间"}
等用户确认后再执行删除。

【修改确认】
收到修改指令后，返回：
{"action":"confirm_update","id":数字,"field":"字段","old_value":"旧值","new_value":"新值"}
等用户确认后再执行修改。

【查询结果格式】
查询到日程后，返回：
{"action":"query_result","schedules":[{"id":数字,"time":"HH:MM","content":"内容"},...]}


你只返回JSON，不要返回其他任何文字。"""


def build_system_prompt():
    """构建带日期变量的系统Prompt"""
    from datetime import timedelta
    today = date.today()
    tomorrow = today + timedelta(days=1)
    day_after = today + timedelta(days=2)
    # 用replace避免与JSON花括号冲突
    prompt = SYSTEM_PROMPT
    prompt = prompt.replace("{today}", today.strftime("%Y-%m-%d"))
    prompt = prompt.replace("{tomorrow}", tomorrow.strftime("%Y-%m-%d"))
    prompt = prompt.replace("{day_after_tomorrow}", day_after.strftime("%Y-%m-%d"))
    return prompt


# ==================== DeepSeek API ====================

def call_deepseek(messages):
    """调用DeepSeek API"""
    headers = {
        "Authorization": "Bearer " + config.DEEPSEEK_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    resp = requests.post(
        config.DEEPSEEK_BASE_URL + "/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ==================== 意图解析与执行 ====================

def parse_llm_output(raw_text):
    """从LLM输出中提取JSON"""
    text = raw_text.strip()
    # 去除markdown代码块标记
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # 找到第一个 { 和最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def execute_action(parsed):
    """执行LLM返回的JSON指令，返回给前端的回复"""
    if not parsed:
        return "抱歉，我没有理解您的意思，请再说一次。"

    action = parsed.get("action", "")

    # ---- 引导追问 ----
    if action == "ask":
        return parsed.get("message", "请补充完整信息。")

    # ---- 添加日程 ----
    if action == "add":
        schedule_date = parsed.get("date", "")
        schedule_time = parsed.get("time", "")
        content = parsed.get("content", "")
        repeat = parsed.get("repeat", "null")

        if not schedule_date or not schedule_time or not content:
            return "信息不完整，请补充日期、时间和事项内容。"

        # 冲突检测：同一天、同一时间、同一内容
        conflicts = database.check_time_conflict(schedule_date, schedule_time, content)
        if conflicts:
            conflict_text = "\n".join(["  {} {}".format(c["time"], c["content"]) for c in conflicts])
            return "⚠️ 该日程已存在！\n{}\n\n如需重新添加，请修改时间或内容。".format(conflict_text)

        # repeat=null时传None
        if repeat in ("null", "none", "None", ""):
            repeat = None

        new_id = database.add_schedule(schedule_date, schedule_time, content, repeat)
        # 注册精确提醒
        schedule_reminder(new_id, schedule_date, schedule_time, repeat)
        # 记录操作日志
        database.add_operation_log("add", new_id, content, source="chat")
        repeat_text = ""
        if repeat:
            repeat_map = {"daily": "每天", "weekly": "每周", "monthly": "每月"}
            repeat_text = "（循环规则：{}）".format(repeat_map.get(repeat, repeat))

        return "已添加日程：\n{} {}\n{}\n日程编号：{}{}".format(
            schedule_date, schedule_time, content, new_id, repeat_text
        )

    # ---- 查询日程 ----
    if action == "query":
        query_date = parsed.get("date", date.today().strftime("%Y-%m-%d"))
        schedules = database.get_today_schedules(query_date)

        if not schedules:
            return "{} 没有任何日程安排。".format(query_date)

        lines = ["{} 的日程列表：".format(query_date)]
        for s in schedules:
            t = format_time(s["schedule_time"])
            lines.append("{}|{:07d}|{}".format(t, s["id"], s["content"]))
        return "\n".join(lines)

    # ---- 删除日程（确认） ----
    if action == "confirm_delete":
        sid = parsed.get("id")
        # 从数据库获取实际数据，不依赖LLM返回的占位符
        record = database.get_schedule_by_id(sid)
        if record:
            t = format_time(record["schedule_time"])
            return "确认要删除日程{}吗？内容是：{} {}".format(sid, t, record["content"])
        return "未找到日程{}。".format(sid)

    # ---- 删除日程（执行） ----
    if action == "delete":
        sid = parsed.get("id")
        cancel_reminder(sid)
        record = database.delete_schedule(sid)
        if record:
            t = format_time(record["schedule_time"])
            database.add_operation_log("delete", sid, record["content"], source="chat")
            return "已删除日程{}，删除的日程内容是：{} {}".format(sid, t, record["content"])
        return "未找到日程{}或已被删除。".format(sid)

    # ---- 修改日程（确认） ----
    if action == "confirm_update":
        sid = parsed.get("id")
        field = parsed.get("field", "")
        new_val = parsed.get("new_value", "")
        # 从数据库获取实际旧值
        record = database.get_schedule_by_id(sid)
        if record:
            old_val = str(record.get(field, ""))
            return "确认要将日程{}的{}从\"{}\"修改为\"{}\"吗？".format(sid, field, old_val, new_val)
        return "未找到日程{}。".format(sid)

    # ---- 修改日程（执行） ----
    if action == "update":
        sid = parsed.get("id")
        field = parsed.get("field", "")
        value = parsed.get("value", "")
        record = database.update_schedule(sid, field, value)
        if record:
            database.add_operation_log("update", sid, record.get("content"), detail="修改{}={}".format(field, value), source="chat")
            return "已修改日程{}的{}为\"{}\"。".format(sid, field, value)
        return "未找到日程{}或已被删除。".format(sid)

    return "未知操作：{}".format(action)


# ==================== 主对话处理 ====================

# 会话TTL：30分钟（秒）
SESSION_TTL = 30 * 60

# 临时存储确认状态: {session_id: {"action": "confirm_delete", "parsed": {...}, "ts": 时间戳}}
_confirm_buffer = {}

# 多轮对话历史: {session_id: [{"role": "user/assistant", "content": "..."}]}
_chat_history = {}
MAX_HISTORY = 10  # 保留最近10轮对话


def _cleanup_expired_sessions():
    """清理所有空闲超过30分钟的会话，防止内存无限增长"""
    now = time.time()
    # 清理对话历史
    expired_sessions = [sid for sid, data in _chat_history.items()
                        if not data or now - data[-1].get("_ts", 0) > SESSION_TTL]
    for sid in expired_sessions:
        del _chat_history[sid]
    # 清理确认缓冲
    expired_confirms = [sid for sid, data in _confirm_buffer.items()
                        if now - data.get("_ts", 0) > SESSION_TTL]
    for sid in expired_confirms:
        del _confirm_buffer[sid]


def handle_user_input(user_text, session_id="default"):
    """处理用户输入的主入口"""
    user_text = user_text.strip()
    if not user_text:
        return "请输入您的日程需求。"
    # 每次请求时清理过期会话
    _cleanup_expired_sessions()

    # 检查是否有待确认的操作
    if session_id in _confirm_buffer:
        pending = _confirm_buffer.pop(session_id)
        confirm_text = user_text.lower()
        if confirm_text in ("确认", "是", "好的", "yes", "y", "ok", "确定"):
            return execute_action(pending["parsed"])
        else:
            return "已取消操作。"

    # 构建消息并调用LLM（带多轮对话历史）
    system_prompt = build_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    # 添加历史对话（过滤掉内部时间戳字段）
    if session_id in _chat_history:
        for msg in _chat_history[session_id]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_text})

    try:
        raw_reply = call_deepseek(messages)
    except Exception as e:
        import traceback
        print(f"[LLM ERROR] {e}")
        print(f"[LLM ERROR DETAIL] {traceback.format_exc()}")
        return "调用AI服务失败：{}" .format(str(e))

    # 保存对话历史
    if session_id not in _chat_history:
        _chat_history[session_id] = []
    _chat_history[session_id].append({"role": "user", "content": user_text, "_ts": time.time()})
    # 解析LLM输出
    parsed = parse_llm_output(raw_reply)

    # 执行动作，获取人类可读回复（用于前端显示和对话历史）
    result = execute_action(parsed)

    # 只存储人类可读文本，过滤掉JSON指令，避免LLM被自己的JSON输出混淆
    _chat_history[session_id].append({"role": "assistant", "content": result, "_ts": time.time()})

    # 限制历史长度
    if len(_chat_history[session_id]) > MAX_HISTORY * 2:
        _chat_history[session_id] = _chat_history[session_id][-MAX_HISTORY * 2:]
    # 如果是确认类操作，缓存并返回确认提示
    if parsed and parsed.get("action") in ("confirm_delete", "confirm_update", "confirm_add"):
        # 缓存实际要执行的动作，而不是确认动作
        execute_parsed = dict(parsed)
        if parsed["action"] == "confirm_delete":
            execute_parsed["action"] = "delete"
        elif parsed["action"] == "confirm_update":
            execute_parsed["action"] = "update"
            execute_parsed["value"] = execute_parsed.get("new_value", "")
        elif parsed["action"] == "confirm_add":
            execute_parsed["action"] = "add"
        _confirm_buffer[session_id] = {"parsed": execute_parsed, "_ts": time.time()}
        return execute_action(parsed)

    return result
