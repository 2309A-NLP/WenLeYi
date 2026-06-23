# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
"""
Agent核心逻辑：调用LLM → 解析JSON → 分发到数据库操作
"""

import json
from datetime import datetime
import requests

from config import LLM_CONFIG
import database as db
from prompt import SYSTEM_PROMPT, build_user_prompt
from tools import normalize_member, format_records_list, format_summary, getCatIcon


class AccountingAgent:
    """记账本智能体"""

    def __init__(self):
        # 存放最近一次LLM返回的结构化数据（供调试界面查看）
        self.last_call_log = {}

    def chat(self, user_input):
        """处理用户输入，返回回复文本"""
        today = datetime.now()
        user_msg = build_user_prompt(user_input, today)

        try:
            # 调用LLM
            url = LLM_CONFIG["base_url"].rstrip("/") + "/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLM_CONFIG['api_key']}",
            }
            payload = {
                "model": LLM_CONFIG["model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.1,
                "max_tokens": 1024,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # LLM调用失败时返回友好提示，避免500错误
            import traceback
            print(f"[LLM ERROR] {e}")
            print(f"[LLM ERROR DETAIL] {traceback.format_exc()}")
            print(f"[LLM CONFIG] api_key={LLM_CONFIG['api_key'][:10]}..., base_url={LLM_CONFIG['base_url']}")
            self.last_call_log = {
                "user_input": user_input,
                "error": str(e),
                "timestamp": today.strftime("%Y-%m-%d %H:%M:%S"),
            }
            return f"抱歉，AI服务暂时不可用: {str(e)}"


        # 记录调用信息
        self.last_call_log = {
            "user_input": user_input,
            "llm_raw": raw,
            "timestamp": today.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 解析JSON
        result = self._parse_json(raw)
        if result is None:
            return "抱歉，我没有理解你的意思，请再说一遍~"

        self.last_call_log["parsed"] = result
        action = result.get("action", "")

        # 分发到数据库操作
        reply = self._dispatch(action, result, user_input=user_input)
        return reply

    def _parse_json(self, raw):
        """从LLM回复中提取JSON"""
        try:
            # 尝试直接解析
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # 尝试提取 ```json ... ``` 块
        if "```" in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start:end])
                except json.JSONDecodeError:
                    pass
        return None

    def _dispatch(self, action, result, user_input=""):
        """根据action分发到对应操作"""
        fields = result.get("fields", {})
        reply = result.get("reply", "")

        # 关键词保底：用户输入含删除关键词时强制走delete
        delete_keywords = ["删除", "删掉", "去掉", "取消", "删除了"]
        if any(kw in user_input for kw in delete_keywords):
            if action not in ("confirm_delete",):
                action = "delete"
                if not fields.get("description"):
                    for kw in delete_keywords:
                        if kw in user_input:
                            idx = user_input.index(kw)
                            rest = user_input[idx + len(kw):]
                            fields["description"] = rest.strip("的了")
                            break

        # 关键词保底：用户输入含修改关键词时强制走update
        update_keywords = ["改成", "修改", "更正", "不对", "应该是", "改为"]
        if any(kw in user_input for kw in update_keywords):
            if action not in ("confirm_update",):
                action = "update"

        # 智能保底：检测"今日/今天/当天"→强制只查当天日期
        today_keywords = ["今日", "今天", "当天", "今日汇总", "今天汇总", "今日明细", "今天明细"]
        is_today_query = any(kw in user_input for kw in today_keywords)
        is_month_query = any(kw in user_input for kw in ["本月", "这个月", "这月", "月度"])

        if action in ("query", "summary") and is_today_query and not is_month_query:
            from datetime import datetime
            today_str = datetime.now().strftime("%Y-%m-%d")
            action = "summary"
            fields["date"] = today_str
            fields.pop("month", None)

        # 智能保底：如果LLM返回ask_more但用户其实说了金额，强制走add
        if action == "ask_more":
            import re
            has_amount = bool(re.search(r'(\d+\.?\d*)\s*(元|块|¥|￥|块钱|百|千|万)', user_input))
            if has_amount and fields.get("amount"):
                action = "add"
                # 补全缺失字段
                from tools import normalize_member
                from datetime import datetime
                if not fields.get("member"):
                    for word in ["我", "女儿", "闺女", "孩子", "爸爸", "父亲", "老妈", "妈妈", "母亲"]:
                        if word in user_input:
                            fields["member"] = normalize_member(word)
                            break
                    if not fields.get("member"):
                        fields["member"] = "女儿"  # 默认女儿
                if not fields.get("date"):
                    fields["date"] = datetime.now().strftime("%Y-%m-%d")
                if not fields.get("type"):
                    if any(w in user_input for w in ["收入", "收到", "报销", "工资", "赚"]):
                        fields["type"] = "收入"
                    else:
                        fields["type"] = "支出"
                if not fields.get("category"):
                    fields["category"] = fields.get("description", "其他") or "其他"
                if not fields.get("description"):
                    fields["description"] = fields.get("category", "")

        if action == "ask_more":
            return reply or "请补充完整信息哦~"

        if action == "add":
            return self._do_add(fields, reply)

        if action == "query":
            return self._do_query(fields, reply)

        if action == "summary":
            return self._do_summary(fields, reply)

        if action == "delete":
            return self._do_delete(result, reply)

        if action == "confirm_delete":
            return self._do_confirm_delete(result)

        if action == "update":
            return self._do_update(result, reply)

        if action == "confirm_update":
            return self._do_confirm_update(result)

        return reply or "已处理~"

    # ---- 各操作实现 ----

    def _do_add(self, fields, default_reply):
        """执行记账"""
        member = normalize_member(fields.get("member"))
        date_str = fields.get("date")
        type_ = fields.get("type", "支出")
        category = fields.get("category", "其他")
        description = fields.get("description", "")
        amount = fields.get("amount")

        if not all([member, date_str, type_, amount]):
            return "信息不完整，请告诉我：谁、哪天、收入还是支出、多少钱？"

        record_id = db.add(member, date_str, type_, category, description, amount)
        self.last_call_log["db_result"] = f"新增记录 id={record_id}"

        return default_reply or f"已记录：{date_str}，{member}，{category}，{description}，{type_} {amount}元"

    def _do_query(self, fields, default_reply):
        """执行查询"""
        member = normalize_member(fields.get("member"))
        records = db.query(
            member=member,
            date_str=fields.get("date"),
            month=fields.get("month"),
            category=fields.get("category"),
            description=fields.get("description"),
            type_=fields.get("type"),
        )
        self.last_call_log["db_result"] = f"查询到 {len(records)} 条记录"

        if not records:
            return "没有找到相关记录哦~"
        if default_reply and len(records) <= 3:
            return default_reply
        return format_records_list(records)

    def _do_summary(self, fields, default_reply):
        """执行汇总（支持按日期或按月）"""
        member = normalize_member(fields.get("member"))
        date_str = fields.get("date")
        month = fields.get("month")

        # 如果指定了具体日期，只查当天
        if date_str:
            records = db.query(member=member, date_str=date_str)
            expense = sum(float(r['amount']) for r in records if r['type'] == '支出')
            income = sum(float(r['amount']) for r in records if r['type'] == '收入')
            total_count = len(records)
            if not records:
                return f"{date_str} 没有账单记录"
            lines = [f"📅 {date_str} 账单明细："]
            for i, r in enumerate(records, 1):
                from tools import getCatIcon
                icon = getCatIcon(r.get('category', ''))
                sign = '+' if r['type'] == '收入' else '-'
                lines.append(f"{i}. {icon} {r['member']} {r.get('description','')} {sign}{r['amount']}元")
            lines.append(f"合计：{len(records)}笔，支出{expense}元，收入{income}元")
            return "\n".join(lines)

        # 按月汇总
        summary = db.get_monthly_summary(member=member, month=month)
        self.last_call_log["db_result"] = f"汇总: {summary['count']}笔, 总额{summary['total']}元"
        return format_summary(summary)

    def _do_delete(self, result, default_reply):
        """删除：先搜索匹配记录"""
        record_ids = result.get("record_ids", [])
        fields = result.get("fields", {})
        # 如果LLM没返回record_ids，主动搜索
        if not record_ids:
            desc = fields.get("description", "")
            member = normalize_member(fields.get("member"))
            records = db.find_by_description(desc, member=member)
            if not records:
                return "没有找到匹配的记录，无法删除~"
            record_ids = [r["id"] for r in records]
            self.last_call_log["matched_records"] = [dict(r) for r in records]
            lines = ["找到以下匹配记录："]
            for i, r in enumerate(records, 1):
                lines.append(f"  {i}. [{r['id']}] {r['date']} {r['member']} {r.get('category','')} {r.get('description','')} {r['type']} {r['amount']}元")
            lines.append("请问要删除哪条？请告诉我编号。")
            return "\n".join(lines)
        # LLM已经找到了记录
        return default_reply

    def _do_confirm_delete(self, result):
        """确认删除"""
        record_ids = result.get("record_ids", [])
        deleted = []
        for rid in record_ids:
            db.delete(rid)
            deleted.append(rid)
        self.last_call_log["db_result"] = f"已删除记录: {deleted}"
        return f"已删除记录 {deleted}，操作完成~"

    def _do_update(self, result, default_reply):
        """修改：先显示原记录和新值"""
        record_id = result.get("update_id")
        fields = result.get("fields", {})
        if not record_id:
            return "请告诉我要修改哪条记录的编号~"
        old = db.get_by_id(record_id)
        if not old:
            return f"记录 {record_id} 不存在~"
        self.last_call_log["old_record"] = dict(old)
        self.last_call_log["new_fields"] = fields
        return default_reply

    def _do_confirm_update(self, result):
        """确认修改"""
        record_id = result.get("update_id")
        fields = result.get("fields", {})
        if not record_id:
            return "请告诉我要修改哪条记录~"
        # 标准化成员
        if "member" in fields:
            fields["member"] = normalize_member(fields["member"])
        db.update(record_id, **fields)
        self.last_call_log["db_result"] = f"已修改记录 {record_id}"
        return f"记录 {record_id} 已修改完成~"

    def get_call_log(self):
        """获取最近一次调用信息（供调试界面）"""
        return self.last_call_log
