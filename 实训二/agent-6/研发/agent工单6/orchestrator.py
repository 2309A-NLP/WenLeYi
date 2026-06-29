# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-Agent数字人项目-Agent编排器任务
编排器核心逻辑：意图识别 → 参数提取 → 工具调用 → 结果整合
使用LLM做意图分类和参数提取，然后分发到对应的工具服务
"""
import json
import logging
import requests
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from tool_client import tool_client

logger = logging.getLogger(__name__)


# ==================== 系统Prompt ====================

ORCHESTRATOR_PROMPT = """你是智能编排器，负责分析用户输入，决定调用哪个工具，并提取所需参数。

【可用工具】
1. 记账工具 - 记录/查询/修改/删除家庭账目
   - 新增: {"tool":"记账","action":"add","params":{"member":"成员","date":"YYYY-MM-DD","type":"收入/支出","category":"分类","description":"描述","amount":数字}}
   - 查询: {"tool":"记账","action":"query","params":{"member":"成员(可选)","date":"YYYY-MM-DD(可选)","month":"YYYY-MM(可选)","category":"分类(可选)","type":"收入/支出(可选)"}}
   - 删除: {"tool":"记账","action":"delete","params":{"id":记录ID}}
   - 修改: {"tool":"记账","action":"update","params":{"id":记录ID,"字段名":"新值"}}

2. 日程工具 - 管理日程提醒
   - 新增: {"tool":"日程","action":"add","params":{"schedule_date":"YYYY-MM-DD","schedule_time":"HH:MM","content":"事项内容","repeat_rule":"daily/weekly/monthly/null(可选)"}}
   - 查询: {"tool":"日程","action":"query","params":{"date":"YYYY-MM-DD(可选)","keyword":"关键词(可选)"}}
   - 删除: {"tool":"日程","action":"delete","params":{"id":日程ID}}
   - 修改: {"tool":"日程","action":"update","params":{"id":日程ID,"字段名":"新值"}}

3. 文生图工具 - 根据参考图片生成不同角度的图片
   - 生成单张: {"tool":"文生图","action":"generate","params":{"prompt":"提示词","type":"left/right/front","reference_image":"base64图片数据"}}
   - 三张全生成: {"tool":"文生图","action":"generate_all","params":{"reference_image":"base64图片数据"}}

4. 基金工具 - 基金数据问答
   - 问答: {"tool":"基金","action":"ask","params":{"question":"问题内容"}}

5. 招股书工具 - 招股书数据问答
   - 问答: {"tool":"招股书","action":"query","params":{"question":"问题内容"}}

6. 多工具 - 同时调用多个工具
   - {"tool":"多工具","actions":[{"tool":"记账","action":"add","params":{...}},{"tool":"日程","action":"add","params":{...}}]}

【日期时间规则】
- "今天" = {today}
- "明天" = {tomorrow}
- "后天" = {day_after_tomorrow}
- "本月" = {current_month}
- "下午5点" = 17:00
- "上午9点" = 09:00
- "十点半" = 10:30

【判断规则】
- 涉及记账、花钱、收入、支出、账单、消费、金额 → 记账工具
- 涉及日程、提醒、会议、约会、安排、日历 → 日程工具
- 涉及画图、生成图片、AI画、文生图 → 文生图工具
- 涉及基金、股票、投资、净值、收益率 → 基金工具
- 涉及招股书、IPO、上市、招股 → 招股书工具
- 同时涉及多个领域 → 多工具
- 以上都不是 → 返回 {"tool":"闲聊","reply":"自然语言回复"}

【输出格式】
只返回JSON，不要返回其他文字。
闲聊时返回: {"tool":"闲聊","reply":"回复内容"}
调用工具时返回: {"tool":"工具名","action":"动作","params":{...}}
多工具时返回: {"tool":"多工具","actions":[...]}
"""


class Orchestrator:
    """Agent编排器"""

    def __init__(self):
        # 对话历史（按session_id隔离）
        self._chat_history = {}

    def _build_system_prompt(self):
        """构建带日期变量的系统Prompt"""
        from datetime import datetime, timedelta
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        day_after = today + timedelta(days=2)
        current_month = today.strftime("%Y-%m")

        prompt = ORCHESTRATOR_PROMPT
        prompt = prompt.replace("{today}", today.strftime("%Y-%m-%d"))
        prompt = prompt.replace("{tomorrow}", tomorrow.strftime("%Y-%m-%d"))
        prompt = prompt.replace("{day_after_tomorrow}", day_after.strftime("%Y-%m-%d"))
        prompt = prompt.replace("{current_month}", current_month)
        return prompt

    def _call_llm(self, messages):
        """调用LLM进行意图识别 - 使用requests直接调用"""
        try:
            logger.info("LLM调用: model=%s, base_url=%s, api_key=%s...", 
                       LLM_MODEL, LLM_BASE_URL, LLM_API_KEY[:10] if LLM_API_KEY else "EMPTY")
            url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLM_API_KEY}",
            }
            payload = {
                "model": LLM_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1024,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
            logger.info("LLM返回: %s", raw[:200])
            return raw
        except Exception as e:
            import traceback
            logger.error("LLM调用失败: %s", str(e))
            logger.error("LLM调用失败详情: %s", traceback.format_exc())
            logger.error("LLM配置: api_key=%s..., base_url=%s, model=%s", 
                        LLM_API_KEY[:10] if LLM_API_KEY else "EMPTY", LLM_BASE_URL, LLM_MODEL)
            return None

    def _parse_json(self, raw_text):
        """从LLM输出中提取JSON"""
        if not raw_text:
            return None
        text = raw_text.strip()
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试提取 ```json ... ``` 块
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        # 尝试找第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return None

    def _execute_tool(self, parsed):
        """根据解析结果执行工具调用"""
        tool_name = parsed.get("tool", "")

        # 闲聊
        if tool_name == "闲聊":
            return {
                "success": True,
                "tool": "闲聊",
                "reply": parsed.get("reply", "您好，请问有什么可以帮助您的？"),
            }

        # 多工具调用
        if tool_name == "多工具":
            return self._execute_multi_tools(parsed.get("actions", []))

        # 单工具调用
        action = parsed.get("action", "")
        params = parsed.get("params", {})
        return self._dispatch_single(tool_name, action, params)

    def _dispatch_single(self, tool_name, action, params):
        """分发到单个工具执行"""
        try:
            if tool_name == "记账":
                return self._exec_accounting(action, params)
            elif tool_name == "日程":
                return self._exec_schedule(action, params)
            elif tool_name == "文生图":
                return self._exec_image(action, params)
            elif tool_name == "基金":
                return self._exec_fund(action, params)
            elif tool_name == "招股书":
                return self._exec_prospectus(action, params)
            else:
                return {"success": False, "reply": f"不支持的工具: {tool_name}"}
        except Exception as e:
            logger.error("工具执行异常 [%s/%s]: %s", tool_name, action, str(e))
            return {"success": False, "reply": f"执行出错: {str(e)}"}

    def _exec_accounting(self, action, params):
        """执行记账工具"""
        if action == "add":
            result = tool_client.accounting_add(
                member=params.get("member", ""),
                date=params.get("date", ""),
                type_=params.get("type", "支出"),
                category=params.get("category", "其他"),
                description=params.get("description", ""),
                amount=params.get("amount", 0),
            )
        elif action == "query":
            result = tool_client.accounting_query(**params)
        elif action == "delete":
            result = tool_client.accounting_delete(params.get("id"))
        elif action == "update":
            rid = params.pop("id", None)
            result = tool_client.accounting_update(rid, **params)
        else:
            return {"success": False, "reply": f"记账工具不支持操作: {action}"}

        return self._format_result("记账", result)

    def _exec_schedule(self, action, params):
        """执行日程工具"""
        if action == "add":
            result = tool_client.schedule_add(
                schedule_date=params.get("schedule_date", ""),
                schedule_time=params.get("schedule_time", ""),
                content=params.get("content", ""),
                repeat_rule=params.get("repeat_rule"),
            )
        elif action == "query":
            result = tool_client.schedule_query(**params)
        elif action == "delete":
            result = tool_client.schedule_delete(params.get("id"))
        elif action == "update":
            sid = params.pop("id", None)
            result = tool_client.schedule_update(sid, **params)
        else:
            return {"success": False, "reply": f"日程工具不支持操作: {action}"}

        return self._format_result("日程", result)

    def _exec_image(self, action, params):
        """执行文生图工具"""
        if action == "generate":
            reference_image = params.get("reference_image", "")
            if not reference_image:
                return {"success": False, "reply": "请先上传一张参考图片，我才能帮您生成效果图哦~"}
            result = tool_client.image_generate(
                prompt=params.get("prompt", ""),
                image_type=params.get("type", "right"),
                reference_image=reference_image,
            )
        elif action == "generate_all":
            reference_image = params.get("reference_image", "")
            if not reference_image:
                return {"success": False, "reply": "请先上传一张参考图片，我才能帮您生成效果图哦~"}
            result = tool_client.image_generate_all(
                reference_image=reference_image,
            )
        else:
            return {"success": False, "reply": f"文生图工具不支持操作: {action}"}

        return self._format_result("文生图", result)

    def _exec_fund(self, action, params):
        """执行基金工具"""
        if action == "ask":
            result = tool_client.fund_ask(params.get("question", ""))
        else:
            return {"success": False, "reply": f"基金工具不支持操作: {action}"}

        return self._format_result("基金", result)

    def _exec_prospectus(self, action, params):
        """执行招股书工具"""
        if action == "query":
            result = tool_client.prospectus_query(params.get("question", ""))
        else:
            return {"success": False, "reply": f"招股书工具不支持操作: {action}"}

        return self._format_result("招股书", result)

    def _execute_multi_tools(self, actions):
        """执行多个工具调用"""
        results = []
        for act in actions:
            tool_name = act.get("tool", "")
            action = act.get("action", "")
            params = act.get("params", {})
            r = self._dispatch_single(tool_name, action, params)
            results.append({"tool": tool_name, "action": action, "result": r})

        # 整合结果
        all_success = all(r["result"].get("success", False) for r in results)
        replies = []
        for r in results:
            reply_text = r["result"].get("reply", "")
            if reply_text:
                replies.append(f"【{r['tool']}】{reply_text}")

        return {
            "success": all_success,
            "tool": "多工具",
            "reply": "\n".join(replies) if replies else "多工具操作已完成",
            "details": results,
        }

    def _format_result(self, tool_name, result):
        """将工具返回结果格式化为用户友好的回复"""
        if result.get("success"):
            data = result.get("data")
            message = result.get("message", "success")

            # 根据不同工具和数据格式化回复
            if tool_name == "记账":
                return self._format_accounting_reply(data, message)
            elif tool_name == "日程":
                return self._format_schedule_reply(data, message)
            elif tool_name == "文生图":
                return self._format_image_reply(data, message)
            elif tool_name == "基金":
                return self._format_fund_reply(data, message)
            elif tool_name == "招股书":
                return self._format_prospectus_reply(data, message)
            else:
                return {"success": True, "reply": message, "data": data}
        else:
            return {
                "success": False,
                "reply": f"操作失败: {result.get('message', '未知错误')}",
            }

    def _format_accounting_reply(self, data, message):
        """格式化记账回复"""
        if not data:
            return {"success": True, "reply": message}
        # 查询结果
        if "records" in data:
            records = data["records"]
            count = data.get("count", len(records))
            if count == 0:
                return {"success": True, "reply": "没有找到相关账目记录"}
            lines = [f"查询到 {count} 条账目记录："]
            for i, r in enumerate(records, 1):
                sign = "+" if r.get("type") == "收入" else "-"
                lines.append(f"{i}. {r.get('date','')} {r.get('member','')} {r.get('category','')} {r.get('description','')} {sign}{r.get('amount',0)}元")
            return {"success": True, "reply": "\n".join(lines), "data": data}
        # 新增结果：显示详细信息
        if "id" in data:
            type_ = data.get("type", "")
            member = data.get("member", "")
            date = data.get("date", "")
            category = data.get("category", "")
            desc = data.get("description", "")
            amount = data.get("amount", 0)
            if type_ and member:
                # 新增/更新：显示完整记录
                reply = f"{type_} 已记录：{date}，{member}，{desc or category}，{type_}{amount}元"
            else:
                reply = f"操作成功，记录ID: {data['id']}"
            return {"success": True, "reply": reply, "data": data}
        return {"success": True, "reply": message, "data": data}

    def _format_schedule_reply(self, data, message):
        """格式化日程回复"""
        if not data:
            return {"success": True, "reply": message}
        # 查询结果
        if "items" in data:
            items = data["items"]
            total = data.get("total", len(items))
            if total == 0:
                date_str = data.get("date", "")
                return {"success": True, "reply": f"{date_str} 没有日程安排"}
            date_str = data.get("date", "")
            lines = [f"{date_str} 共 {total} 个日程："]
            for i, s in enumerate(items, 1):
                lines.append(f"{i}. [{s.get('id','')}] {s.get('time','') or s.get('schedule_time','')} {s.get('content','')}")
            return {"success": True, "reply": "\n".join(lines), "data": data}
        # 新增/删除/更新
        if isinstance(data, dict) and "id" in data:
            # 新增日程时，显示详细信息
            sid = data.get("id", "")
            date_str = data.get("schedule_date", "")
            time_str = data.get("schedule_time", "")
            content = data.get("content", "")
            repeat = data.get("repeat_rule", "")
            reply = f"已添加日程：{date_str} {time_str} {content}"
            if repeat:
                reply += f"（重复：{repeat}）"
            reply += "\n到时间会自动提醒您。"
            return {"success": True, "reply": reply, "data": data}
        return {"success": True, "reply": message, "data": data}

    def _format_image_reply(self, data, message):
        """格式化文生图回复"""
        if not data:
            error_msg = data.get("error", "") if data else ""
            if error_msg:
                return {"success": False, "reply": f"生成失败：{error_msg}"}
            return {"success": True, "reply": message}
        url = data.get("image_url", "")
        if url:
            return {"success": True, "reply": f"图片生成成功！\n{url}", "data": data}
        error_msg = data.get("error", "")
        if error_msg:
            return {"success": False, "reply": f"生成失败：{error_msg}"}
        return {"success": True, "reply": message, "data": data}

    def _format_fund_reply(self, data, message):
        """格式化基金回复"""
        if not data:
            return {"success": True, "reply": message}
        answer = data.get("answer", "")
        if answer:
            return {"success": True, "reply": answer, "data": data}
        return {"success": True, "reply": message, "data": data}

    def _format_prospectus_reply(self, data, message):
        """格式化招股书回复"""
        if not data:
            return {"success": True, "reply": message}
        answer = data.get("answer", "")
        if answer:
            return {"success": True, "reply": answer, "data": data}
        return {"success": True, "reply": message, "data": data}

    def chat(self, user_input, session_id="default"):
        """
        主对话入口：接收用户输入，完成意图识别→工具调用→结果返回
        参数:
            user_input: 用户自然语言输入
            session_id: 会话ID（用于隔离多轮对话历史）
        返回:
            {"reply": "回复文本", "tool_used": "使用的工具", "raw": "LLM原始输出"}
        """
        user_input = user_input.strip()
        if not user_input:
            return {"reply": "请输入您的需求", "tool_used": None, "raw": ""}

        # 构建消息
        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]

        # 添加对话历史
        if session_id in self._chat_history:
            for msg in self._chat_history[session_id]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_input})

        # 调用LLM
        raw = self._call_llm(messages)
        if raw is None:
            return {"reply": "AI服务暂时不可用，请稍后再试", "tool_used": None, "raw": ""}

        # 解析JSON
        parsed = self._parse_json(raw)
        if parsed is None:
            # LLM返回了非JSON格式，当作闲聊处理
            return {"reply": raw, "tool_used": None, "raw": raw}

        # 保存对话历史
        if session_id not in self._chat_history:
            self._chat_history[session_id] = []
        self._chat_history[session_id].append({"role": "user", "content": user_input})
        # 只保存LLM的原始输出作为assistant回复（避免上下文混淆）
        self._chat_history[session_id].append({"role": "assistant", "content": raw})

        # 限制历史长度（最近10轮对话）
        max_history = 20
        if len(self._chat_history[session_id]) > max_history:
            self._chat_history[session_id] = self._chat_history[session_id][-max_history:]

        # 执行工具调用
        result = self._execute_tool(parsed)

        return {
            "reply": result.get("reply", "已处理"),
            "tool_used": result.get("tool", parsed.get("tool", "")),
            "raw": raw,
            "data": result.get("data"),
        }

    def clear_history(self, session_id="default"):
        """清除指定会话的对话历史"""
        if session_id in self._chat_history:
            del self._chat_history[session_id]


# 全局单例
orchestrator = Orchestrator()
