# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-Agent数字人项目-Agent编排器任务
工具服务HTTP客户端：负责与5个子工具服务通信
统一处理请求、响应解析、错误处理
"""
import requests
import logging
from config import TOOL_SERVICES, HTTP_TIMEOUT

logger = logging.getLogger(__name__)


class ToolClient:
    """工具服务HTTP客户端"""

    def __init__(self):
        # 记录每个服务的可用状态
        self._service_status = {}
        for key, svc in TOOL_SERVICES.items():
            self._service_status[key] = True  # 默认可用

    def check_health(self, service_key):
        """
        检查某个工具服务是否在线
        返回: True/False
        """
        svc = TOOL_SERVICES.get(service_key)
        if not svc:
            return False
        url = svc["base_url"] + svc["health"]
        try:
            resp = requests.get(url, timeout=5)
            # 如果返回200，就认为服务在线（兼容返回HTML或JSON的服务）
            ok = resp.status_code == 200
            self._service_status[service_key] = ok
            return ok
        except Exception as e:
            logger.warning("健康检查失败 [%s]: %s", service_key, str(e))
            self._service_status[service_key] = False
            return False

    def check_all_health(self):
        """检查所有服务的健康状态"""
        results = {}
        for key in TOOL_SERVICES:
            results[key] = self.check_health(key)
        return results

    def call_service(self, service_key, endpoint_key, method="POST", params=None, json_data=None):
        """
        通用的服务调用方法
        参数:
            service_key: 服务标识 (记账/日程/文生图/基金/招股书)
            endpoint_key: 接口标识 (add/query/delete/update/ask等)
            method: HTTP方法 (GET/POST)
            params: URL查询参数 (GET请求用)
            json_data: JSON请求体 (POST请求用)
        返回:
            {"success": True/False, "code": 0/-1, "message": "...", "data": {...}}
        """
        svc = TOOL_SERVICES.get(service_key)
        if not svc:
            return {"success": False, "code": -1, "message": f"未知服务: {service_key}", "data": None}

        endpoint = svc["endpoints"].get(endpoint_key)
        if not endpoint:
            return {"success": False, "code": -1, "message": f"未知接口: {endpoint_key}", "data": None}

        url = svc["base_url"] + endpoint
        try:
            if method.upper() == "GET":
                resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            else:
                resp = requests.post(url, json=json_data, timeout=HTTP_TIMEOUT)

            result = resp.json()
            code = result.get("code", -1)
            return {
                "success": code == 0,
                "code": code,
                "message": result.get("message", ""),
                "data": result.get("data"),
            }
        except requests.exceptions.ConnectionError:
            msg = f"无法连接到 {svc['name']} ({url})，请确认服务已启动"
            logger.error(msg)
            return {"success": False, "code": -1, "message": msg, "data": None}
        except requests.exceptions.Timeout:
            msg = f"{svc['name']} 请求超时 ({HTTP_TIMEOUT}秒)"
            logger.error(msg)
            return {"success": False, "code": -1, "message": msg, "data": None}
        except Exception as e:
            msg = f"{svc['name']} 调用异常: {str(e)}"
            logger.error(msg)
            return {"success": False, "code": -1, "message": msg, "data": None}

    # ==================== 便捷方法：记账 ====================

    def accounting_add(self, member, date, type_, category, description, amount):
        """记账 - 新增记录"""
        return self.call_service("记账", "add", method="POST", json_data={
            "member": member,
            "date": date,
            "type": type_,
            "category": category,
            "description": description,
            "amount": amount,
        })

    def accounting_query(self, **kwargs):
        """记账 - 查询记录"""
        # 过滤掉None值
        params = {k: v for k, v in kwargs.items() if v is not None}
        return self.call_service("记账", "query", method="GET", params=params)

    def accounting_delete(self, record_id):
        """记账 - 删除记录"""
        return self.call_service("记账", "delete", method="POST", json_data={"id": record_id})

    def accounting_update(self, record_id, **fields):
        """记账 - 更新记录"""
        data = {"id": record_id}
        data.update(fields)
        return self.call_service("记账", "update", method="POST", json_data=data)

    # ==================== 便捷方法：日程 ====================

    def schedule_add(self, schedule_date, schedule_time, content, repeat_rule=None):
        """日程 - 新增"""
        return self.call_service("日程", "add", method="POST", json_data={
            "schedule_date": schedule_date,
            "schedule_time": schedule_time,
            "content": content,
            "repeat_rule": repeat_rule,
        })

    def schedule_query(self, date=None, keyword=None, schedule_id=None):
        """日程 - 查询"""
        params = {}
        if date:
            params["date"] = date
        if keyword:
            params["keyword"] = keyword
        if schedule_id:
            params["id"] = schedule_id
        return self.call_service("日程", "query", method="GET", params=params)

    def schedule_delete(self, schedule_id):
        """日程 - 删除"""
        return self.call_service("日程", "delete", method="POST", json_data={"id": schedule_id})

    def schedule_update(self, schedule_id, **fields):
        """日程 - 更新"""
        data = {"id": schedule_id}
        data.update(fields)
        return self.call_service("日程", "update", method="POST", json_data=data)

    # ==================== 便捷方法：文生图 ====================

    def image_generate(self, prompt, image_type="right", reference_image=""):
        """文生图 - 生成单张"""
        return self.call_service("文生图", "generate", method="POST", json_data={
            "prompt": prompt,
            "type": image_type,
            "reference_image": reference_image,
        })

    def image_generate_all(self, reference_image):
        """文生图 - 一键生成三个方向"""
        return self.call_service("文生图", "generate_all", method="POST", json_data={
            "reference_image": reference_image,
        })

    # ==================== 便捷方法：基金问答 ====================

    def fund_ask(self, question):
        """基金 - 问答（适配工单4的返回格式）"""
        import requests as req
        svc = TOOL_SERVICES.get("基金")
        if not svc:
            return {"success": False, "code": -1, "message": "基金服务未配置", "data": None}
        url = svc["base_url"] + svc["endpoints"]["ask"]
        try:
            resp = req.post(url, json={"question": question}, timeout=HTTP_TIMEOUT)
            result = resp.json()
            # 工单4返回格式: {"question": ..., "answer": ..., "sql": ..., "error": ...}
            # 转换为工单6统一格式
            if result.get("error"):
                return {"success": False, "code": -1, "message": result["error"], "data": None}
            return {
                "success": True,
                "code": 0,
                "message": "success",
                "data": {"answer": result.get("answer", "未找到相关数据")}
            }
        except Exception as e:
            return {"success": False, "code": -1, "message": f"基金问答失败: {str(e)}", "data": None}

    # ==================== 便捷方法：招股书问答 ====================

    def prospectus_query(self, question):
        """招股书 - 问答"""
        return self.call_service("招股书", "query", method="POST", json_data={
            "question": question,
        })


# 全局单例
tool_client = ToolClient()
