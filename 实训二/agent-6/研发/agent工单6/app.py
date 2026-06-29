# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-Agent数字人项目-Agent编排器任务
Flask主程序：路由定义 + 启动服务
编排器负责接收用户自然语言输入，智能分发到5个子工具服务
"""
import logging
import os
import requests as http_requests  # 重命名避免与flask request冲突
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from orchestrator import orchestrator
from tool_client import tool_client
import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 启用跨域支持


# ==================== 页面路由 ====================

@app.route("/")
def index():
    """主页 - 聊天界面"""
    return render_template("index.html")


# ==================== 核心API ====================

@app.route("/chat", methods=["POST"])
def chat():
    """
    对话接口 - 接收用户自然语言，编排器智能分发
    请求体: {"message": "用户输入", "session_id": "会话ID(可选)"}
    响应: {"reply": "回复文本", "tool_used": "使用的工具"}
    """
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"reply": "请输入您的需求", "tool_used": None})

    try:
        result = orchestrator.chat(user_message, session_id)
        return jsonify({
            "reply": result.get("reply", ""),
            "tool_used": result.get("tool_used"),
        })
    except Exception as e:
        logger.error("对话处理异常: %s", str(e))
        return jsonify({
            "reply": "抱歉，处理出错了，请稍后再试",
            "tool_used": None,
        }), 500


# ==================== 健康检查 ====================

@app.route("/health")
def health():
    """
    编排器健康检查
    同时检测所有子工具服务的在线状态
    """
    service_status = tool_client.check_all_health()
    all_ok = all(service_status.values())
    return jsonify({
        "code": 0,
        "message": "success",
        "data": {
            "status": "ok" if all_ok else "partial",
            "services": service_status,
        }
    })


@app.route("/health/simple")
def health_simple():
    """简单健康检查 - 只返回编排器自身状态"""
    return jsonify({
        "code": 0,
        "message": "success",
        "data": {"status": "ok"}
    })


# ==================== 工具服务管理API ====================

@app.route("/api/v1/services", methods=["GET"])
def list_services():
    """列出所有工具服务及其状态"""
    service_status = tool_client.check_all_health()
    services = []
    for key, svc in config.TOOL_SERVICES.items():
        services.append({
            "key": key,
            "name": svc["name"],
            "base_url": svc["base_url"],
            "endpoints": list(svc["endpoints"].keys()),
            "online": service_status.get(key, False),
        })
    return jsonify({
        "code": 0,
        "message": "success",
        "data": {"services": services, "total": len(services)}
    })


@app.route("/api/v1/services/health", methods=["GET"])
def check_services_health():
    """检查所有工具服务的健康状态"""
    service_status = tool_client.check_all_health()
    return jsonify({
        "code": 0,
        "message": "success",
        "data": service_status
    })


# ==================== 工具直通API ====================
# 以下接口绕过LLM意图识别，直接调用对应工具
# 适用于明确知道要调用哪个工具的场景

@app.route("/api/v1/accounting/add", methods=["POST"])
def direct_accounting_add():
    """直接调用记账-新增"""
    data = request.get_json(force=True)
    result = tool_client.accounting_add(
        member=data.get("member", ""),
        date=data.get("date", ""),
        type_=data.get("type", "支出"),
        category=data.get("category", "其他"),
        description=data.get("description", ""),
        amount=data.get("amount", 0),
    )
    return jsonify(result)


@app.route("/api/v1/accounting/query", methods=["GET"])
def direct_accounting_query():
    """直接调用记账-查询"""
    params = {k: v for k, v in request.args.items() if v}
    result = tool_client.accounting_query(**params)
    return jsonify(result)


@app.route("/api/v1/accounting/delete", methods=["POST"])
def direct_accounting_delete():
    """直接调用记账-删除"""
    data = request.get_json(force=True)
    result = tool_client.accounting_delete(data.get("id"))
    return jsonify(result)


@app.route("/api/v1/accounting/update", methods=["POST"])
def direct_accounting_update():
    """直接调用记账-更新"""
    data = request.get_json(force=True)
    rid = data.pop("id", None)
    result = tool_client.accounting_update(rid, **data)
    return jsonify(result)


@app.route("/api/v1/schedule/add", methods=["POST"])
def direct_schedule_add():
    """直接调用日程-新增"""
    data = request.get_json(force=True)
    result = tool_client.schedule_add(
        schedule_date=data.get("schedule_date", ""),
        schedule_time=data.get("schedule_time", ""),
        content=data.get("content", ""),
        repeat_rule=data.get("repeat_rule"),
    )
    return jsonify(result)


@app.route("/api/v1/schedule/query", methods=["GET"])
def direct_schedule_query():
    """直接调用日程-查询"""
    result = tool_client.schedule_query(
        date=request.args.get("date"),
        keyword=request.args.get("keyword"),
        schedule_id=request.args.get("id"),
    )
    return jsonify(result)


@app.route("/api/v1/schedule/delete", methods=["POST"])
def direct_schedule_delete():
    """直接调用日程-删除"""
    data = request.get_json(force=True)
    result = tool_client.schedule_delete(data.get("id"))
    return jsonify(result)


@app.route("/api/v1/schedule/update", methods=["POST"])
def direct_schedule_update():
    """直接调用日程-更新"""
    data = request.get_json(force=True)
    sid = data.pop("id", None)
    result = tool_client.schedule_update(sid, **data)
    return jsonify(result)


@app.route("/api/v1/image/generate", methods=["POST"])
def direct_image_generate():
    """直接调用文生图-生成"""
    data = request.get_json(force=True)
    result = tool_client.image_generate(
        prompt=data.get("prompt", ""),
        image_type=data.get("type", "right"),
        reference_image=data.get("reference_image", ""),
    )
    return jsonify(result)


@app.route("/api/v1/image/generate_all", methods=["POST"])
def direct_image_generate_all():
    """直接调用文生图-三张全生成"""
    data = request.get_json(force=True)
    result = tool_client.image_generate_all(
        reference_image=data.get("reference_image", ""),
    )
    return jsonify(result)


@app.route("/api/v1/fund/ask", methods=["POST"])
def direct_fund_ask():
    """直接调用基金-问答"""
    data = request.get_json(force=True)
    result = tool_client.fund_ask(data.get("question", ""))
    return jsonify(result)


@app.route("/api/v1/prospectus/query", methods=["POST"])
def direct_prospectus_query():
    """直接调用招股书-问答"""
    data = request.get_json(force=True)
    result = tool_client.prospectus_query(data.get("question", ""))
    return jsonify(result)


# ==================== 对话历史管理 ====================

@app.route("/api/v1/history/clear", methods=["POST"])
def clear_history():
    """清除对话历史"""
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", "default")
    orchestrator.clear_history(session_id)
    return jsonify({
        "code": 0,
        "message": "success",
        "data": {"session_id": session_id}
    })


# ==================== 工具API代理路由 ====================
# 前端通过代理访问各工具服务的API，避免CORS问题

# 代理映射配置: 中文名 -> 英文key、端口、API前缀
TOOL_PROXY_MAP = {
    "记账": {"key": "accounting",  "port": 5001, "api_prefix": "/api/v1/record"},
    "日程": {"key": "schedule",    "port": 5002, "api_prefix": "/api/v1/schedule"},
    "文生图": {"key": "image",      "port": 5003, "api_prefix": "/api/v1/image"},
    "基金": {"key": "fund",        "port": 5004, "api_prefix": "/api/v1/fund"},
    "招股书": {"key": "prospectus", "port": 5005, "api_prefix": "/api/v1/prospectus"},
}


def _proxy_request(tool_key, sub_path):
    """
    通用代理转发函数
    将前端请求转发到对应的工具服务，支持GET/POST/PUT/DELETE
    """
    proxy_cfg = TOOL_PROXY_MAP.get(tool_key)
    if not proxy_cfg:
        return jsonify({"code": -1, "message": f"未知工具: {tool_key}"}), 404

    # 拼接目标URL
    target_url = f"http://127.0.0.1:{proxy_cfg['port']}{proxy_cfg['api_prefix']}/{sub_path}"
    method = request.method.upper()

    # 构造转发请求
    headers = {k: v for k, v in request.headers if k.lower() not in ("host", "content-length")}
    try:
        if method == "GET":
            resp = http_requests.get(target_url, params=request.args, headers=headers, timeout=config.HTTP_TIMEOUT)
        elif method == "POST":
            # 支持JSON和表单数据
            if request.is_json:
                resp = http_requests.post(target_url, json=request.get_json(force=True), headers=headers, timeout=config.HTTP_TIMEOUT)
            else:
                resp = http_requests.post(target_url, data=request.get_data(), headers=headers, timeout=config.HTTP_TIMEOUT)
        elif method == "PUT":
            if request.is_json:
                resp = http_requests.put(target_url, json=request.get_json(force=True), headers=headers, timeout=config.HTTP_TIMEOUT)
            else:
                resp = http_requests.put(target_url, data=request.get_data(), headers=headers, timeout=config.HTTP_TIMEOUT)
        elif method == "DELETE":
            resp = http_requests.delete(target_url, headers=headers, timeout=config.HTTP_TIMEOUT)
        else:
            return jsonify({"code": -1, "message": f"不支持的HTTP方法: {method}"}), 405

        # 返回工具服务的响应（透传状态码和内容）
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
    except http_requests.exceptions.ConnectionError:
        logger.error("代理请求失败: 无法连接到 %s (port %s)", tool_key, proxy_cfg["port"])
        return jsonify({"code": -1, "message": f"工具服务 {tool_key} 无法连接，请确认服务已启动"}), 502
    except http_requests.exceptions.Timeout:
        logger.error("代理请求超时: %s", tool_key)
        return jsonify({"code": -1, "message": f"工具服务 {tool_key} 请求超时"}), 504
    except Exception as e:
        logger.error("代理请求异常 [%s]: %s", tool_key, str(e))
        return jsonify({"code": -1, "message": f"代理请求异常: {str(e)}"}), 500


# ---------- 记账代理 ----------
@app.route("/api/proxy/image/fetch", methods=["GET"])
def proxy_image_fetch():
    """代理获取远程图片，解决CORS问题。返回base64 data URL"""
    url = request.args.get("url", "")
    if not url:
        return jsonify({"code": -1, "message": "缺少url参数"}), 400
    try:
        resp = http_requests.get(url, timeout=60)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "image/png")
        import base64
        b64 = base64.b64encode(resp.content).decode("utf-8")
        data_url = f"data:{content_type};base64,{b64}"
        return jsonify({"code": 0, "data": {"data_url": data_url}})
    except Exception as e:
        return jsonify({"code": -1, "message": f"获取图片失败: {str(e)}"}), 500


@app.route("/api/proxy/accounting/<path:sub_path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_accounting(sub_path):
    """代理记账工具API请求"""
    return _proxy_request("记账", sub_path)


# ---------- 日程代理 ----------
@app.route("/api/proxy/schedule/<path:sub_path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_schedule(sub_path):
    """代理日程工具API请求"""
    return _proxy_request("日程", sub_path)


# ---------- 文生图代理 ----------
@app.route("/api/proxy/image/<path:sub_path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_image(sub_path):
    """代理文生图工具API请求"""
    return _proxy_request("文生图", sub_path)


# ---------- 基金代理 ----------
@app.route("/api/proxy/fund/<path:sub_path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_fund(sub_path):
    """代理基金工具API请求"""
    return _proxy_request("基金", sub_path)


# ---------- 招股书代理 ----------
@app.route("/api/proxy/prospectus/<path:sub_path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_prospectus(sub_path):
    """代理招股书工具API请求"""
    return _proxy_request("招股书", sub_path)


# ==================== 工具前端页面代理 ====================
# 通过iframe嵌入各工具的独立前端页面

def _proxy_tool_page(tool_name, tool_key, port):
    """
    代理工具前端页面
    将工具服务的HTML页面内容透传给前端iframe
    """
    target_url = f"http://127.0.0.1:{port}/"
    try:
        resp = http_requests.get(target_url, timeout=10)
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "text/html"),
        )
    except http_requests.exceptions.ConnectionError:
        return f"<html><body><h2>工具服务 {tool_name} 未启动</h2><p>请确认 http://127.0.0.1:{port} 已启动</p></body></html>", 502
    except Exception as e:
        return f"<html><body><h2>加载失败</h2><p>{str(e)}</p></body></html>", 500


@app.route("/tool/accounting")
def tool_page_accounting():
    """记账工具前端页面代理"""
    return _proxy_tool_page("记账", "accounting", 5001)


@app.route("/tool/schedule")
def tool_page_schedule():
    """日程工具前端页面代理"""
    return _proxy_tool_page("日程", "schedule", 5002)


@app.route("/tool/image")
def tool_page_image():
    """文生图工具前端页面代理"""
    return _proxy_tool_page("文生图", "image", 5003)


@app.route("/tool/fund")
def tool_page_fund():
    """基金工具前端页面代理"""
    return _proxy_tool_page("基金", "fund", 5004)


@app.route("/tool/prospectus")
def tool_page_prospectus():
    """招股书工具前端页面代理"""
    return _proxy_tool_page("招股书", "prospectus", 5005)


# ==================== 启动 ====================

# ==================== 豆包文生图API ====================
import base64 as b64mod

DOUBAO_API_KEY = os.environ.get("DOUBAO_API_KEY", "ark-30ee86b2-e709-43a3-af57-b2f44874aeb8-c3371")
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
DOUBAO_MODEL = "ep-20260629111310-268wr"


@app.route("/api/text2img", methods=["POST"])
def text2img():
    """
    纯文字生图 - 使用豆包Seedream模型
    请求体: {"prompt": "描述文本"}
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"code": -1, "message": "请输入图片描述", "data": None}), 400

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DOUBAO_API_KEY}",
        }
        payload = {
            "model": DOUBAO_MODEL,
            "prompt": prompt,
            "size": "1920x1920",
            "n": 1,
        }
        resp = http_requests.post(DOUBAO_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()

        if "data" in result and len(result["data"]) > 0:
            img = result["data"][0]
            # 支持URL和base64两种返回格式
            img_url = img.get("url", "")
            if not img_url and img.get("b64_json"):
                img_url = f"data:image/png;base64,{img['b64_json']}"
            if img_url:
                return jsonify({
                    "code": 0,
                    "message": "success",
                    "data": {"image_url": img_url, "prompt": prompt}
                })

        return jsonify({"code": -1, "message": "API返回中没有图片数据", "data": None}), 500

    except Exception as e:
        logger.error("豆包文生图失败: %s", str(e))
        return jsonify({"code": -1, "message": f"文生图失败: {str(e)}", "data": None}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  Agent编排器启动成功！")
    print(f"  访问地址: http://localhost:{config.FLASK_PORT}")
    print(f"  健康检查: http://localhost:{config.FLASK_PORT}/health")
    print(f"  对话接口: http://localhost:{config.FLASK_PORT}/chat")
    print("  工具服务:")
    for key, svc in config.TOOL_SERVICES.items():
        print(f"    - {svc['name']}: {svc['base_url']}")
    print("=" * 60)
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
