"""
AI聊天系统 - 展示逻辑模块

职责：
- 提供登录、注册、角色选择与聊天 UI
- 代理转发后端接口（同源访问，避免浏览器跨域问题）
- 支持流式聊天输出与健康检查状态展示

依赖：
- web_frontend_style.py: 提供页面样式定义（CSS_STYLES）
"""

# 导入标准库模块
import os               # 操作系统接口，用于读取环境变量和路径操作
import socket           # 套接字库，可能用于网络检测（虽未直接使用，但保留）
from typing import Any, Dict, List, Optional   # 类型注解支持

# 导入第三方库
import requests         # HTTP 请求库，用于向后端 API 发送请求
from dotenv import load_dotenv          # 从 .env 文件加载环境变量
from fastapi import FastAPI, HTTPException          # FastAPI 框架核心类和异常类
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse  # 各种响应类型
from pydantic import BaseModel          # Pydantic 数据模型基类

# 导入样式定义模块（包含前端页面的 CSS 样式字符串）
from web_frontend_style import CSS_STYLES

# 加载环境变量（从项目根目录的 .env 文件中读取配置）
load_dotenv()

# 定义后端服务的候选地址列表（按优先级排序）
# 当未通过环境变量指定 BACKEND_BASE_URL 时，会依次尝试这些地址
DEFAULT_BACKEND_CANDIDATES = [
    "http://127.0.0.1:8001",   # 常用备用端口
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003",
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
    "http://127.0.0.1:8000",   # 默认主后端端口
    "http://localhost:8000",
]

# 获取后端基础 URL，优先从环境变量读取，否则使用候选列表的第一个地址
# 去除末尾的斜杠，避免拼接 URL 时出现双斜杠
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", DEFAULT_BACKEND_CANDIDATES[0]).rstrip("/")

# 创建 FastAPI 应用实例，用于提供前端页面和代理接口
app = FastAPI(title="Web Frontend (no Streamlit)")

# ==================== Pydantic 请求模型定义 ====================

class LoginRequest(BaseModel):
    """登录请求的数据模型"""
    username: str   # 用户名
    password: str   # 密码

class RegisterRequest(BaseModel):
    """注册请求的数据模型"""
    username: str   # 用户名
    password: str   # 密码

class FileData(BaseModel):
    """文件数据模型，用于前端上传 Base64 编码的文件"""
    name: str       # 文件名
    type: str       # MIME 类型
    data: str       # Base64 编码的文件内容

class ChatRequest(BaseModel):
    """聊天请求的数据模型（与后端 ChatRequest 保持一致）"""
    user_id: int                       # 用户ID
    role_id: int                       # 角色ID
    message: str                       # 用户消息文本
    file: Optional[FileData] = None    # 可选的上传文件

# ==================== 辅助函数 ====================

def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """兼容 Pydantic v1 和 v2 的模型转字典方法"""
    # 检查模型是否具有 model_dump 方法（Pydantic v2）
    if hasattr(model, "model_dump"):
        return model.model_dump()
    # 否则使用 v1 的 dict 方法
    return model.dict()

# 全局异常处理器：捕获所有未被处理的异常，返回统一的 JSON 错误响应
@app.exception_handler(Exception)
async def _unhandled_exception_handler(_request, exc: Exception):
    # 返回 HTTP 500 内部服务器错误，错误详情包含异常类型名称和异常信息
    return JSONResponse(
        status_code=500,
        content={"detail": f"frontend error: {type(exc).__name__}: {str(exc)}"},
    )


def _proxy_get(path: str, timeout: int = 15) -> requests.Response:
    """
    向后端发送 GET 请求的代理函数
    :param path: 请求路径（相对于 BACKEND_BASE_URL）
    :param timeout: 超时时间（秒），默认 15 秒
    :return: requests.Response 对象
    """
    # 拼接完整的后端 URL 并发送 GET 请求
    return requests.get(f"{BACKEND_BASE_URL}{path}", timeout=timeout)


def _proxy_post(path: str, payload: Dict[str, Any], timeout: int = 30) -> requests.Response:
    """
    向后端发送 POST 请求的代理函数
    :param path: 请求路径（相对于 BACKEND_BASE_URL）
    :param payload: 要发送的 JSON 数据字典
    :param timeout: 超时时间（秒），默认 30 秒
    :return: requests.Response 对象
    """
    # 拼接完整的后端 URL 并发送 JSON 格式的 POST 请求
    return requests.post(f"{BACKEND_BASE_URL}{path}", json=payload, timeout=timeout)


def _health_check_backend(base_url: str, timeout: int = 3) -> bool:
    """
    检查指定的后端服务是否健康可用
    :param base_url: 后端服务的基础 URL（例如 http://127.0.0.1:8001）
    :param timeout: 连接超时时间（秒），默认 3 秒
    :return: 如果后端健康且服务标识匹配则返回 True，否则 False
    """
    try:
        # 向后端的 /health 端点发送 GET 请求
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        # 状态码不是 200 则认为不健康
        if resp.status_code != 200:
            return False
        # 尝试解析 JSON 响应，如果无内容则返回空字典
        data = resp.json() if resp.content else {}
        # 检查响应中的 service 字段是否为 "financial-rag-system"（与后端服务标识一致）
        return data.get("service") == "financial-rag-system"
    except requests.RequestException:
        # 网络或请求异常，则认为不健康
        return False
    except ValueError:
        # JSON 解析失败，则认为不健康
        return False


def _resolve_backend_base_url() -> str:
    """
    智能解析后端基础 URL：
    1. 优先使用环境变量 BACKEND_BASE_URL 中指定的地址（并验证健康）
    2. 如果环境变量未配置或验证失败，则依次尝试 DEFAULT_BACKEND_CANDIDATES 中的地址
    3. 若都不可用，则返回环境变量中的地址或默认第一个候选地址（不做健康保证）
    :return: 确定的后端基础 URL（末尾无斜杠）
    """
    # 获取环境变量中指定的后端地址
    env_url = os.getenv("BACKEND_BASE_URL")
    if env_url:
        env_url = env_url.rstrip("/")  # 去除末尾斜杠
        # 验证该地址是否健康可用
        if _health_check_backend(env_url):
            return env_url  # 健康则直接使用
    # 遍历默认候选地址列表，找到第一个健康的地址
    for candidate in DEFAULT_BACKEND_CANDIDATES:
        if _health_check_backend(candidate):
            return candidate
    # 如果所有候选地址都不可用，返回环境变量中的地址（或默认第一个候选地址），
    # 注意：此时后端可能实际上不可用，但前端会尝试连接并可能在后续请求中失败
    return (env_url or DEFAULT_BACKEND_CANDIDATES[0]).rstrip("/")


# 重新解析并设置后端基础 URL（覆盖之前从环境变量直接读取的值，因为之前的值可能未经健康检查）
BACKEND_BASE_URL = _resolve_backend_base_url()


def _is_port_available(host: str, port: int) -> bool:
    """
    检查指定主机上的端口是否可用（未被占用）
    :param host: 主机地址（如 "127.0.0.1" 或 "0.0.0.0"）
    :param port: 端口号
    :return: 端口可用（可以绑定）则返回 True，否则 False
    """
    # 创建 IPv4 TCP 套接字
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # 设置套接字选项 SO_REUSEADDR，允许地址重用，避免 TIME_WAIT 状态导致误判
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # 尝试绑定到指定地址和端口，如果成功说明端口未被占用
            sock.bind((host, port))
            return True
        except OSError:
            # 绑定失败（通常是因为端口已被占用），返回 False
            return False


def _resolve_frontend_port(default_port: int = 8503, host: str = "127.0.0.1") -> int:
    env_port = os.getenv("FRONTEND_PORT")
    preferred_port = default_port
    if env_port:
        try:
            preferred_port = int(env_port)
        except ValueError:
            preferred_port = default_port

    if _is_port_available(host, preferred_port):
        return preferred_port

    for offset in range(1, 50):
        candidate = preferred_port + offset
        if _is_port_available(host, candidate):
            return candidate

    raise RuntimeError(f"No available frontend port from {preferred_port}")


# HTML 页面模板 - 包含动态逻辑部分
HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI聊天系统</title>
  <style>
    {css_styles}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="brand">
        <div class="logo">AI</div>
        <div>
          <div class="title">AI聊天系统</div>
        </div>
      </div>
      <div class="kpis">
        <span class="chip" id="chipBackend">后端：检测中…</span>
        <span class="chip success" id="chipFrontend">前端：在线</span>
      </div>
    </div>

    <div class="row">
      <div class="left" id="left">
        <div class="card" id="authCard">
          <div style="display:flex; gap:10px;">
            <button class="btn primary" id="tabLogin">登录</button>
            <button class="btn" id="tabRegister">注册</button>
          </div>
          <div class="hr"></div>

          <div id="loginPane">
            <div style="margin-bottom:12px;">
              <label>账号</label>
              <input id="loginUsername" placeholder="输入账号" />
            </div>
            <div style="margin-bottom:12px;">
              <label>密码</label>
              <input id="loginPassword" type="password" placeholder="输入密码" />
            </div>
            <div style="margin-bottom:14px;">
              <label>图形验证码</label>
              <div class="captcha-row">
                <input id="loginCaptcha" placeholder="输入验证码" style="flex:1;" />
                <img id="captchaImg" class="captcha-img" src="" alt="验证码" onclick="refreshCaptcha()" />
              </div>
            </div>
            <button class="btn primary full" id="btnLogin">登录</button>
            <div class="muted" style="margin-top:10px;">提示：登录后会按用户隔离历史记录。</div>
          </div>

          <div id="registerPane" class="hidden">
            <div style="margin-bottom:12px;">
              <label>账号</label>
              <input id="regUsername" placeholder="新账号" />
            </div>
            <div style="margin-bottom:12px;">
              <label>密码</label>
              <input id="regPassword" type="password" placeholder="设置密码" />
            </div>
            <div style="margin-bottom:14px;">
              <label>图形验证码</label>
              <div class="captcha-row">
                <input id="regCaptcha" placeholder="输入验证码" style="flex:1;" />
                <img id="regCaptchaImg" class="captcha-img" src="" alt="验证码" onclick="refreshRegCaptcha()" />
              </div>
            </div>
            <button class="btn good full" id="btnRegister">注册</button>
            <div class="muted" style="margin-top:10px;">注册成功后请切回“登录”。</div>
          </div>
        </div>

        <div class="hidden" id="userCard">
          <div class="card-header">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
              <div>
                <div style="font-size:12px; color:var(--text-muted); margin-bottom:2px;">当前用户</div>
                <div style="font-weight:500; color:var(--text); font-size:14px;" id="userLine">—</div>
              </div>
              <button class="btn danger small" id="btnLogout">退出</button>
            </div>
          </div>
          <div class="card-section">
            <div class="form-label">选择角色</div>
            <div id="roleList" class="roleList"></div>
            <div class="text-muted" id="roleDesc" style="margin-top:6px;">请选择对话角色</div>
          </div>
          <div class="card-section compact">
            <div class="form-label">对话历史</div>
            <div class="grid2">
              <button class="btn outline" id="btnLoadHistory">
                <span id="loadHistoryText">加载历史</span>
                <span id="loadHistorySpinner" class="loading-spinner hidden" style="margin-left:6px;"></span>
              </button>
              <button class="btn danger" id="btnClearHistory">清空历史</button>
            </div>
          </div>
          <div class="card-section compact">
            <div class="form-label">股票快捷查询</div>
            <input id="stockQuick" placeholder="输入股票代码/简称" style="margin-bottom:8px;" />
            <div class="text-muted" style="margin-bottom:10px;">示例：600519 / 茅台</div>
            <button class="btn good full" id="btnStockQuick">查询并发送</button>
            <div class="text-muted" style="margin-top:8px;">从东方财富数据集检索行情</div>
          </div>
        </div>
      </div>

      <div class="main" id="main">
        <div id="welcomeScreen" style="display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; min-height:60vh; padding:40px;">
          <div style="width:120px; height:120px; border-radius:50%; background:linear-gradient(135deg, #a855f7, #ec4899); display:grid; place-items:center; margin-bottom:24px; box-shadow:0 20px 40px rgba(168,85,247,.3);">
            <span style="font-size:48px; font-weight:800; color:white;">AI</span>
          </div>
          <h1 style="font-size:32px; font-weight:780; color:#1f2937; margin:0 0 12px 0;">欢迎使用</h1>
          <p style="color:#6b7280; font-size:16px; margin:0;">AI聊天系统</p>
          <p style="color:#9ca3af; font-size:13px; margin-top:32px;">请先在左侧登录或注册账号</p>
        </div>

        <div id="chatScreen" class="hidden" style="display:flex; flex-direction:column; height:100%;">
          <div class="chatHeader">
            <div>
              <div class="chatTitle" id="chatTitle">请选择角色</div>
              <div class="chatDesc" id="chatDesc">选择角色开始对话。</div>
            </div>
            <div class="muted">消息会自动追加并可手动加载/清空历史</div>
          </div>

          <div class="chatWrap" id="chatWrap"></div>

          <div class="composer">
            <div class="input-wrapper">
              <div class="file-preview" id="filePreview">
                <img id="previewImg" class="preview-image" src="" alt="预览">
                <div class="preview-info">
                  <span class="preview-name" id="previewName"></span>
                  <button class="btn-remove-file" id="btnRemoveFile">移除</button>
                </div>
              </div>
              <textarea id="msgInput" placeholder="输入消息…" disabled></textarea>
              <div class="composer-hint" id="composerHint">Shift+Enter 换行</div>
              <div class="composer-actions">
                <button class="composer-btn" id="btnEmoji" title="表情">😊</button>
                <button class="composer-btn" id="btnAttach" title="附件">📎</button>
                <button class="composer-btn" id="btnQuick" title="快捷提问">✨</button>
              </div>
            </div>
            <div style="display: flex; gap: 8px;">
              <button class="btn primary" id="btnSend" disabled>发送</button>
              <button class="btn danger" id="btnEndChat" disabled>停止对话</button>
            </div>
          </div>

          <div class="quick-prompts" id="quickPrompts" style="display:none;">
            <div class="quick-prompts-header">
              <span style="font-size:13px; font-weight:500; color:var(--text-primary);">快捷提问</span>
              <button class="btn small outline" id="btnCloseQuick" style="margin-left:auto;">关闭</button>
            </div>
            <div class="quick-prompts-list" id="quickPromptsList">
              <div class="quick-prompt">你有什么特点？</div>
              <div class="quick-prompt">推荐一些好的书籍</div>
              <div class="quick-prompt">帮我分析一下当前市场趋势</div>
              <div class="quick-prompt">如何提高工作效率？</div>
              <div class="quick-prompt">给我讲个笑话</div>
              <div class="quick-prompt">今天天气怎么样？</div>
            </div>
          </div>

          <div class="emoji-panel" id="emojiPanel" style="display:none;">
            <div class="emoji-header">
              <span style="font-size:13px; font-weight:500; color:var(--text-primary);">选择表情</span>
              <button class="btn small outline" id="btnCloseEmoji" style="margin-left:auto;">关闭</button>
            </div>
            <div class="emoji-grid" id="emojiGrid">
            </div>
          </div>

          <input type="file" id="fileInput" style="display:none;" accept="image/*,audio/*,video/*,.txt,.pdf,.doc,.docx">
          
          <button class="composer-btn" id="btnRecord" title="语音录制" style="display:none;">🎤</button>
          
          <div class="recording-indicator" id="recordingIndicator" style="display:none;">
            <div class="recording-dot"></div>
            <span class="recording-text">录音中...</span>
          </div>
      </div>
    </div>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);

    const state = {
      user: null,
      roles: [],
      role: null,
      messages: [],
      busy: false,
      paused: false,
      stopped: false,
      pendingBuffer: '',
      currentReader: null,
    };

    function generateCaptcha(length = 4) {
      const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789';
      let result = '';
      for (let i = 0; i < length; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
      }
      return result;
    }

    function createCaptchaImage(text) {
      const canvas = document.createElement('canvas');
      canvas.width = 100;
      canvas.height = 40;
      const ctx = canvas.getContext('2d');
      
      ctx.fillStyle = '#fdf2f8';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      for (let i = 0; i < 6; i++) {
        ctx.beginPath();
        ctx.strokeStyle = `rgba(168,85,247,${Math.random() * 0.3})`;
        ctx.lineWidth = 1;
        ctx.moveTo(Math.random() * canvas.width, Math.random() * canvas.height);
        ctx.lineTo(Math.random() * canvas.width, Math.random() * canvas.height);
        ctx.stroke();
      }
      
      for (let i = 0; i < 15; i++) {
        ctx.beginPath();
        ctx.fillStyle = `rgba(168,85,247,${Math.random() * 0.2})`;
        ctx.arc(Math.random() * canvas.width, Math.random() * canvas.height, Math.random() * 2, 0, Math.PI * 2);
        ctx.fill();
      }
      
      ctx.font = 'bold 22px Arial';
      ctx.textBaseline = 'middle';
      
      const colors = ['#7c3aed', '#9333ea', '#a855f7', '#ec4899'];
      for (let i = 0; i < text.length; i++) {
        const char = text.charAt(i);
        const x = 15 + i * 20;
        const y = canvas.height / 2;
        const angle = (Math.random() - 0.5) * 0.4;
        const color = colors[i % colors.length];
        
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        ctx.fillStyle = color;
        ctx.fillText(char, 0, 0);
        ctx.restore();
      }
      
      return canvas.toDataURL();
    }

    window.refreshCaptcha = function() {
      const captcha = generateCaptcha();
      $('captchaImg').src = createCaptchaImage(captcha);
      $('captchaImg').dataset.captcha = captcha;
    }

    window.refreshRegCaptcha = function() {
      const captcha = generateCaptcha();
      $('regCaptchaImg').src = createCaptchaImage(captcha);
      $('regCaptchaImg').dataset.captcha = captcha;
    }

    function setBackendChip(ok, text){
      const el = $('chipBackend');
      el.textContent = text;
      el.classList.toggle('success', ok);
    }

    function toast(msg){
      const div = document.createElement('div');
      div.textContent = msg;
      div.style.position = 'fixed';
      div.style.left = '50%';
      div.style.bottom = '18px';
      div.style.transform = 'translateX(-50%)';
      div.style.padding = '10px 12px';
      div.style.borderRadius = '12px';
      div.style.border = '1px solid rgba(255,255,255,.14)';
      div.style.background = 'rgba(0,0,0,.45)';
      div.style.color = 'rgba(255,255,255,.92)';
      div.style.boxShadow = '0 18px 40px rgba(0,0,0,.35)';
      div.style.zIndex = 9999;
      document.body.appendChild(div);
      setTimeout(() => { div.remove(); }, 2600);
    }

    async function api(path, opts={}){
      const res = await fetch(path, {
        headers: { 'Content-Type': 'application/json', ...(opts.headers||{}) },
        ...opts,
      });
      const txt = await res.text();
      let data = null;
      try { data = txt ? JSON.parse(txt) : null; } catch(e){ data = null; }
      if(!res.ok){
        const detail = data && data.detail ? data.detail : (txt || ('HTTP ' + res.status));
        throw new Error(detail);
      }
      return data;
    }

    function persist(){
      localStorage.setItem('web_front_user', JSON.stringify(state.user || null));
      localStorage.setItem('web_front_role_id', state.role ? String(state.role.id) : '');
    }
    function clearPersist(){
      localStorage.removeItem('web_front_user');
      localStorage.removeItem('web_front_role_id');
    }
    function restore(){
      try{
        const u = JSON.parse(localStorage.getItem('web_front_user') || 'null');
        if(u && u.user_id && u.username) state.user = u;
      }catch(e){}
    }

    function setAuthPane(tab){
      $('tabLogin').classList.toggle('primary', tab==='login');
      $('tabRegister').classList.toggle('primary', tab==='register');
      $('loginPane').classList.toggle('hidden', tab!=='login');
      $('registerPane').classList.toggle('hidden', tab!=='register');
    }

    function getRoleDisplayName(name){
      const displayNames = {
        'doctor': '医生',
        'financial_advisor': '金融理财师',
        'financial_planner': '财务规划师',
        'investment_advisor': '投资顾问',
        'psychologist': '心理医生',
        'virtual_friend': '虚拟朋友',
        'teacher': '教师',
        'lawyer': '律师',
        'scientist': '科学家',
        'english_tutor': '英语学习助手',
        'stock_analyst': '股票分析师'
      };
      return displayNames[name] || name;
    }

    function getRoleIcon(name){
      const icons = {
        doctor: '医',
        financial_advisor: '财',
        financial_planner: '规',
        investment_advisor: '投',
        psychologist: '心',
        virtual_friend: '友',
        teacher: '师',
        lawyer: '法',
        scientist: '科',
        english_tutor: '英',
        stock_analyst: '股'
      };
      return icons[name] || 'AI';
    }

    function renderUserUI(){
      const loggedIn = !!state.user;
      $('authCard').classList.toggle('hidden', loggedIn);
      $('userCard').classList.toggle('hidden', !loggedIn);
      $('welcomeScreen').classList.toggle('hidden', loggedIn);
      $('chatScreen').classList.toggle('hidden', !loggedIn);
      
      $('msgInput').disabled = !loggedIn || !state.role || state.busy;
      $('btnSend').disabled = !loggedIn || !state.role || state.busy;
      $('btnEndChat').disabled = !loggedIn || !state.role;
      
      $('btnSend').textContent = '发送';
      if(loggedIn){
        $('userLine').textContent = `用户：${state.user.username}`;
      }
    }

    function renderRoles(){
      const list = $('roleList');
      list.innerHTML = '';
      for(const r of state.roles){
        const item = document.createElement('div');
        item.className = 'roleCard' + (state.role && String(state.role.id) === String(r.id) ? ' active' : '');
        item.dataset.roleId = String(r.id);
        item.innerHTML = `
          <div class="roleIcon">${getRoleIcon(r.name)}</div>
          <div class="roleMeta">
            <div class="roleName">${getRoleDisplayName(r.name)}</div>
            <div class="roleBrief">${r.description || '选择角色开始对话'}</div>
          </div>
        `;
        item.addEventListener('click', async () => {
          state.role = r;
          state.messages = [];
          persist();
          renderRoles();
          renderUserUI();
          renderChat();
        });
        list.appendChild(item);
      }
      $('roleDesc').textContent = state.role ? (state.role.description || '请选择对话角色') : '请选择对话角色';
    }

    function sanitizeDisplayText(text){
      let t = (text || '').toString().trim();
      try{
        if(t.startsWith('{') && t.endsWith('}')){
          const obj = JSON.parse(t);
          if(obj && typeof obj === 'object'){
            t = obj.response || obj.answer || obj.content || obj.text || t;
          }
        }
      }catch(e){}
      const withoutMd = t
        .replace(/^\\s*\\{\\s*"(response|answer|content|text)"\\s*:\\s*"/i, '')
        .replace(/"\\s*\\}\\s*$/i, '')
        .replace(/^\\s{0,3}#{1,6}\\s+/gm, '')
        .replace(/^\\s*[*-]\\s+/gm, '')
        .replace(/\\*\\*/g, '')
        .replace(/\\\\"/g, '"')
        .replace(/[#*]/g, '');
      return withoutMd.trim();
    }

    function renderChat(){
      const wrap = $('chatWrap');
      wrap.innerHTML = '';
      for(const m of state.messages){
        const row = document.createElement('div');
        row.className = 'msg ' + (m.role === 'user' ? 'user' : 'assistant');
        const av = document.createElement('div');
        av.className = 'avatar';
        av.textContent = (m.role === 'user') ? '你' : 'AI';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        
        const content = (m.role === 'assistant') ? sanitizeDisplayText(m.content || '') : (m.content || '');
        
        if(m.role === 'user' && m.fileData && m.fileData.type && m.fileData.type.startsWith('image/')){
          const img = document.createElement('img');
          img.src = m.fileData.data;
          img.className = 'msg-image';
          img.alt = '图片预览';
          bubble.appendChild(img);
        }
        
        const textNode = document.createTextNode(content);
        bubble.appendChild(textNode);
        
        row.appendChild(av);
        row.appendChild(bubble);
        wrap.appendChild(row);
      }
      wrap.scrollTop = wrap.scrollHeight;

      if(!state.user){
        $('chatTitle').textContent = '请先登录';
        $('chatDesc').textContent = '登录后选择角色开始多轮对话。';
      }else if(!state.role){
        $('chatTitle').textContent = '请选择角色';
        $('chatDesc').textContent = '选择角色开始对话。';
      }else{
        $('chatTitle').textContent = `与「${getRoleDisplayName(state.role.name)}」对话`;
        $('chatDesc').textContent = state.role.description || '—';
      }
    }

    async function health(){
      if(state.busy) return;
      
      try{
        const data = await api('/api/health', { method: 'GET', headers: {} });
        setBackendChip(true, '后端：正常');
      }catch(e){
        if(!state.busy){
          setBackendChip(false, '后端：不可用');
        }
      }
    }

    async function loadRoles(){
      const data = await api('/api/roles', { method: 'GET', headers: {} });
      state.roles = data.roles || [];
      if(state.roles.length === 0){
        state.role = null;
        renderRoles();
        renderUserUI();
        renderChat();
        return;
      }
      const savedId = localStorage.getItem('web_front_role_id');
      const first = state.roles[0];
      let chosen = first;
      if(savedId){
        const found = state.roles.find(r => String(r.id) === String(savedId));
        if(found) chosen = found;
      }
      state.role = chosen;
      renderRoles();
      renderUserUI();
      renderChat();
    }

    async function loadHistory(){
      if(!state.user || !state.role) return;
      $('loadHistoryText').classList.add('hidden');
      $('loadHistorySpinner').classList.remove('hidden');
      $('btnLoadHistory').disabled = true;
      try{
        const data = await api(`/api/chat_history/${state.user.user_id}/${state.role.id}`, { method: 'GET', headers: {} });
        state.messages = (data.history || []).map(m => ({
          role: (m.role === 'user' ? 'user' : 'assistant'),
          content: m.content || ''
        }));
        renderChat();
      }finally{
        $('loadHistoryText').classList.remove('hidden');
        $('loadHistorySpinner').classList.add('hidden');
        $('btnLoadHistory').disabled = false;
      }
    }

    async function clearHistory(){
      if(!state.user || !state.role) return;
      await api(`/api/clear_history/${state.user.user_id}/${state.role.id}`, { method: 'POST', body: JSON.stringify({}) });
      state.messages = [];
      renderChat();
    }

    async function stopChat(){
      if(!state.user || !state.role) return;
      
      state.stopped = true;
      
      if(state.currentReader){
        try{
          await state.currentReader.cancel();
        }catch(e){
          console.log('取消读取流:', e);
        }
        state.currentReader = null;
      }
      
      state.busy = false;
      state.paused = false;
      state.stopped = false;
      state.pendingBuffer = '';
      
      renderUserUI();
      renderChat();
      
      toast('对话已停止');
    }

    async function doLogin(){
      const username = $('loginUsername').value.trim();
      const password = $('loginPassword').value;
      const captcha = $('loginCaptcha').value.trim();
      const expectedCaptcha = $('captchaImg').dataset.captcha || '';
      
      if(!username || !password){ toast('请输入账号和密码'); return; }
      if(!captcha){ toast('请输入图形验证码'); return; }
      if(captcha.toLowerCase() !== expectedCaptcha.toLowerCase()){ 
        toast('验证码错误，请重试'); 
        refreshCaptcha();
        $('loginCaptcha').value = '';
        return; 
      }
      
      const data = await api('/api/login', { method: 'POST', body: JSON.stringify({ username, password }) });
      state.user = { user_id: data.user_id, username: data.username };
      state.messages = [];
      persist();
      await loadRoles();
      toast('登录成功');
      renderUserUI();
      renderChat();
    }

    async function doRegister(){
      const username = $('regUsername').value.trim();
      const password = $('regPassword').value;
      const captcha = $('regCaptcha').value.trim();
      const expectedCaptcha = $('regCaptchaImg').dataset.captcha || '';
      
      if(!username || !password){ toast('请输入账号和密码'); return; }
      if(!captcha){ toast('请输入图形验证码'); return; }
      if(captcha.toLowerCase() !== expectedCaptcha.toLowerCase()){ 
        toast('验证码错误，请重试'); 
        refreshRegCaptcha();
        $('regCaptcha').value = '';
        return; 
      }
      
      await api('/api/register', { method: 'POST', body: JSON.stringify({ username, password }) });
      toast('注册成功，请切换到登录');
      setAuthPane('login');
    }

    function splitToSmallParts(chunk){
      const text = (chunk || '').toString();
      const parts = [];
      let buf = '';
      for(const ch of text){
        buf += ch;
        if('。！？!?，,；;：:\\\\n'.includes(ch) || buf.length >= 6){
          parts.push(buf);
          buf = '';
        }
      }
      if(buf) parts.push(buf);
      return parts;
    }

    async function streamAssistantReply(userText, fileData = null){
      const payload = { 
        user_id: state.user.user_id, 
        role_id: state.role.id, 
        message: userText 
      };
      
      if(fileData){
        payload.file = fileData;
      }
      
      const resp = await fetch('/api/chat_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if(!resp.ok){
        const err = await resp.text();
        throw new Error(err || ('HTTP ' + resp.status));
      }
      if(!resp.body){
        throw new Error('浏览器不支持流式读取');
      }

      const assistantMsg = { role: 'assistant', content: '' };
      state.messages.push(assistantMsg);
      renderChat();

      const reader = resp.body.getReader();
      state.currentReader = reader;
      const decoder = new TextDecoder('utf-8');

      try{
        while(true){
          if(state.stopped){
            break;
          }
          
          const { value, done } = await reader.read();
          if(done) break;
          
          const chunk = decoder.decode(value, { stream: true });
          const smallParts = splitToSmallParts(chunk);
          for(const p of smallParts){
            if(state.stopped){
              break;
            }
            if(state.paused){
              state.pendingBuffer += p;
            }else{
              assistantMsg.content += p;
            }
            renderChat();
            await new Promise(resolve => setTimeout(resolve, 30));
          }
          
          if(state.stopped){
            break;
          }
        }
      }finally{
        state.currentReader = null;
        if(state.stopped){
          try{
            await reader.cancel();
          }catch(e){
            console.log('取消读取:', e);
          }
        }
      }
    }

    async function sendMsg(){
      const text = $('msgInput').value.trim();
      if(!text || !state.user || !state.role || state.busy) return;
      $('msgInput').value = '';
      
      const fileData = pendingFileData;
      pendingFileData = null;
      hideFilePreview();
      
      const msg = { role: 'user', content: text };
      if(fileData && fileData.type && fileData.type.startsWith('image/')){
        msg.fileData = fileData;
      }
      state.messages.push(msg);
      renderChat();
      state.busy = true;
      state.paused = false;
      state.stopped = false;
      state.pendingBuffer = '';
      renderUserUI();
      try{
        await streamAssistantReply(text, fileData);
      }catch(e){
        toast('发送失败：' + e.message);
      }finally{
        state.busy = false;
        state.paused = false;
        state.stopped = false;
        state.pendingBuffer = '';
        state.currentReader = null;
        renderUserUI();
      }
    }

    function togglePauseResume(){
      if(!state.busy) return;
      state.paused = !state.paused;
      if(!state.paused && state.pendingBuffer){
        const last = state.messages.length ? state.messages[state.messages.length - 1] : null;
        if(last && last.role === 'assistant'){
          last.content += state.pendingBuffer;
        }
        state.pendingBuffer = '';
      }
      renderUserUI();
      renderChat();
    }

    function bind(){
      $('tabLogin').addEventListener('click', () => setAuthPane('login'));
      $('tabRegister').addEventListener('click', () => setAuthPane('register'));
      $('btnLogin').addEventListener('click', async () => {
        try{ await doLogin(); }catch(e){ toast('登录失败：' + e.message); }
      });
      $('btnRegister').addEventListener('click', async () => {
        try{ await doRegister(); }catch(e){ toast('注册失败：' + e.message); }
      });
      $('btnLogout').addEventListener('click', () => {
        state.user = null;
        state.role = null;
        state.messages = [];
        clearPersist();
        renderUserUI();
        renderChat();
        toast('已退出');
      });
      $('btnLoadHistory').addEventListener('click', async () => {
        try{ await loadHistory(); toast('历史已加载'); }catch(e){ toast('加载失败：' + e.message); }
      });
      $('btnClearHistory').addEventListener('click', async () => {
        try{ await clearHistory(); toast('历史已清空'); }catch(e){ toast('清空失败：' + e.message); }
      });
      $('btnSend').addEventListener('click', () => {
        sendMsg();
      });
      $('btnEndChat').addEventListener('click', async () => {
        try{
          await stopChat();
        }catch(e){
          toast('停止对话失败：' + e.message);
        }
      });
      $('btnStockQuick').addEventListener('click', async () => {
        const kw = $('stockQuick').value.trim();
        if(!kw){ toast('请输入股票代码或简称'); return; }
        $('msgInput').value = `查询股票 ${kw}`;
        await sendMsg();
      });
      $('msgInput').addEventListener('keydown', (e) => {
        if(e.key === 'Enter' && !e.shiftKey){
          e.preventDefault();
          if(!state.busy) sendMsg();
        }
      });
      
      $('btnQuick').addEventListener('click', () => {
        const quickPrompts = $('quickPrompts');
        quickPrompts.style.display = quickPrompts.style.display === 'none' ? 'block' : 'none';
      });
      
      $('btnCloseQuick').addEventListener('click', () => {
        $('quickPrompts').style.display = 'none';
      });
      
      document.querySelectorAll('.quick-prompt').forEach(el => {
        el.addEventListener('click', () => {
          $('msgInput').value = el.textContent;
          $('quickPrompts').style.display = 'none';
          sendMsg();
        });
      });
      
      $('btnEmoji').addEventListener('click', () => {
        const emojiPanel = $('emojiPanel');
        const quickPrompts = $('quickPrompts');
        
        quickPrompts.style.display = 'none';
        
        emojiPanel.style.display = emojiPanel.style.display === 'none' ? 'block' : 'none';
        
        if(emojiPanel.style.display === 'block'){
          initEmojiPanel();
        }
      });
      
      $('btnCloseEmoji').addEventListener('click', () => {
        $('emojiPanel').style.display = 'none';
      });
      
      $('btnAttach').addEventListener('click', () => {
        $('fileInput').click();
      });
      
      $('fileInput').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if(file){
          handleFileUpload(file);
        }
        e.target.value = '';
      });
      
      $('btnRemoveFile').addEventListener('click', () => {
        removePendingFile();
      });
      
      $('btnRecord').addEventListener('click', () => {
        if(mediaRecorder && mediaRecorder.state === 'recording'){
          stopRecording();
        }else{
          startRecording();
        }
      });
      
      document.addEventListener('click', (e) => {
        const emojiPanel = $('emojiPanel');
        const quickPrompts = $('quickPrompts');
        const btnEmoji = $('btnEmoji');
        const btnQuick = $('btnQuick');
        
        if(!btnEmoji.contains(e.target) && !emojiPanel.contains(e.target)){
          emojiPanel.style.display = 'none';
        }
        if(!btnQuick.contains(e.target) && !quickPrompts.contains(e.target)){
          quickPrompts.style.display = 'none';
        }
      });
    }
    
    const emojiList = [
      '😀', '😃', '😄', '😁', '😆', '😅', '🤣', '😂', '🙂', '😊',
      '😇', '🥰', '😍', '🤩', '😘', '😗', '😚', '😙', '🥲', '😋',
      '😛', '😜', '🤪', '😝', '🤑', '🤗', '🤭', '🤫', '🤔', '🤐',
      '🤨', '😐', '😑', '😶', '😏', '😒', '🙄', '😬', '🤥', '😌',
      '😔', '😪', '🤤', '😴', '😷', '🤒', '🤕', '🤢', '🤮', '🥵',
      '🥶', '🥴', '😵', '🤯', '🤠', '🥳', '🥸', '😎', '🤓', '🧐',
      '😕', '😟', '🙁', '☹️', '😮‍💨', '😌', '😴', '🤤', '😪', '😫',
      '🥱', '😴', '😵‍💫', '🤯', '🤠', '🥳', '🥸', '😎', '🤓', '🧐',
      '👋', '🤚', '✋', '🖐️', '🖖', '🤟', '🤞', '🤝', '👍', '👎',
      '✊', '🤛', '🤜', '🤞', '✌️', '🤟', '🤘', '👌', '👈', '👉',
      '👆', '👇', '☝️', '✋', '🤚', '🖐️', '🖖', '👋', '🤙', '💪',
      '🦾', '🦵', '🦿', '🦶', '👀', '👁️', '👅', '👄', '👃', '🧠',
      '🫀', '🫁', '🦷', '🦴', '👂', '👃', '👣', '🦶', '🦵', '🦿',
      '🦾', '💪', '🤲', '🙌', '👏', '🙏', '🤝', '👍', '👎', '👊'
    ];
    
    function initEmojiPanel(){
      const grid = $('emojiGrid');
      if(grid.children.length > 0) return;
      
      emojiList.forEach(emoji => {
        const item = document.createElement('div');
        item.className = 'emoji-item';
        item.textContent = emoji;
        item.addEventListener('click', () => {
          insertEmoji(emoji);
          $('emojiPanel').style.display = 'none';
        });
        grid.appendChild(item);
      });
    }
    
    function insertEmoji(emoji){
      const textarea = $('msgInput');
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const value = textarea.value;
      
      textarea.value = value.substring(0, start) + emoji + value.substring(end);
      
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + emoji.length;
        textarea.focus();
      }, 0);
    }
    
    let pendingFileData = null;
    
    let mediaRecorder = null;
    let audioChunks = [];
    
    function checkMediaSupport(){
      if('MediaRecorder' in window && navigator.mediaDevices && navigator.mediaDevices.getUserMedia){
        $('btnRecord').style.display = 'grid';
        return true;
      }
      return false;
    }
    
    async function startRecording(){
      try{
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (e) => {
          if(e.data.size > 0){
            audioChunks.push(e.data);
          }
        };
        
        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
          handleAudioRecording(audioBlob);
        };
        
        mediaRecorder.start();
        $('btnRecord').textContent = '⏹️';
        $('btnRecord').title = '停止录制';
        $('recordingIndicator').style.display = 'flex';
        toast('开始录音，再次点击停止');
        
      }catch(e){
        console.error('录音失败:', e);
        toast('录音失败: ' + e.message);
      }
    }
    
    function stopRecording(){
      if(mediaRecorder && mediaRecorder.state !== 'inactive'){
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        $('btnRecord').textContent = '🎤';
        $('btnRecord').title = '语音录制';
        $('recordingIndicator').style.display = 'none';
      }
    }
    
    function handleAudioRecording(audioBlob){
      const fileName = `recording_${Date.now()}.webm`;
      const reader = new FileReader();
      
      reader.onload = (e) => {
        const base64Data = e.target.result;
        
        pendingFileData = {
          name: fileName,
          type: 'audio/webm',
          data: base64Data
        };
        
        showFilePreview(fileName, base64Data);
        $('msgInput').value = `[语音] ${fileName}\n\n请帮我识别这段语音的内容。`;
        toast('录音完成，点击发送');
      };
      
      reader.readAsDataURL(audioBlob);
    }
    
    function handleFileUpload(file){
      const maxSize = 50 * 1024 * 1024;
      
      if(file.size > maxSize){
        toast('文件大小不能超过50MB');
        return;
      }
      
      console.log(`准备上传文件: ${file.name}, 大小: ${file.size} bytes, 类型: ${file.type}`);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64Data = e.target.result;
        const fileName = file.name;
        const fileType = file.type;
        
        console.log(`文件读取完成: ${base64Data.length} 字符`);
        
        pendingFileData = {
          name: fileName,
          type: fileType,
          data: base64Data
        };
        
        showFilePreview(fileName, base64Data);
        
        if(fileType.startsWith('image/')){
          $('msgInput').value = `[图片] ${fileName}\n\n请帮我分析这张图片的内容。`;
        }else if(fileType.startsWith('audio/')){
          $('msgInput').value = `[语音] ${fileName}\n\n请帮我识别这段语音的内容。`;
        }else if(fileType.startsWith('video/')){
          $('msgInput').value = `[视频] ${fileName}\n\n请帮我分析这个视频中的内容。`;
        }else{
          $('msgInput').value = `[文件] ${fileName}\n\n请帮我分析这个文件的内容。`;
        }
        toast('文件已准备好发送');
      };
      
      reader.onerror = (error) => {
        console.error('文件读取失败:', error);
        toast('文件读取失败: ' + error.message);
        pendingFileData = null;
      };
      
      reader.readAsDataURL(file);
    }
    
    function showFilePreview(fileName, base64Data){
      const preview = $('filePreview');
      const previewImg = $('previewImg');
      const previewName = $('previewName');
      
      previewName.textContent = fileName;
      
      if(base64Data.startsWith('data:image/')){
        previewImg.src = base64Data;
        previewImg.style.display = 'block';
      }else{
        previewImg.style.display = 'none';
      }
      
      preview.classList.add('show');
    }
    
    function hideFilePreview(){
      const preview = $('filePreview');
      preview.classList.remove('show');
      $('previewImg').src = '';
      $('previewName').textContent = '';
    }
    
    function removePendingFile(){
      pendingFileData = null;
      hideFilePreview();
      $('msgInput').value = '';
      toast('文件已移除');
    }

    async function main(){
      bind();
      setAuthPane('login');
      checkMediaSupport();
      clearPersist();
      state.user = null;
      state.role = null;
      refreshCaptcha();
      refreshRegCaptcha();
      await health();
      renderUserUI();
      renderChat();
      setInterval(health, 8000);
    }

    main().catch((e) => {
      console.error('frontend init failed:', e);
      setBackendChip(false, '后端：不可用');
      toast('页面初始化失败：' + (e && e.message ? e.message : '未知错误'));
    });
  </script>
</body>
</html>"""

# 根路径路由：返回主页面（HTML）
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """返回主页面，将CSS样式注入到HTML模板中"""
    # 将 HTML 模板中的占位符 "{css_styles}" 替换为实际的 CSS 样式字符串，然后返回完整的 HTML 页面
    return HTML_TEMPLATE.replace("{css_styles}", CSS_STYLES)


# 健康检查接口：用于探测前端是否能正常访问后端服务
@app.get("/api/health")
def api_health() -> JSONResponse:
    """健康检查接口"""
    try:
        # 向后端的 /health 端点发送 GET 请求，超时 5 秒，检查后端是否存活
        resp = _proxy_get("/health", timeout=5)
        # 如果后端返回 HTTP 200，则认为后端健康
        if resp.status_code == 200:
            return JSONResponse({"status": "ok"})
        # 如果响应状态码不是 200，抛出 502 错误（网关错误）
        raise HTTPException(status_code=502, detail="backend health not ok")
    except requests.RequestException:
        # 网络异常或请求超时时，认为后端不可达，抛出 502
        raise HTTPException(status_code=502, detail="backend unreachable")


# 获取角色列表接口：从后端获取所有可用角色
@app.get("/api/roles")
def api_roles() -> JSONResponse:
    """获取角色列表接口"""
    try:
        # 向后端的 /roles 端点发送 GET 请求，超时 10 秒
        resp = _proxy_get("/roles", timeout=10)
        # 如果后端响应不是 200，则抛出异常
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"backend roles not ok: {resp.status_code}")
        # 解析后端返回的 JSON 数据（可能为空）
        data = resp.json() if resp.content else []
        # 构建前端需要的角色列表格式
        roles: List[Dict[str, Any]] = []
        # 遍历后端返回的每个角色
        for r in (data or []):
            roles.append(
                {
                    "id": int(r.get("id")),                     # 角色ID，转为整数
                    "name": r.get("role_name") or r.get("name") or f"role_{r.get('id')}",  # 角色名称，优先取 role_name
                    "description": r.get("description") or "",   # 角色描述，缺失时为空字符串
                }
            )
        # 如果后端没有返回任何角色，则提供一个默认角色作为兜底
        if not roles:
            roles = [{"id": 1, "name": "financial_advisor", "description": "默认角色"}]
        # 返回符合前端预期的 JSON 结构：{"roles": [...]}
        return JSONResponse({"roles": roles})
    except requests.RequestException:
        # 后端不可达时抛出 502
        raise HTTPException(status_code=502, detail="backend unreachable")


@app.post("/api/register")
def api_register(req: RegisterRequest) -> Response:
    """用户注册接口"""
    try:
        resp = _proxy_post("/register", _model_to_dict(req), timeout=15)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.post("/api/login")
def api_login(req: LoginRequest) -> Response:
    """用户登录接口"""
    try:
        resp = _proxy_post("/login", _model_to_dict(req), timeout=15)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.get("/api/chat_history/{user_id}/{role_id}")
def api_chat_history(user_id: int, role_id: int) -> Response:
    """获取聊天历史接口"""
    try:
        resp = _proxy_get(f"/chat_history/{user_id}/{role_id}", timeout=20)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.post("/api/clear_history/{user_id}/{role_id}")
def api_clear_history(user_id: int, role_id: int) -> Response:
    """清空聊天历史接口"""
    try:
        resp = _proxy_post(f"/clear_history/{user_id}/{role_id}", payload={}, timeout=20)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> Response:
    """非流式聊天接口"""
    try:
        resp = _proxy_post("/chat", _model_to_dict(req), timeout=90)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.post("/api/chat_stream")
def api_chat_stream(req: ChatRequest) -> StreamingResponse:
    """流式聊天接口"""
    payload = _model_to_dict(req)
    payload["stream"] = True

    try:
        upstream = requests.post(
            f"{BACKEND_BASE_URL}/api/chat_stream",
            json=payload,
            timeout=120,
            stream=True,
            headers={"Accept": "text/plain"},
        )
    except requests.RequestException as e:
        print(f"Backend request error: {e}")
        raise HTTPException(status_code=502, detail="backend unreachable")

    if upstream.status_code != 200:
        body = upstream.text
        print(f"Backend error: {upstream.status_code} - {body}")
        return StreamingResponse(
            iter([body]),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/plain; charset=utf-8"),
        )

    def generate():
        try:
            buffer = ""
            for chunk in upstream.iter_content(chunk_size=128, decode_unicode=True):
                if chunk:
                    buffer += chunk
                    while len(buffer) > 0:
                        try:
                            yield buffer[0]
                            buffer = buffer[1:]
                        except Exception:
                            break
                    print(f"Processed {len(chunk)} bytes, buffer remaining: {len(buffer)}")
        except Exception as e:
            print(f"Stream generation error: {e}")
        finally:
            upstream.close()
            print("Stream closed")

    response = StreamingResponse(
        generate(), 
        media_type="text/plain; charset=utf-8",
        headers={
            "Transfer-Encoding": "chunked",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
    return response


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("FRONTEND_HOST", "127.0.0.1")
    port = _resolve_frontend_port(default_port=8503, host=host)
    print(f"[web_frontend] proxy backend -> {BACKEND_BASE_URL}")
    print(f"[web_frontend] serving on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="debug")
