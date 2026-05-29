from __future__ import annotations

"""
角色扮演系统 Web 前端（FastAPI 单文件页面）。

职责：
- 提供登录、注册、角色选择与聊天 UI
- 代理转发后端接口（同源访问，避免浏览器跨域问题）
- 支持流式聊天输出与健康检查状态展示
"""

import os
import socket
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel


load_dotenv()

DEFAULT_BACKEND_CANDIDATES = [
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003",
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", DEFAULT_BACKEND_CANDIDATES[0]).rstrip("/")

app = FastAPI(title="Web Frontend (no Streamlit)")


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class FileData(BaseModel):
    name: str
    type: str
    data: str

class ChatRequest(BaseModel):
    user_id: int
    role_id: int
    message: str
    file: Optional[FileData] = None


def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """
    Compatible with both Pydantic v1 (.dict) and v2 (.model_dump).
    """
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()  # type: ignore[call-arg]


@app.exception_handler(Exception)
async def _unhandled_exception_handler(_request, exc: Exception):
    # Return the actual error message so the UI can show it,
    # otherwise users only see "Internal Server Error".
    return JSONResponse(
        status_code=500,
        content={"detail": f"frontend error: {type(exc).__name__}: {str(exc)}"},
    )


def _proxy_get(path: str, timeout: int = 15) -> requests.Response:
    return requests.get(f"{BACKEND_BASE_URL}{path}", timeout=timeout)


def _proxy_post(path: str, payload: Dict[str, Any], timeout: int = 30) -> requests.Response:
    return requests.post(f"{BACKEND_BASE_URL}{path}", json=payload, timeout=timeout)


def _health_check_backend(base_url: str, timeout: int = 3) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        if resp.status_code != 200:
            return False
        data = resp.json() if resp.content else {}
        return data.get("service") == "financial-rag-system"
    except requests.RequestException:
        return False
    except ValueError:
        return False


def _resolve_backend_base_url() -> str:
    env_url = os.getenv("BACKEND_BASE_URL")
    if env_url:
        env_url = env_url.rstrip("/")
        if _health_check_backend(env_url):
            return env_url
    for candidate in DEFAULT_BACKEND_CANDIDATES:
        if _health_check_backend(candidate):
            return candidate
    return (env_url or DEFAULT_BACKEND_CANDIDATES[0]).rstrip("/")


BACKEND_BASE_URL = _resolve_backend_base_url()


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # Single-file HTML (CSS+JS inline) to keep it easy to run.
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI聊天系统</title>
  <style>
    :root{
      --primary: #3B82F6;
      --primary-light: #589CFC;
      --secondary: #8B5CF6;
      --text-primary: #1F2937;
      --text-secondary: #6B7280;
      --text-muted: #9CA3AF;
      --border: #E5E7EB;
      --border-light: #F3F4F6;
      --bg: #F9FAFB;
      --panel: #FFFFFF;
      --bg-chat: #F3F4F6;
      --danger: #EF4444;
      --success: #10B981;
      --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);
      --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.04), 0 2px 4px rgba(0, 0, 0, 0.03), 0 1px 3px rgba(0, 0, 0, 0.02);
      --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.05), 0 4px 6px rgba(0, 0, 0, 0.04), 0 2px 4px rgba(0, 0, 0, 0.03);
      --shadow-hover: 0 12px 20px rgba(0, 0, 0, 0.08), 0 6px 8px rgba(0, 0, 0, 0.05), 0 2px 4px rgba(0, 0, 0, 0.03);
      --radius-sm: 12px;
      --radius-md: 16px;
      --radius-lg: 20px;
      --radius-xl: 24px;
      --radius-bubble: 18px;
      --transition-fast: 0.15s ease-in-out;
      --transition-normal: 0.2s ease-in-out;
      --transition-slow: 0.3s ease-in-out;
    }

    @media (prefers-color-scheme: dark) {
      :root{
        --primary: #60A5FA;
        --primary-light: #93C5FD;
        --secondary: #A78BFA;
        --text-primary: #F9FAFB;
        --text-secondary: #D1D5DB;
        --text-muted: #9CA3AF;
        --border: #374151;
        --border-light: #1F2937;
        --bg: #111827;
        --panel: #1F2937;
        --bg-chat: #111827;
        --danger: #F87171;
        --success: #34D399;
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.2);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3), 0 2px 4px rgba(0, 0, 0, 0.2), 0 1px 3px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.35), 0 4px 6px rgba(0, 0, 0, 0.25), 0 2px 4px rgba(0, 0, 0, 0.15);
        --shadow-hover: 0 12px 20px rgba(0, 0, 0, 0.4), 0 6px 8px rgba(0, 0, 0, 0.3), 0 2px 4px rgba(0, 0, 0, 0.2);
      }
      body{
        background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #1f2937 100%);
      }
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body{
      margin:0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: linear-gradient(135deg, #F3F4F6 0%, #E0E7FF 50%, #F3F4F6 100%);
      color: var(--text-primary);
      font-weight: 400;
      min-height: 100vh;
      transition: background 0.3s ease-in-out;
    }
    a{ color: inherit; }
    .wrap{ max-width: 1300px; margin: 0 auto; padding: 20px 24px; }
    .topbar{
      display:flex; align-items:center; justify-content:space-between;
      gap:16px; padding: 14px 20px; border: 1px solid var(--border);
      background: var(--panel);
      border-radius: var(--radius-lg); box-shadow: var(--shadow-md);
    }
    .brand{ display:flex; align-items:center; gap:12px; }
    .logo{
      width: 42px; height: 42px; border-radius: var(--radius-sm);
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      display:grid; place-items:center; font-weight: 700;
      color: white;
      font-size: 18px;
      box-shadow: var(--shadow-sm);
    }
    .title{ font-weight: 600; letter-spacing: -.02em; font-size: 18px; color: var(--text-primary); }
    .sub{ font-size: 12px; color: var(--text-secondary); margin-top: 1px; }
    .row{ 
      display:flex; 
      gap: 20px; 
      margin-top: 20px; 
      position: relative;
      height: calc(100vh - 140px);
    }
    .left{
      width: 320px; 
      flex: 0 0 320px;
      border: 1px solid var(--border); 
      border-radius: var(--radius-lg);
      background: var(--panel); 
      box-shadow: var(--shadow-md);
      padding: 0;
      overflow: hidden;
      position: fixed;
      left: calc(50% - 630px);
      top: 100px;
      height: calc(100vh - 140px);
      overflow-y: auto;
      z-index: 100;
    }
    .left.hidden{
      display: none;
    }
    .main{
      flex: 1 1 auto;
      margin-left: 340px;
      border: 1px solid var(--border); 
      border-radius: var(--radius-lg);
      background: var(--panel); 
      box-shadow: var(--shadow-md);
      padding: 0;
      height: calc(100vh - 140px);
      display:flex; 
      flex-direction:column;
      overflow: hidden;
    }
    .card{
      background: var(--panel);
      padding: 16px;
    }
    .card-section{
      padding: 16px;
      border-bottom: 1px solid var(--border-light);
    }
    .card-section:last-child{
      border-bottom: none;
    }
    .card-section.compact{
      padding: 14px 16px;
    }
    .card-header{
      background: linear-gradient(135deg, var(--bg-chat), var(--border-light));
      padding: 14px 16px;
      border-bottom: 1px solid var(--border-light);
    }
    .kpis{ display:flex; gap:10px; flex-wrap:wrap; margin-top: 8px; }
    .chip{
      display:inline-flex; align-items:center; gap:8px;
      padding: 5px 12px; border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--bg-chat);
      font-size: 12px;
      color: var(--text-secondary);
      transition: all var(--transition-fast);
    }
    .chip.success{
      background: rgba(16, 185, 129, 0.1);
      border-color: rgba(16, 185, 129, 0.3);
      color: var(--success);
    }
    .grid2{ display:grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    label{ font-size: 12px; color: var(--text-secondary); display:block; margin-bottom: 6px; font-weight: 500; }
    .form-label{ font-size: 13px; color: var(--text-primary); display:block; margin-bottom: 8px; font-weight: 500; }
    input, select, textarea{
      width: 100%;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: var(--bg-chat);
      color: var(--text-primary);
      padding: 10px 14px;
      outline: none;
      transition: border-color var(--transition-fast), background var(--transition-fast), box-shadow var(--transition-fast);
      font-size: 14px;
    }
    input:hover, select:hover, textarea:hover{
      border-color: var(--text-muted);
    }
    input:focus, select:focus, textarea:focus{
      border-color: var(--primary);
      background: var(--panel);
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    textarea{ min-height: 80px; resize: vertical; }
    input::placeholder, textarea::placeholder{ color: var(--text-muted); }
    
    .btn{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: var(--bg-chat);
      color: var(--text-primary);
      padding: 10px 16px;
      cursor: pointer;
      transition: all var(--transition-normal);
      font-weight: 500;
      font-size: 13px;
      line-height: 1;
    }
    .btn:hover{ 
      background: var(--border-light); 
      border-color: var(--text-muted);
      transform: translateY(-1px);
    }
    .btn:active{ 
      transform: translateY(0); 
    }
    .btn:disabled{
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }
    .btn.primary{ 
      background: linear-gradient(135deg, var(--primary), var(--primary-light)); 
      border-color: var(--primary); 
      color: white;
      box-shadow: var(--shadow-sm);
    }
    .btn.primary:hover{ 
      background: linear-gradient(135deg, var(--primary-light), var(--primary)); 
      border-color: var(--primary-light);
      box-shadow: var(--shadow-md);
      transform: translateY(-2px);
    }
    .btn.primary:active{ 
      transform: translateY(-1px); 
    }
    .btn.good{ 
      background: linear-gradient(135deg, var(--success), #34D399); 
      border-color: var(--success); 
      color: white;
      box-shadow: var(--shadow-sm);
    }
    .btn.good:hover{ 
      background: linear-gradient(135deg, #34D399, var(--success)); 
      border-color: #34D399;
      box-shadow: var(--shadow-md);
      transform: translateY(-2px);
    }
    .btn.danger{ 
      background: transparent; 
      border-color: var(--danger); 
      color: var(--danger);
    }
    .btn.danger:hover{ 
      background: rgba(239, 68, 68, 0.05); 
    }
    .btn.outline{
      background: transparent;
      border-color: var(--border);
      color: var(--text-secondary);
    }
    .btn.outline:hover{
      background: var(--bg-chat);
      border-color: var(--text-muted);
    }
    .btn.small{
      padding: 6px 12px;
      font-size: 12px;
    }
    .btn.full{ width: 100%; }
    .text-muted{ color: var(--text-muted); font-size: 12px; }
    .text-secondary{ color: var(--text-secondary); font-size: 13px; }
    .hr{ height:1px; background: var(--border-light); margin: 0; }
    .hidden{ display:none !important; }
    .roleList{
      display:flex;
      flex-direction:column;
      gap: 8px;
      max-height: 380px;
      overflow:auto;
      padding-right: 4px;
    }
    .roleList::-webkit-scrollbar{
      width: 6px;
    }
    .roleList::-webkit-scrollbar-track{
      background: transparent;
    }
    .roleList::-webkit-scrollbar-thumb{
      background: var(--border);
      border-radius: 3px;
    }
    .roleList::-webkit-scrollbar-thumb:hover{
      background: var(--text-muted);
    }
    .roleCard{
      display:flex;
      align-items:flex-start;
      gap:12px;
      padding:14px;
      border:1px solid var(--border);
      border-radius: var(--radius-md);
      background: var(--panel);
      cursor:pointer;
      transition: all var(--transition-normal);
      position: relative;
    }
    .roleCard:hover{
      border-color: var(--primary);
      transform: translateY(-2px);
      box-shadow: var(--shadow-lg);
    }
    .roleCard.active{
      border-color: var(--primary);
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.06));
      box-shadow: var(--shadow-lg);
    }
    .roleCard.active::before{
      content: '';
      position: absolute;
      left: 0;
      top: 50%;
      transform: translateY(-50%);
      width: 3px;
      height: 24px;
      background: linear-gradient(180deg, var(--primary), var(--secondary));
      border-radius: 0 3px 3px 0;
    }
    .roleIcon{
      width:44px;
      height:44px;
      border-radius: var(--radius-sm);
      flex:0 0 44px;
      display:grid;
      place-items:center;
      font-size: 16px;
      font-weight: 700;
      color:#fff;
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      box-shadow: var(--shadow-sm);
    }
    .roleMeta{
      min-width:0;
      flex:1;
    }
    .roleName{
      font-size:15px;
      font-weight:600;
      color:var(--text-primary);
      margin-bottom:3px;
    }
    .roleBrief{
      font-size:12px;
      color:var(--text-secondary);
      line-height:1.5;
      display:-webkit-box;
      -webkit-line-clamp:2;
      -webkit-box-orient:vertical;
      overflow:hidden;
    }

    .chatHeader{ 
      display:flex; 
      align-items:flex-start; 
      justify-content:space-between; 
      gap: 12px; 
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-light);
    }
    .chatTitle{ font-size: 16px; font-weight: 600; color: var(--text-primary); }
    .chatDesc{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }
    #chatScreen{
      display: flex;
      flex-direction: column;
      height: 100%;
    }
    .chatWrap{
      flex: 1 1 auto;
      overflow-y: auto;
      overflow-x: hidden;
      padding: 16px 20px;
      -webkit-overflow-scrolling: touch;
      background: var(--bg-chat);
    }
    .chatWrap::-webkit-scrollbar{
      width: 6px;
    }
    .chatWrap::-webkit-scrollbar-track{
      background: transparent;
    }
    .chatWrap::-webkit-scrollbar-thumb{
      background: var(--border);
      border-radius: 3px;
    }
    .chatWrap::-webkit-scrollbar-thumb:hover{
      background: var(--text-muted);
    }
    .msg{ 
      display:flex; 
      gap: 12px; 
      margin: 16px 0; 
      max-width: 100%;
    }
    .msg.user{ justify-content: flex-start; flex-direction: row-reverse; }
    
    .msg-image{
      max-width: 200px;
      max-height: 200px;
      border-radius: var(--radius-sm);
      object-fit: cover;
      margin-bottom: 8px;
    }
    
    .file-preview{
      position: absolute;
      bottom: 100%;
      left: 0;
      right: 0;
      padding: 12px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      box-shadow: var(--shadow-lg);
      margin-bottom: 8px;
      display: none;
      flex-direction: column;
      gap: 8px;
    }
    
    .file-preview.show{
      display: flex;
    }
    
    .preview-image{
      max-width: 150px;
      max-height: 150px;
      border-radius: var(--radius-sm);
      object-fit: cover;
      align-self: flex-start;
    }
    
    .preview-info{
      display: flex;
      align-items: center;
      gap: 10px;
    }
    
    .preview-name{
      font-size: 13px;
      color: var(--text-primary);
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    
    .btn-remove-file{
      padding: 4px 10px;
      font-size: 12px;
      background: transparent;
      border: 1px solid var(--danger);
      color: var(--danger);
      border-radius: var(--radius-sm);
      cursor: pointer;
    }
    
    .btn-remove-file:hover{
      background: rgba(239, 68, 68, 0.08);
    }
    
    .recording-indicator{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      border-radius: var(--radius-sm);
      color: var(--danger);
      font-size: 13px;
      animation: pulse 1.5s ease-in-out infinite;
    }
    
    .recording-dot{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--danger);
      animation: blink 1s ease-in-out infinite;
    }
    
    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.3; }
    }
    
    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
      50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
    }
    
    .recording-text{
      font-weight: 500;
    }
    
    .msg.assistant{ justify-content: flex-start; }
    .avatar{
      width: 36px; 
      height: 36px; 
      border-radius: 50%;
      display:grid; 
      place-items:center; 
      font-weight: 600;
      background: var(--border-light);
      border: 1px solid var(--border);
      flex: 0 0 36px;
      color: var(--text-secondary);
      font-size: 14px;
    }
    .bubble{
      border-radius: var(--radius-bubble);
      padding: 14px 18px;
      max-width: 75%;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.6;
      font-size: 15px;
      position: relative;
    }
    .msg.user .avatar{ 
      background: linear-gradient(135deg, var(--secondary), #F472B6); 
      border-color: rgba(139, 92, 246, 0.3); 
      color: white;
      box-shadow: var(--shadow-sm);
    }
    .msg.user .bubble{ 
      background: linear-gradient(135deg, var(--primary-light), var(--primary)); 
      color: white;
      border-radius: var(--radius-bubble) var(--radius-bubble) 6px var(--radius-bubble);
      box-shadow: var(--shadow-sm);
    }
    .msg.user .bubble::after{
      content: '';
      position: absolute;
      right: -6px;
      top: 12px;
      width: 0;
      height: 0;
      border-top: 6px solid transparent;
      border-bottom: 6px solid transparent;
      border-left: 6px solid var(--primary);
    }
    .msg.assistant .avatar{ 
      background: linear-gradient(135deg, var(--primary), var(--secondary)); 
      border-color: rgba(59, 130, 246, 0.3); 
      color: white;
      box-shadow: var(--shadow-sm);
    }
    .msg.assistant .bubble{ 
      background: var(--panel);
      border: 1px solid var(--border);
      color: var(--text-primary);
      border-radius: var(--radius-bubble) var(--radius-bubble) var(--radius-bubble) 6px;
      box-shadow: var(--shadow-sm);
    }
    .msg.assistant .bubble::after{
      content: '';
      position: absolute;
      left: -6px;
      top: 12px;
      width: 0;
      height: 0;
      border-top: 6px solid transparent;
      border-bottom: 6px solid transparent;
      border-right: 6px solid var(--border);
    }
    .msg-time{
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 4px;
      text-align: right;
    }
    .msg.user .msg-time{
      text-align: left;
    }

    .composer{
      padding: 16px 20px;
      background: var(--panel);
      border-top: 1px solid var(--border-light);
      display:flex; 
      gap: 12px; 
      align-items:flex-end;
      position: relative;
    }
    .composer .input-wrapper{
      flex: 1;
      position: relative;
    }
    .composer textarea{ 
      width: 100%;
      min-height: 56px; 
      max-height: 180px;
      border-radius: var(--radius-xl);
      border: 1px solid var(--border);
      background: var(--bg-chat);
      padding: 14px 50px 14px 16px;
      resize: none;
      transition: all var(--transition-normal);
      font-size: 15px;
      line-height: 1.5;
    }
    .composer textarea:focus{
      border-color: var(--primary);
      background: var(--panel);
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.08), 0 4px 12px rgba(59, 130, 246, 0.08);
    }
    .composer-actions{
      position: absolute;
      right: 12px;
      bottom: 12px;
      display: flex;
      gap: 8px;
    }
    .composer-btn{
      width: 28px;
      height: 28px;
      border-radius: 50%;
      border: none;
      background: transparent;
      color: var(--text-muted);
      cursor: pointer;
      display: grid;
      place-items: center;
      transition: all var(--transition-fast);
    }
    .composer-btn:hover{
      background: var(--border-light);
      color: var(--text-primary);
    }
    .composer-hint{
      position: absolute;
      right: 12px;
      top: 8px;
      font-size: 11px;
      color: var(--text-muted);
    }
    .emoji-panel{
      position: absolute;
      top: calc(100% + 8px);
      left: 20px;
      right: 20px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-lg);
      padding: 12px;
      z-index: 200;
      max-height: 300px;
      overflow-y: auto;
    }
    .emoji-header{
      display: flex;
      align-items: center;
      padding: 0 8px 12px;
      border-bottom: 1px solid var(--border-light);
      margin-bottom: 12px;
    }
    .emoji-grid{
      display: grid;
      grid-template-columns: repeat(10, 1fr);
      gap: 4px;
      max-height: 200px;
      overflow-y: auto;
    }
    .emoji-item{
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      cursor: pointer;
      border-radius: 8px;
      transition: all var(--transition-fast);
    }
    .emoji-item:hover{
      background: var(--border-light);
      transform: scale(1.15);
    }

    .quick-prompts{
      padding: 12px 20px;
      background: var(--panel);
      border-top: 1px solid var(--border-light);
      animation: slideUp 0.2s ease-out;
    }
    @keyframes slideUp {
      from {
        opacity: 0;
        transform: translateY(8px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    .quick-prompts-header{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border-light);
    }
    .quick-prompts-list{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .quick-prompt{
      padding: 8px 14px;
      background: var(--bg-chat);
      border: 1px solid var(--border);
      border-radius: var(--radius-xl);
      font-size: 13px;
      color: var(--text-secondary);
      cursor: pointer;
      transition: all var(--transition-fast);
    }
    .quick-prompt:hover{
      background: rgba(59, 130, 246, 0.08);
      border-color: var(--primary);
      color: var(--primary);
    }

    .captcha-row{
      display:flex; gap: 10px; align-items:center;
    }
    .captcha-img{
      width: 100px;
      height: 40px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: var(--gray-50);
      cursor: pointer;
    }

    .loading-spinner{
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid var(--gray-200);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @media (max-width: 980px){
      .row{ flex-direction:column; }
      .left{ width: 100%; flex: 1 1 auto; }
    }
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

          <!-- 表情选择面板 -->
          <div class="emoji-panel" id="emojiPanel" style="display:none;">
            <div class="emoji-header">
              <span style="font-size:13px; font-weight:500; color:var(--text-primary);">选择表情</span>
              <button class="btn small outline" id="btnCloseEmoji" style="margin-left:auto;">关闭</button>
            </div>
            <div class="emoji-grid" id="emojiGrid">
            </div>
          </div>

          <!-- 隐藏的文件上传input -->
          <input type="file" id="fileInput" style="display:none;" accept="image/*,audio/*,video/*,.txt,.pdf,.doc,.docx">
          
          <!-- 语音录制按钮 -->
          <button class="composer-btn" id="btnRecord" title="语音录制" style="display:none;">🎤</button>
          
          <!-- 录制状态指示器 -->
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
      user: null, // { user_id, username }
      roles: [],
      role: null, // { id, name, description }
      messages: [], // { role: 'user'|'assistant', content: string }
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
      // Minimal toast (avoid dependencies)
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
      
      // 发送按钮始终显示"发送"
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
      // 先尝试解开 {"response":"..."} 这类后端字符串化 JSON
      try{
        if(t.startsWith('{') && t.endsWith('}')){
          const obj = JSON.parse(t);
          if(obj && typeof obj === 'object'){
            t = obj.response || obj.answer || obj.content || obj.text || t;
          }
        }
      }catch(e){}
      // 去除常见 markdown 符号（不展示 #、* 等）
      // 同时去掉形如 {"response":"..."} 的残留包裹
      const withoutMd = t
        .replace(/^\s*\{\s*"(response|answer|content|text)"\s*:\s*"/i, '')
        .replace(/"\s*\}\s*$/i, '')
        .replace(/^\s{0,3}#{1,6}\s+/gm, '')
        .replace(/^\s*[*-]\s+/gm, '')
        .replace(/\*\*/g, '')
        .replace(/\\"/g, '"')
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
        
        // 处理消息内容
        const content = (m.role === 'assistant') ? sanitizeDisplayText(m.content || '') : (m.content || '');
        
        // 检查是否包含图片引用
        if(m.role === 'user' && m.fileData && m.fileData.type && m.fileData.type.startsWith('image/')){
          // 创建图片元素
          const img = document.createElement('img');
          img.src = m.fileData.data;
          img.className = 'msg-image';
          img.alt = '图片预览';
          bubble.appendChild(img);
        }
        
        // 添加文字内容
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

    let healthCheckInterval = null;
    
    async function health(){
      // 如果正在处理消息，跳过健康检查以避免误报
      if(state.busy) return;
      
      try{
        const data = await api('/api/health', { method: 'GET', headers: {} });
        setBackendChip(true, '后端：正常');
      }catch(e){
        // 只有在非忙碌状态下才显示不可用
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
      
      // 设置停止标志，让流式读取停止
      state.stopped = true;
      
      // 如果有正在读取的流，取消它
      if(state.currentReader){
        try{
          await state.currentReader.cancel();
        }catch(e){
          console.log('取消读取流:', e);
        }
        state.currentReader = null;
      }
      
      // 重置状态
      state.busy = false;
      state.paused = false;
      state.stopped = false;
      state.pendingBuffer = '';
      
      // 更新UI
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
        if('。！？!?，,；;：:\\n'.includes(ch) || buf.length >= 6){
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
      
      // 如果有文件数据，添加到payload中
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
          // 检查是否被停止
          if(state.stopped){
            break;
          }
          
          const { value, done } = await reader.read();
          if(done) break;
          
          const chunk = decoder.decode(value, { stream: true });
          const smallParts = splitToSmallParts(chunk);
          for(const p of smallParts){
            // 每次处理前检查是否被停止
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
        // 如果被停止，取消读取并关闭流
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
      
      // 保存当前的文件数据并清空待发送状态
      const fileData = pendingFileData;
      pendingFileData = null;
      hideFilePreview();
      
      // 保存消息，包括文件数据（用于显示图片预览）
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
        // 恢复时把暂停期间积累的内容一次性追加上去
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
      
      // 快捷提问按钮
      $('btnQuick').addEventListener('click', () => {
        const quickPrompts = $('quickPrompts');
        quickPrompts.style.display = quickPrompts.style.display === 'none' ? 'block' : 'none';
      });
      
      // 关闭快捷提问
      $('btnCloseQuick').addEventListener('click', () => {
        $('quickPrompts').style.display = 'none';
      });
      
      // 快捷提问项点击
      document.querySelectorAll('.quick-prompt').forEach(el => {
        el.addEventListener('click', () => {
          $('msgInput').value = el.textContent;
          $('quickPrompts').style.display = 'none';
          sendMsg();
        });
      });
      
      // 表情按钮
      $('btnEmoji').addEventListener('click', () => {
        const emojiPanel = $('emojiPanel');
        const quickPrompts = $('quickPrompts');
        
        // 隐藏快捷提问面板
        quickPrompts.style.display = 'none';
        
        // 切换表情面板
        emojiPanel.style.display = emojiPanel.style.display === 'none' ? 'block' : 'none';
        
        // 如果显示表情面板，初始化表情
        if(emojiPanel.style.display === 'block'){
          initEmojiPanel();
        }
      });
      
      // 关闭表情面板
      $('btnCloseEmoji').addEventListener('click', () => {
        $('emojiPanel').style.display = 'none';
      });
      
      // 附件按钮
      $('btnAttach').addEventListener('click', () => {
        $('fileInput').click();
      });
      
      // 文件选择处理
      $('fileInput').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if(file){
          handleFileUpload(file);
        }
        // 重置input以便可以重新选择同一文件
        e.target.value = '';
      });
      
      // 移除文件按钮
      $('btnRemoveFile').addEventListener('click', () => {
        removePendingFile();
      });
      
      // 语音录制按钮
      $('btnRecord').addEventListener('click', () => {
        if(mediaRecorder && mediaRecorder.state === 'recording'){
          stopRecording();
        }else{
          startRecording();
        }
      });
      
      // 点击外部关闭面板
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
    
    // 表情列表
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
    
    // 初始化表情面板
    function initEmojiPanel(){
      const grid = $('emojiGrid');
      if(grid.children.length > 0) return; // 已经初始化
      
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
    
    // 在光标位置插入表情
    function insertEmoji(emoji){
      const textarea = $('msgInput');
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const value = textarea.value;
      
      textarea.value = value.substring(0, start) + emoji + value.substring(end);
      
      // 将光标移动到表情后面
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + emoji.length;
        textarea.focus();
      }, 0);
    }
    
    // 当前待发送的文件数据
    let pendingFileData = null;
    
    // 语音录制相关
    let mediaRecorder = null;
    let audioChunks = [];
    
    // 检查浏览器是否支持语音录制
    function checkMediaSupport(){
      if('MediaRecorder' in window && navigator.mediaDevices && navigator.mediaDevices.getUserMedia){
        $('btnRecord').style.display = 'grid';
        return true;
      }
      return false;
    }
    
    // 开始录音
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
    
    // 停止录音
    function stopRecording(){
      if(mediaRecorder && mediaRecorder.state !== 'inactive'){
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        $('btnRecord').textContent = '🎤';
        $('btnRecord').title = '语音录制';
        $('recordingIndicator').style.display = 'none';
      }
    }
    
    // 处理录制的音频
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
    
    // 处理文件上传
    function handleFileUpload(file){
      const maxSize = 50 * 1024 * 1024; // 50MB（视频文件可能较大）
      
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
        
        // 保存文件数据供发送时使用
        pendingFileData = {
          name: fileName,
          type: fileType,
          data: base64Data
        };
        
        // 显示文件预览
        showFilePreview(fileName, base64Data);
        
        // 根据文件类型自动填充提示文字
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
    
    // 显示文件预览
    function showFilePreview(fileName, base64Data){
      const preview = $('filePreview');
      const previewImg = $('previewImg');
      const previewName = $('previewName');
      
      previewName.textContent = fileName;
      
      // 如果是图片，显示预览
      if(base64Data.startsWith('data:image/')){
        previewImg.src = base64Data;
        previewImg.style.display = 'block';
      }else{
        // 如果不是图片，隐藏预览图
        previewImg.style.display = 'none';
      }
      
      preview.classList.add('show');
    }
    
    // 隐藏文件预览
    function hideFilePreview(){
      const preview = $('filePreview');
      preview.classList.remove('show');
      $('previewImg').src = '';
      $('previewName').textContent = '';
    }
    
    // 移除待发送的文件
    function removePendingFile(){
      pendingFileData = null;
      hideFilePreview();
      $('msgInput').value = '';
      toast('文件已移除');
    }

    async function main(){
      bind();
      setAuthPane('login');
      // 检查浏览器是否支持语音录制
      checkMediaSupport();
      // 强制每次新打开页面都重新登录，避免复用上一位用户身份
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


@app.get("/api/health")
def api_health() -> JSONResponse:
    try:
        resp = _proxy_get("/health", timeout=5)
        if resp.status_code == 200:
            return JSONResponse({"status": "ok"})
        raise HTTPException(status_code=502, detail="backend health not ok")
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")


@app.get("/api/roles")
def api_roles() -> JSONResponse:
    try:
        resp = _proxy_get("/roles", timeout=10)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"backend roles not ok: {resp.status_code}")
        data = resp.json() if resp.content else []
        roles: List[Dict[str, Any]] = []
        for r in (data or []):
            roles.append(
                {
                    "id": int(r.get("id")),
                    "name": r.get("role_name") or r.get("name") or f"role_{r.get('id')}",
                    "description": r.get("description") or "",
                }
            )
        if not roles:
            roles = [{"id": 1, "name": "financial_advisor", "description": "默认角色"}]
        return JSONResponse({"roles": roles})
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")


@app.post("/api/register")
def api_register(req: RegisterRequest) -> Response:
    try:
        resp = _proxy_post("/register", _model_to_dict(req), timeout=15)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.post("/api/login")
def api_login(req: LoginRequest) -> Response:
    try:
        resp = _proxy_post("/login", _model_to_dict(req), timeout=15)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.get("/api/chat_history/{user_id}/{role_id}")
def api_chat_history(user_id: int, role_id: int) -> Response:
    try:
        resp = _proxy_get(f"/chat_history/{user_id}/{role_id}", timeout=20)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.post("/api/clear_history/{user_id}/{role_id}")
def api_clear_history(user_id: int, role_id: int) -> Response:
    try:
        resp = _proxy_post(f"/clear_history/{user_id}/{role_id}", payload={}, timeout=20)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> Response:
    try:
        resp = _proxy_post("/chat", _model_to_dict(req), timeout=90)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="backend unreachable")
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.post("/api/chat_stream")
def api_chat_stream(req: ChatRequest) -> StreamingResponse:
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
            # 使用迭代器模式处理流式响应
            buffer = ""
            for chunk in upstream.iter_content(chunk_size=128, decode_unicode=True):
                if chunk:
                    buffer += chunk
                    # 按字符分割，确保每个yield都是有效的UTF-8字符
                    while len(buffer) > 0:
                        try:
                            # 尝试发送至少一个字符
                            yield buffer[0]
                            buffer = buffer[1:]
                        except Exception:
                            # 如果字符不完整，等待更多数据
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









