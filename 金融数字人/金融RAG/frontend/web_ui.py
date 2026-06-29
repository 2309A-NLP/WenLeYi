# -*- coding: utf-8 -*-
"""金融对话系统 - Web前端（金融风格界面）"""

import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

BACKEND_URL = f"http://127.0.0.1:{os.getenv('SERVER_PORT', 8000)}"
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 3000))

app = FastAPI(title="金融对话系统 - UI")

HTML_CONTENT = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>金融对话系统</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: -apple-system, "Microsoft YaHei", sans-serif;
        background: linear-gradient(135deg, #0a1628, #1a2744, #0d1b3e);
        min-height: 100vh;
        color: #e0e6f0;
    }
    .container { max-width: 900px; margin: 0 auto; padding: 20px; }
    .header {
        text-align: center; padding: 30px 0;
        border-bottom: 1px solid rgba(255,193,7,0.3); margin-bottom: 20px;
    }
    .header h1 {
        font-size: 28px; color: #ffc107;
        text-shadow: 0 0 20px rgba(255,193,7,0.3);
    }
    .header p { color: #8899bb; margin-top: 8px; font-size: 14px; }
    .header .status { display: inline-block; margin-top: 10px; font-size: 13px; }
    .status-dot {
        display: inline-block; width: 8px; height: 8px;
        border-radius: 50%; margin-right: 6px;
    }
    .status-dot.online { background: #4caf50; box-shadow: 0 0 8px #4caf50; }
    .status-dot.offline { background: #f44336; }

    .stats-bar {
        display: flex; gap: 12px; justify-content: center; margin: 15px 0 25px;
        flex-wrap: wrap;
    }
    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px; padding: 12px 20px; text-align: center;
        min-width: 120px;
    }
    .stat-card .num { font-size: 22px; font-weight: bold; color: #ffc107; }
    .stat-card .label { font-size: 12px; color: #8899bb; margin-top: 4px; }

    .chat-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; overflow: hidden;
    }
    .messages {
        height: 460px; overflow-y: auto; padding: 20px;
    }
    .messages::-webkit-scrollbar { width: 6px; }
    .messages::-webkit-scrollbar-thumb { background: #ffc107; border-radius: 3px; }
    .messages::-webkit-scrollbar-track { background: transparent; }

    .msg { margin-bottom: 16px; display: flex; }
    .msg.user { justify-content: flex-end; }
    .msg.assistant { justify-content: flex-start; }

    .msg .bubble {
        max-width: 75%; padding: 12px 16px;
        border-radius: 12px; line-height: 1.6; font-size: 14px;
    }
    .msg.user .bubble {
        background: linear-gradient(135deg, #ffc107, #ffb300);
        color: #1a1a2e; border-bottom-right-radius: 4px;
    }
    .msg.assistant .bubble {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-bottom-left-radius: 4px;
    }
    .msg .bubble .time {
        font-size: 11px; color: #667799; margin-top: 6px; text-align: right;
    }
    .msg.user .bubble .time { color: rgba(0,0,0,0.4); }

    .sources { font-size: 12px; margin-top: 8px; color: #8899bb; }
    .sources span { display: inline-block; background: rgba(255,193,7,0.1);
        padding: 2px 8px; border-radius: 4px; margin: 2px;
        border: 1px solid rgba(255,193,7,0.2); }

    .input-area {
        display: flex; padding: 16px; gap: 10px;
        border-top: 1px solid rgba(255,255,255,0.08);
        background: rgba(0,0,0,0.2);
    }
    .input-area input {
        flex: 1; padding: 12px 16px; border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px; background: rgba(255,255,255,0.05);
        color: #e0e6f0; font-size: 14px; outline: none;
    }
    .input-area input:focus { border-color: #ffc107; }
    .input-area input::placeholder { color: #556688; }
    .input-area button {
        padding: 12px 24px; border: none; border-radius: 10px;
        background: linear-gradient(135deg, #ffc107, #ff9800);
        color: #1a1a2e; font-weight: bold; cursor: pointer;
        transition: all 0.2s;
    }
    .input-area button:hover { transform: scale(1.03); }
    .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }

    .quick-questions {
        display: flex; gap: 8px; padding: 12px 16px;
        flex-wrap: wrap; border-top: 1px solid rgba(255,255,255,0.05);
    }
    .quick-questions button {
        padding: 6px 14px; border: 1px solid rgba(255,193,7,0.3);
        border-radius: 20px; background: transparent;
        color: #ffc107; font-size: 12px; cursor: pointer; transition: all 0.2s;
    }
    .quick-questions button:hover { background: rgba(255,193,7,0.1); }

    .loading-dots::after {
        content: '...'; animation: dots 1.5s steps(4, end) infinite;
    }
    @keyframes dots { 0%, 20% { content: '.'; } 40% { content: '..'; }
        60% { content: '...'; } 80%, 100% { content: ''; } }

    .disclaimer {
        text-align: center; font-size: 11px; color: #445566;
        padding: 15px; border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 20px;
    }
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>💰 金融对话系统</h1>
        <p>基于RAG的金融知识智能问答 · 专业 | 严谨 | 合规</p>
        <div class="status">
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText">检查连接中...</span>
        </div>
    </div>

    <div class="stats-bar" id="statsBar">
        <div class="stat-card"><div class="num">-</div><div class="label">知识条目</div></div>
        <div class="stat-card"><div class="num">-</div><div class="label">分类数量</div></div>
        <div class="stat-card"><div class="num">-</div><div class="label">服务状态</div></div>
    </div>

    <div class="chat-box">
        <div class="messages" id="messages"></div>
        <div class="quick-questions" id="quickQuestions"></div>
        <div class="input-area">
            <input type="text" id="input" placeholder="输入金融问题..." onkeydown="if(event.key==='Enter') send()">
            <button id="sendBtn" onclick="send()">发送</button>
        </div>
    </div>
    <div class="disclaimer">
        ⚠️ 本系统提供的信息仅供参考，不构成投资建议。投资有风险，决策需谨慎。
    </div>
</div>

<script>
const BACKEND = '""" + BACKEND_URL + """';
let loading = false;

// 快捷问题
const QUICK_QS = [
    '什么是PE（市盈率）？',
    'A股涨跌幅限制是多少？',
    '什么是注册制？',
    '基金有哪些分类？',
];

// 初始化
async function init() {
    try {
        const resp = await fetch(BACKEND + '/api/v1/health');
        const data = await resp.json();
        document.getElementById('statusDot').className = 'status-dot online';
        document.getElementById('statusText').textContent = '在线';
        document.querySelector('.stat-card:nth-child(1) .num').textContent = data.knowledge_items || '-';
        document.querySelector('.stat-card:nth-child(2) .num').textContent = (data.categories || []).length + ' 类';
        document.querySelector('.stat-card:nth-child(3) .num').textContent = '✅';
        addMsg('assistant', '您好！我是金融助手，可以为您解答股票、基金、债券、保险等金融问题。请直接提问！');
    } catch(e) {
        document.getElementById('statusDot').className = 'status-dot offline';
        document.getElementById('statusText').textContent = '离线';
        addMsg('assistant', '⚠️ 后端服务未连接，请确保服务已启动。');
    }

    // 快捷按钮
    const qc = document.getElementById('quickQuestions');
    QUICK_QS.forEach(q => {
        const btn = document.createElement('button');
        btn.textContent = q;
        btn.onclick = () => {
            document.getElementById('input').value = q;
            send();
        };
        qc.appendChild(btn);
    });
}

function addMsg(role, text, sources) {
    const m = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    const now = new Date();
    const timeStr = now.getHours().toString().padStart(2,'0') + ':' + now.getMinutes().toString().padStart(2,'0');

    let bubble = '<div class="bubble">' + text.replace(/\n/g, '<br>');
    if (sources && sources.length > 0) {
        bubble += '<div class="sources">📖 参考: ' + sources.map(s => '<span>' + s.category + '</span>').join(' ') + '</div>';
    }
    bubble += '<div class="time">' + timeStr + '</div></div>';
    div.innerHTML = bubble;
    m.appendChild(div);
    m.scrollTop = m.scrollHeight;
}

async function send() {
    const input = document.getElementById('input');
    const btn = document.getElementById('sendBtn');
    const text = input.value.trim();
    if (!text || loading) return;

    input.value = '';
    loading = true;
    btn.disabled = true;
    addMsg('user', text);

    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'msg assistant';
    loadingDiv.innerHTML = '<div class="bubble"><span class="loading-dots">思考中</span></div>';
    document.getElementById('messages').appendChild(loadingDiv);

    try {
        const resp = await fetch(BACKEND + '/api/v1/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: text,
                use_knowledge: true,
                user_id: 0,
            })
        });
        const data = await resp.json();
        loadingDiv.remove();
        addMsg('assistant', data.answer, data.sources || []);
    } catch(e) {
        loadingDiv.remove();
        addMsg('assistant', '请求失败，请检查网络或稍后重试。');
    }

    loading = false;
    btn.disabled = false;
    input.focus();
}

init();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTMLResponse(HTML_CONTENT)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_ui:app", host="0.0.0.0", port=FRONTEND_PORT)
