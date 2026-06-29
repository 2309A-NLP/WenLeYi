# 智能体调度层（工单6）

## 工单编号
人工智能NLP-Agent数字人项目-智能体任务

## 项目简介
智能体调度层，集成5个工具，根据用户Query自主选择并调用对应工具提供服务。

## 集成工具
| 工具 | 端口 | 功能 |
|------|------|------|
| 记账本 | 5001 | 收支记录、查询、统计 |
| 日程提醒 | 5002 | 日程管理、定时提醒 |
| 文生图 | 5003 | 文本生成图片 |
| 基金数据问答 | 5004 | 基金信息查询、分析 |
| 招股书数据问答 | 5005 | 招股书内容查询 |

## 快速开始

### 1. 安装依赖
```bash
cd agent工单6
pip install -r requirements.txt
```

### 2. 配置环境变量
编辑.env文件，填入API密钥：
```
LLM_API_KEY=你的API密钥
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

### 3. 启动服务
```bash
# 先启动5个工具服务（每个单独终端）
cd ../agent工单1 && python app.py  # 端口5001
cd ../agent工单2 && python app.py  # 端口5002
cd ../agent工单3 && python app.py  # 端口5003
cd ../agent工单4 && python web.py  # 端口5004
cd ../agent工单5 && python web.py  # 端口5005

# 启动Agent调度层
cd ../agent工单6 && python app.py  # 端口5060
```

### 4. 访问界面
浏览器打开 http://localhost:5060

## 功能特性
- 自然语言交互（文字+语音）
- 智能意图识别（5种工具自动路由）
- 多轮对话支持
- 图片预览和下载
- 数据图表可视化
- 工具状态监控
- 错误引导和提示

## API接口

### 发送消息
```
POST /api/chat
Body: {"message": "今天午饭花了35块", "session_id": "user1"}
Response: {"code": 0, "data": {"reply": "已记录...", "tool_used": "accounting"}}
```

### 健康检查
```
GET /api/health
Response: {"code": 0, "data": {"status": "ok", "tools": {...}}}
```

## 测试
```bash
python test_accuracy.py
```

## 文件结构
```
agent工单6/
├── config.py          # 配置文件
├── orchestrator.py    # 核心调度器
├── tool_client.py     # 工具HTTP客户端
├── app.py             # Flask Web服务
├── async_tasks.py     # 异步任务管理
├── templates/
│   └── index.html     # 前端界面
├── test_accuracy.py   # 准确率测试
├── requirements.txt   # 依赖列表
├── .env               # 环境变量
├── 设计文档.md         # 设计文档
├── 过程记录.md         # 开发过程记录
└── README.md          # 本文件
```
