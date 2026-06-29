# 金融数字人对话系统

基于 Linly-Talker + 自研金融RAG 的智能数字人问答系统。

## 系统架构

用户提问 → Whisper语音识别 → 金融RAG知识库检索 → DeepSeek LLM生成回答 → PaddleTTS语音合成 → SadTalker数字人视频

## 主要功能

1. 语音交互：Whisper ASR 识别用户语音输入
2. 知识检索：RAG系统从金融知识库中检索相关答案
3. 智能回答：DeepSeek LLM 基于检索结果生成专业回答
4. 语音合成：PaddleTTS 将文本转为自然语音
5. 数字人驱动：SadTalker 生成口型同步的数字人视频

## 项目结构

```
金融数字人/
├── webui.py              # 主入口（已集成金融RAG）
├── configs.py            # 配置文件
├── ASR/                  # 语音识别模块
├── LLM/                  # 大语言模型模块
├── TTS/                  # 语音合成模块
├── src/                  # 核心源码
├── 金融RAG/              # 自研金融RAG系统
│   ├── backend/          # FastAPI后端
│   ├── frontend/         # 前端页面
│   ├── 数据/             # 金融知识库数据
│   ├── requirements.txt  # 依赖列表
│   └── start.py          # 启动脚本
└── checkpoints/          # 模型文件（需单独下载）
```

## 快速开始

### 1. 环境要求
- Python 3.10+
- CUDA 11.8+
- AutoDL云服务器（推荐4090D显卡）

### 2. 安装依赖
```bash
pip install -r requirements_webui.txt
pip install -r 金融RAG/requirements.txt
```

### 3. 下载模型
```bash
python -c "from modelscope import snapshot_download; snapshot_download('Kedreamix/Linly-Talker', local_dir='checkpoints')"
```

### 4. 启动服务
```bash
# 窗口1：启动金融RAG后端
cd 金融RAG/backend && python main.py

# 窗口2：启动数字人WebUI
python webui.py
```

### 5. 访问
通过SSH端口转发访问：
```bash
ssh -L 6006:localhost:6006 -p <端口> root@<服务器>
# 浏览器打开 http://localhost:6006
```

## 技术栈
- Linly-Talker：数字人视频生成框架
- Whisper：语音识别
- RAG：检索增强生成
- DeepSeek：大语言模型
- PaddleTTS：语音合成
- SadTalker：面部动画驱动
- Gradio：Web界面

## 注意事项
1. 模型文件（checkpoints目录）需单独下载，不包含在本仓库中
2. 金融RAG的知识库数据在 金融RAG/数据/ 目录
3. .env 文件需自行配置，包含 DEEPSEEK_API_KEY
4. AutoDL部署详见 AutoDL部署.md
