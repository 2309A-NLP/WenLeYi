# RAG 智能问答系统 2.0

基于检索增强生成技术的智能文档问答系统，支持 PDF 图片解析、Reranker 精排、Milvus 向量库、多轮对话。

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 放入文档
将 PDF 文件放入 `documents/` 目录。

### 3. 构建索引
```bash
python scripts/build_index.py
```

### 4. 启动服务
```bash
python scripts/app.py
```

### 5. 使用
- 浏览器打开 `http://localhost:5000`
- API: POST `http://localhost:5000/api/query`，body: `{"question": "你的问题"}`

## 配置说明
所有配置在 `.env` 文件中修改，详见 `.env.example`。

## 项目结构
- `core/` — 核心代码（配置、文档处理、向量存储、检索、LLM）
- `scripts/` — 启动和构建脚本
- `templates/` — 前端页面
- `documents/` — 存放文档
- `vector_store/` — 索引存放
- `uploads/` — 上传文件目录

## 功能特性
- PDF 文档解析 + 图片提取识别
- 混合检索（BM25 + Milvus 向量检索）
- Reranker 精排（bge-reranker-base）
- 多轮对话支持
- SSE 流式输出
- 日间/夜间模式
- 中英文切换
- 语音输入 + 图片上传
- 文件上传管理（增量索引）
