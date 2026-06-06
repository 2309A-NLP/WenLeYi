# RAG金融问答系统 Docker 部署文档

**工单编号**：人工智能NLP-RAG项目-金融问答系统部署

---

## 一、环境要求

- Docker Desktop（已安装并运行）
- Docker Compose（Docker Desktop 自带）
- 磁盘空间：至少 10GB（用于镜像和数据卷）
- 模型文件：`D:\桌面\模型\m3e-base` 和 `D:\桌面\模型\bge-reranker-base`

---

## 二、部署步骤

### 1. 进入项目目录

```bash
cd D:\桌面\gongdanzuoye\gongdan10
```

### 2. 构建镜像

```bash
docker compose build
```

首次构建约需 5-10 分钟，后续增量构建更快。

### 3. 启动所有服务

```bash
docker compose up -d
```

### 4. 检查服务状态

```bash
docker compose ps
```

正常状态：
- `gongdan10-app` → running
- `gongdan10-milvus` → running (healthy)
- `gongdan10-redis` → running

### 5. 查看应用日志

```bash
docker compose logs -f rag-app
```

看到 `Running on http://0.0.0.0:5000` 表示启动成功。

### 6. 首次构建索引

```bash
docker compose exec rag-app python scripts/build_index.py
```

### 7. 访问服务

浏览器打开：**http://localhost:5000**

---

## 三、常用运维命令

```bash
# 停止服务
docker compose down

# 重启服务
docker compose restart

# 查看实时日志
docker compose logs -f

# 进入容器内部调试
docker compose exec rag-app bash

# 重建后重启（代码有更新时）
docker compose up -d --build
```

---

## 四、数据持久化说明

| 数据类型 | 宿主机目录 | 容器内路径 | 说明 |
|---------|-----------|-----------|------|
| 文档文件 | ./documents | /app/documents | PDF等原始文档 |
| 上传文件 | ./uploads | /app/uploads | 用户上传的文件 |
| Milvus数据 | Docker卷 milvus-data | /var/lib/milvus | 向量数据库持久化 |
| Redis数据 | Docker卷 redis-data | /data | 缓存持久化 |

---

## 五、验证部署成功

### 验收标准1：容器启动与运行

```bash
# 检查容器状态
docker compose ps

# 检查无异常日志
docker compose logs rag-app 2>&1 | grep -i error
```

### 验收标准2：网络配置

```bash
# 检查容器间通信
docker compose exec rag-app ping milvus
docker compose exec rag-app ping redis
```

---

## 六、注意事项

1. **模型文件**：需确保 `D:\桌面\模型\m3e-base` 和 `D:\桌面\模型\bge-reranker-base` 目录存在
2. **端口冲突**：如5000端口被占用，可在 docker-compose.yml 中修改
3. **首次拉取镜像**：Milvus镜像较大（约1GB），首次需等待下载
4. **API Key**：`.env.docker` 中的 `LLM_API_KEY` 和 `VISION_API_KEY` 需与 `.env` 保持一致
5. **Neo4j**：当前配置已关闭（ENABLE_NEO4J=0），如需启用请修改 .env.docker

---

## 七、工单注释

> 注释需包括工单编号：人工智能NLP-RAG-金融问答系统部署
