FROM python:3.9-slim

LABEL maintainer="gongdan10-rag"
LABEL description="RAG金融问答系统"

WORKDIR /app

# 配置 pip 使用阿里云镜像
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 CPU 版 torch（pytorch 官方源，加 extra-index-url 不覆盖阿里云）
COPY requirements.txt .
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

# 安装其余依赖（用阿里云，加 --no-deps 避免 torch 依赖冲突）
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY core/ ./core/
COPY scripts/ ./scripts/
COPY templates/ ./templates/

# 创建数据目录
RUN mkdir -p documents uploads vector_store

EXPOSE 5000

CMD ["python", "scripts/app.py"]
