# 工单编号：人工智能NLP-Agent数字人项目-招股书数据问答智能体任务
"""
索引构建模块 - 读取招股书TXT文件，分块后向量化存入Milvus
功能：
1. 读取80份招股书TXT文件
2. 按固定长度分块（带重叠）
3. 用m3e-base模型向量化
4. 存入Milvus向量数据库
"""
import os
import json
import pickle
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from sentence_transformers import SentenceTransformer
from config import (
    TXT_DIR, DATA_DIR, COLLECTION_NAME,
    MILVUS_HOST, MILVUS_PORT,
    EMBEDDING_MODEL_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP,
)


def read_txt_files(txt_dir):
    """
    读取目录下所有TXT文件，返回文件名和内容的列表
    返回：[(文件名, 文件内容), ...]
    """
    documents = []
    # 遍历所有txt文件
    for filename in sorted(os.listdir(txt_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(txt_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append((filename, content))
            print(f"  已读取: {filename} ({len(content)} 字符)")
    print(f"共读取 {len(documents)} 个TXT文件")
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    将文本按固定长度分块，相邻块之间有重叠部分
    参数：
        text: 原始文本
        chunk_size: 每块的字符数
        overlap: 相邻块重叠的字符数
    返回：文本块列表
    """
    chunks = []
    # 清理文本中的多余空白和换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去除连续空行
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # 跳过空块
            chunks.append(chunk)
        # 下一块的起始位置 = 当前起始位置 + 块大小 - 重叠大小
        start += chunk_size - overlap
    return chunks


def build_chunks(documents):
    """
    对所有文档进行分块，记录每块的来源信息
    参数：
        documents: [(文件名, 文件内容), ...]
    返回：分块列表，每块包含文本和来源信息
    """
    all_chunks = []
    for filename, content in documents:
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,              # 文本块内容
                "source_file": filename,     # 来源文件名
                "chunk_index": i,            # 在该文件中的第几块
                "total_chunks": len(chunks), # 该文件总块数
            })
    print(f"共生成 {len(all_chunks)} 个文本块")
    return all_chunks


def connect_milvus():
    """连接Milvus向量数据库"""
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"已连接Milvus: {MILVUS_HOST}:{MILVUS_PORT}")


def create_collection(dim):
    """
    在Milvus中创建集合（如果已存在则先删除重建）
    参数：
        dim: 向量维度（m3e-base输出768维）
    返回：Milvus Collection对象
    """
    # 如果集合已存在，先删除
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"已删除旧集合: {COLLECTION_NAME}")

    # 定义字段 schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),      # 文本块内容
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256), # 来源文件
        FieldSchema(name="chunk_index", dtype=DataType.INT64),                   # 块索引
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),      # 向量
    ]

    # 创建集合
    schema = CollectionSchema(fields=fields, description="招股书文本块向量索引")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"已创建集合: {COLLECTION_NAME}, 向量维度: {dim}")
    return collection


def build_index():
    """主函数：构建Milvus向量索引"""
    print("=" * 60)
    print("开始构建招股书向量索引")
    print("=" * 60)

    # 第一步：读取所有TXT文件
    print("\n[1/5] 读取TXT文件...")
    documents = read_txt_files(TXT_DIR)

    # 第二步：文本分块
    print("\n[2/5] 文本分块...")
    all_chunks = build_chunks(documents)

    # 保存分块数据到本地（用于后续检索时获取原始文本）
    os.makedirs(DATA_DIR, exist_ok=True)
    chunks_path = os.path.join(DATA_DIR, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"分块数据已保存: {chunks_path}")

    # 第三步：加载embedding模型
    print("\n[3/5] 加载embedding模型...")
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    print(f"模型加载完成: {EMBEDDING_MODEL_PATH}")

    # 第四步：向量化所有文本块
    print("\n[4/5] 向量化文本块...")
    texts = [chunk["text"] for chunk in all_chunks]
    # 批量编码，提高效率
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    print(f"向量化完成，共 {len(embeddings)} 条，维度: {embeddings.shape[1]}")

    # 第五步：存入Milvus
    print("\n[5/5] 存入Milvus...")
    connect_milvus()
    collection = create_collection(dim=embeddings.shape[1])

    # 分批插入，避免超过gRPC 64MB消息大小限制
    batch_size = 5000
    total = len(all_chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_data = [
            [chunk["text"] for chunk in all_chunks[start:end]],
            [chunk["source_file"] for chunk in all_chunks[start:end]],
            [chunk["chunk_index"] for chunk in all_chunks[start:end]],
            embeddings[start:end].tolist(),
        ]
        collection.insert(batch_data)
        print(f"  已插入 {end}/{total} 条")
    # 刷新数据确保写入生效
    collection.flush()
    print(f"共插入 {collection.num_entities} 条记录")

    # 创建向量索引（加速检索）
    index_params = {
        "metric_type": "IP",        # 内积相似度（因为向量已归一化，等价于余弦相似度）
        "index_type": "IVF_FLAT",   # IVF倒排索引
        "params": {"nlist": 128},   # 聚类中心数
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("向量索引创建完成")

    # 加载集合到内存（用于检索）
    collection.load()
    print("\n" + "=" * 60)
    print("索引构建完成！")
    print(f"  - 文本块总数: {collection.num_entities}")
    print(f"  - Milvus集合: {COLLECTION_NAME}")
    print(f"  - 分块数据: {chunks_path}")
    print("=" * 60)


if __name__ == "__main__":
    build_index()
