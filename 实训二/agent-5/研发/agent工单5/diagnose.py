# 工单编号：人工智能NLP-Agent数字人项目-招股书数据问答智能体任务
"""
诊断脚本 - 逐步排查检索问题
对指定问题，打印每个环节的详细信息，定位检索失败的原因
"""
from config import (
    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME,
    EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH,
    SEARCH_TOP_K, FINAL_TOP_K,
)
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer


def diagnose(question):
    """对一个问题进行全流程诊断"""
    print("=" * 60)
    print(f"问题: {question}")
    print("=" * 60)

    # 第一步：加载模型
    print("\n[1/5] 加载模型...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    print(f"  embedding模型: {EMBEDDING_MODEL_PATH}")

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(RERANKER_MODEL_PATH)
    print(f"  reranker模型: {RERANKER_MODEL_PATH}")

    # 第二步：连接Milvus，检查数据量
    print("\n[2/5] 连接Milvus...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(name=COLLECTION_NAME)
    collection.load()
    print(f"  集合: {COLLECTION_NAME}")
    print(f"  总记录数: {collection.num_entities}")

    # 第三步：向量检索，打印所有候选
    print(f"\n[3/5] 向量检索 top-{SEARCH_TOP_K}...")
    question_vec = embedding_model.encode([question], normalize_embeddings=True)
    search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
    results = collection.search(
        data=question_vec.tolist(),
        anns_field="embedding",
        param=search_params,
        limit=SEARCH_TOP_K,
        output_fields=["text", "source_file", "chunk_index"],
    )

    candidates = []
    for hits in results:
        for rank, hit in enumerate(hits):
            text = hit.entity.get("text")
            source = hit.entity.get("source_file")
            score = hit.score
            candidates.append({
                "text": text,
                "source_file": source,
                "score": score,
            })
            # 打印每个候选的摘要
            preview = text[:150].replace("\n", " ")
            print(f"  #{rank+1} 分数={score:.4f} 来源={source}")
            print(f"       内容: {preview}...")

    # 第四步：Reranker重排序
    print(f"\n[4/5] Reranker重排序，取 top-{FINAL_TOP_K}...")
    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = scores[i]
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    top_chunks = candidates[:FINAL_TOP_K]
    for rank, chunk in enumerate(top_chunks, 1):
        preview = chunk["text"][:200].replace("\n", " ")
        print(f"  #{rank} rerank分数={chunk['rerank_score']:.4f} 来源={chunk['source_file']}")
        print(f"       内容: {preview}...")

    # 第五步：检查是否包含关键词
    print(f"\n[5/5] 关键词匹配检查...")
    keywords = ["研发", "部门", "中心", "负责"]
    for kw in keywords:
        found_in = []
        for i, c in enumerate(candidates):
            if kw in c["text"]:
                found_in.append(f"#{i+1}")
        if found_in:
            print(f"  关键词'{kw}'出现在候选: {', '.join(found_in)}")
        else:
            print(f"  关键词'{kw}'在所有候选中均未出现！")

    # 检查top-50中有没有包含答案的
    print(f"\n  检查全部{len(candidates)}个候选中是否包含'研发'...")
    for i, c in enumerate(candidates):
        if "研发" in c["text"]:
            print(f"  ✓ 候选#{i+1}包含'研发'，分数={c['score']:.4f}，rerank分数={c.get('rerank_score', 'N/A')}")
            preview = c["text"][:300].replace("\n", " ")
            print(f"    内容: {preview}...")


if __name__ == "__main__":
    # 测试问题
    test_questions = [
        "云南沃森生物技术股份有限公司负责产品研发的是什么部门？",
        "湖南长远锂科股份有限公司变更设立时作为发起人的法人有哪些？",
    ]
    for q in test_questions:
        diagnose(q)
        print("\n\n")
