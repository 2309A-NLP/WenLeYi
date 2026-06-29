# 工单编号：人工智能NLP-Agent数字人项目-招股书数据问答智能体任务
"""
检索与生成模块 - 根据问题从Milvus检索相关文档片段，调用LLM生成答案
功能：
1. 向量化用户问题
2. 从Milvus检索最相关的文本块
3. 用bge-reranker对检索结果重排序
4. 拼接Prompt调用DeepSeek生成答案
"""
import requests
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from config import (
    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME,
    EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    SEARCH_TOP_K, FINAL_TOP_K,
    TEMPERATURE, MAX_TOKENS, REQUEST_TIMEOUT,
)

# ==================== 全局变量（惰性加载，避免重复初始化） ====================
_embedding_model = None    # embedding模型实例
_reranker_model = None     # reranker模型实例
_collection = None         # Milvus集合实例


def safe_print(msg):
    """安全打印，避免Windows GBK编码错误"""
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            print(str(msg).encode('gbk', errors='ignore').decode('gbk', errors='ignore'))
        except:
            pass


def get_embedding_model():
    """获取embedding模型（单例模式，只加载一次）"""
    global _embedding_model
    if _embedding_model is None:
        safe_print("正在加载embedding模型...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        safe_print("embedding模型加载完成")
    return _embedding_model


def get_reranker_model():
    """获取reranker模型（单例模式，只加载一次）"""
    global _reranker_model
    if _reranker_model is None:
        safe_print("正在加载reranker模型...")
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(RERANKER_MODEL_PATH)
        safe_print("reranker模型加载完成")
    return _reranker_model


def get_collection():
    """获取Milvus集合（单例模式，带断线重连逻辑）"""
    global _collection
    if _collection is None:
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            _collection = Collection(name=COLLECTION_NAME)
            _collection.load()
            safe_print(f"已加载Milvus集合: {COLLECTION_NAME}, 记录数: {_collection.num_entities}")
        except Exception as e:
            safe_print(f"[Milvus] 连接失败: {e}，尝试重连...")
            _collection = None
            try:
                connections.disconnect("default")
            except Exception:
                pass
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            _collection = Collection(name=COLLECTION_NAME)
            _collection.load()
            safe_print(f"重连成功，已加载Milvus集合: {COLLECTION_NAME}, 记录数: {_collection.num_entities}")
    return _collection


def _do_search(collection, question_embedding, matched_companies, mapping, search_params):
    """执行Milvus向量检索"""
    if matched_companies:
        target_files = [mapping[c] for c in matched_companies if c in mapping]
        safe_print(f"  识别到公司: {matched_companies}，限定搜索文件: {target_files}")
        file_filters = " or ".join([f'source_file == "{f}"' for f in target_files])
        return collection.search(
            data=question_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=min(SEARCH_TOP_K, 50),
            output_fields=["text", "source_file", "chunk_index"],
            expr=file_filters,
        )
    else:
        safe_print(f"  未识别到公司名，全量搜索")
        return collection.search(
            data=question_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=SEARCH_TOP_K,
            output_fields=["text", "source_file", "chunk_index"],
        )


def retrieve_candidates(question):
    """
    从Milvus检索与问题最相关的文本块
    策略：先识别问题中的公司名，只搜该公司的文档，避免跨公司混淆
    参数：
        question: 用户问题
    返回：候选文本块列表
    """
    global _collection
    from company_mapping import load_mapping, find_company_in_question

    # 向量化问题
    model = get_embedding_model()
    question_embedding = model.encode([question], normalize_embeddings=True)

    # 连接Milvus（失败时重置并重试一次）
    try:
        collection = get_collection()
    except Exception as e:
        safe_print(f"[Milvus] 获取集合失败: {e}，重置连接后重试...")
        _collection = None
        try:
            connections.disconnect("default")
        except Exception:
            pass
        collection = get_collection()

    # 识别问题中的公司名
    mapping = load_mapping()
    matched_companies = find_company_in_question(question, mapping) if mapping else []

    # 执行检索，失败时重连并重试一次
    search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
    try:
        results = _do_search(collection, question_embedding, matched_companies, mapping, search_params)
    except Exception as e:
        safe_print(f"[Milvus] 检索失败: {e}，重置连接后重试...")
        _collection = None
        try:
            connections.disconnect("default")
        except Exception:
            pass
        collection = get_collection()
        results = _do_search(collection, question_embedding, matched_companies, mapping, search_params)

    # 整理检索结果
    candidates = []
    for hits in results:
        for hit in hits:
            candidates.append({
                "text": hit.entity.get("text"),
                "source_file": hit.entity.get("source_file"),
                "chunk_index": hit.entity.get("chunk_index"),
                "score": hit.score,
            })
    safe_print(f"  检索到 {len(candidates)} 个候选文本块")
    return candidates


def rerank_candidates(question, candidates, top_k=FINAL_TOP_K):
    """
    用bge-reranker对候选结果重排序，提高检索精度
    参数：
        question: 用户问题
        candidates: 候选文本块列表
        top_k: 重排序后保留的数量
    返回：重排序后的文本块列表
    """
    if not candidates:
        return []

    reranker = get_reranker_model()
    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for i, c in enumerate(candidates):
        c["rerank_score"] = scores[i]
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    result = candidates[:top_k]
    safe_print(f"  重排序后保留 {len(result)} 个文本块")
    return result


def build_prompt(question, text_chunks):
    """
    构建送入LLM的Prompt，包含检索到的文本和约束规则
    """
    context_parts = []
    for i, chunk in enumerate(text_chunks, 1):
        source = chunk.get("source_file", "未知来源")
        context_parts.append(f"【片段{i}】(来源: {source})\n{chunk['text']}")
    context = "\n\n".join(context_parts)
    # 构建完整Prompt
    prompt = f"""你是一个招股书数据问答助手。请根据以下招股书内容回答问题。

重要规则：
1. 仔细阅读下面提供的招股书内容，答案一定在里面
2. 直接从内容中提取答案，不要说"未找到"
3. 回答要完整，包含所有相关信息，不要遗漏
4. 如果涉及多个部门、人员、数字等，全部列出
5. 不要只给关键词，要给出完整的句子

【招股书内容】
{context}

【问题】
{question}

【答案】"""
    return prompt


def call_llm(prompt, max_retries=1):
    """
    调用DeepSeek API生成答案，支持429限流自动重试
    """
    import time
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429:
                wait = 10 * (attempt + 1)
                safe_print(f"  [429限流] 等待{wait}秒后重试({attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"].strip()
            return answer
        except requests.exceptions.Timeout:
            safe_print("  [超时] API请求超时")
            time.sleep(5)
            continue
        except Exception as e:
            import traceback
            safe_print(f"  [错误] API调用失败: {e}")
            safe_print(f"  [错误详情] {traceback.format_exc()}")
            time.sleep(5)
            continue

    safe_print("  [失败] 重试次数耗尽")
    return "未找到相关信息"


def query_answer(question):
    """
    完整的问答流程：检索 → 重排序 → 生成答案
    """
    # 第一步：向量检索候选文本块
    candidates = retrieve_candidates(question)

    # 第二步：跳过Reranker，直接用向量检索结果
    top_chunks = candidates[:FINAL_TOP_K]
    safe_print(f"  直接使用向量检索结果（跳过Reranker）")

    # 打印检索到的内容（调试用）
    safe_print("\n" + "=" * 40)
    safe_print(f"问题: {question}")
    for i, chunk in enumerate(top_chunks, 1):
        safe_print(f"\n--- 候选片段{i} (来源: {chunk.get('source_file', '?')}) ---")
        try:
            safe_print(chunk['text'][:300])
        except:
            safe_print(str(chunk['text'][:300]).encode('gbk', errors='ignore').decode('gbk', errors='ignore'))
    safe_print("=" * 40)

    # 第三步：构建Prompt
    prompt = build_prompt(question, top_chunks)

    # 第四步：调用LLM生成答案
    answer = call_llm(prompt)
    # 去掉markdown格式符号
    answer = answer.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    # 去掉多余空行
    answer = "\n".join(line for line in answer.split("\n") if line.strip())
    return answer


if __name__ == "__main__":
    # 测试：手动输入问题进行问答
    safe_print("=" * 60)
    safe_print("招股书数据问答系统 - 交互测试")
    safe_print("=" * 60)
    while True:
        question = input("\n请输入问题（输入q退出）: ").strip()
        if question.lower() == "q":
            break
        if not question:
            continue
        safe_print("正在检索和生成答案...")
        answer = query_answer(question)
        safe_print(f"\n答案: {answer}")
