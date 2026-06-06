"""检索策略 -- BM25 / 向量 / 混合（RRF 融合）+ 并行检索。

本模块提供三种检索方式:
  1. BM25 基于词频的经典信息检索算法
  2. 向量语义检索（Milvus / FAISS）
  3. Neo4j 知识图谱检索

HybridRetriever 将以上方式通过加权融合并行执行，实现混合检索。
"""

import math
import jieba
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from collections import Counter


def normalize_scores(results: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
    """Min-Max 归一化：将分数缩放到 [0, 1] 区间。

    Args:
        results: [(chunk, score), ...] 原始分数列表

    Returns:
        归一化后的分数列表，若列表为空或所有分数相同则返回原列表。
    """
    if not results:
        return results
    scores = [s for _, s in results]
    min_s = min(scores)
    max_s = max(scores)
    # 如果所有分数相同（max == min），避免除零，返回原列表
    if abs(max_s - min_s) < 1e-9:
        return results
    normalized = [(chunk, (score - min_s) / (max_s - min_s)) for chunk, score in results]
    return normalized


class BM25:
    """简易 BM25 检索器。

    BM25 (Best Matching 25) 是经典的概率信息检索模型，
    基于词频 (TF) 和逆文档频率 (IDF) 计算查询与文档的相关性分数。
    """

    def __init__(self, chunks: List[Dict], k1: float = 1.5, b: float = 0.75):
        """初始化 BM25 检索器，对所有文档进行分词和统计。

        Args:
            chunks: 文档块列表，每个块包含 'text' 字段
            k1: 词频饱和参数，控制词频增长对分数的影响 (默认 1.5)
            b: 文档长度归一化参数，0 表示不归一化，1 表示完全归一化 (默认 0.75)
        """
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.doc_count = len(chunks)
        self.avg_dl = 0
        self.doc_lens = []    # 每篇文档的 token 长度
        self.tf = []          # 每篇文档的词频字典
        self.df = Counter()   # 每个词出现在多少篇文档中（文档频率）

        # 对每个文档进行 jieba 分词，统计词频和文档频率
        for chunk in chunks:
            tokens = list(jieba.cut(chunk["text"]))
            self.doc_lens.append(len(tokens))
            self.avg_dl += len(tokens)
            tf = Counter(tokens)
            self.tf.append(tf)
            for token in set(tokens):
                self.df[token] += 1

        # 计算平均文档长度
        self.avg_dl = self.avg_dl / max(self.doc_count, 1)

        # [RETRIEVER] BM25 初始化完成日志
        print(f"[RETRIEVER] BM25 初始化完成: 文档块数量={self.doc_count}, "
              f"平均文档长度(token)={self.avg_dl:.2f}")

    def _idf(self, token: str) -> float:
        """计算逆文档频率 (IDF)。

        IDF 公式: log((N - n + 0.5) / (n + 0.5) + 1)
        其中 N 为总文档数，n 为包含该词的文档数。
        """
        n = self.df.get(token, 0)
        return math.log((self.doc_count - n + 0.5) / (n + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """BM25 检索：对查询分词后计算每个文档的相关性分数并排序。

        Args:
            query: 用户查询字符串
            top_k: 返回前 k 个最相关的结果

        Returns:
            [(chunk, score), ...] 按分数降序排列的结果列表
        """
        tokens = list(jieba.cut(query))

        # [RETRIEVER] BM25 搜索日志
        print(f"[RETRIEVER] BM25 搜索: 查询=\"{query[:50]}{'...' if len(query) > 50 else ''}\", "
              f"分词token数={len(tokens)}, "
              f"tokens={tokens[:10]}{'...' if len(tokens) > 10 else ''}, "
              f"top_k={top_k}, 候选文档数={self.doc_count}")

        scores = []
        for i, chunk in enumerate(self.chunks):
            score = 0
            dl = self.doc_lens[i]
            # 对查询中的每个 token 累加 BM25 分数
            for token in tokens:
                tf = self.tf[i].get(token, 0)
                idf = self._idf(token)
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                )
            scores.append((chunk, score))

        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]

        # [RETRIEVER] BM25 搜索结果日志
        print(f"[RETRIEVER] BM25 搜索完成: 返回结果数={len(top_results)}, "
              f"最高分数={top_results[0][1]:.4f}" if top_results else
              "[RETRIEVER] BM25 搜索完成: 无匹配结果")

        return top_results


class HybridRetriever:
    """混合检索器：向量 + BM25 + 图谱，加权融合，支持并行执行。

    支持三种检索模式:
      - 'vector': 仅向量语义检索
      - 'bm25':   仅 BM25 关键词检索
      - 其他:     混合加权融合检索（默认模式）
    """

    def __init__(self, vector_store, chunks: List[Dict], config, graph_store=None):
        """初始化混合检索器。

        Args:
            vector_store: 向量存储实例（如 MilvusVectorStore / FaissVectorStore）
            chunks: 文档块列表，用于构建 BM25 索引
            config: 配置对象，包含搜索模式、权重、top_k 等参数
            graph_store: Neo4j 图谱存储实例（可选），为 None 时不启用图谱检索
        """
        self.vector_store = vector_store
        self.bm25 = BM25(chunks) if chunks else None
        self.graph_store = graph_store
        self.config = config
        self._executor = None
        self._executor_workers = max(1, int(getattr(config, "RETRIEVER_WORKERS", 3) or 3))

        # [RETRIEVER] HybridRetriever 初始化完成日志
        bm25_ready = "就绪" if self.bm25 else "未就绪(无文档块)"
        graph_enabled = "已启用" if self.graph_store else "未启用"
        print(f"[RETRIEVER] HybridRetriever 初始化完成: "
              f"BM25={bm25_ready}, 图谱检索={graph_enabled}, "
              f"向量存储={'已加载' if self.vector_store else '未加载'}")

    def _get_executor(self) -> ThreadPoolExecutor:
        """Lazily reuse retrieval workers across queries."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._executor_workers,
                thread_name_prefix="retriever",
            )
        return self._executor

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        """根据配置选择检索模式并执行检索。

        Args:
            query: 用户查询字符串
            top_k: 返回结果数量，为 None 时使用配置中的默认值

        Returns:
            [(chunk, score), ...] 检索结果列表
        """
        top_k = top_k or self.config.TOP_K
        mode = self.config.SEARCH_MODE

        # [RETRIEVER] 检索开始日志
        print(f"[RETRIEVER] 检索开始: 模式={mode}, top_k={top_k}, "
              f"查询=\"{query[:50]}{'...' if len(query) > 50 else ''}\"")

        if mode == "bm25":
            results = self._bm25_search(query, top_k)
        elif mode == "vector":
            results = self._vector_search(query, top_k)
        else:
            results = self._weighted_search(query, top_k)

        print(f"[RETRIEVER] 检索结束: 模式={mode}, 返回结果数={len(results)}")
        return results

    def _vector_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """向量语义检索：通过向量相似度匹配最相关的文档块。"""
        return self.vector_store.search(query, top_k)

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """BM25 关键词检索：通过词频匹配最相关的文档块。"""
        if self.bm25 is None:
            print("[RETRIEVER] BM25 检索跳过: BM25 索引未初始化(无文档块)")
            return []
        return self.bm25.search(query, top_k)

    def _graph_search(self, query: str, limit: int = 5, multihop: bool = True) -> List[Tuple[Dict, float]]:
        """Neo4j 图谱检索：搜索知识图谱中的相关实体关系三元组。

        Args:
            query: 查询字符串
            limit: 返回的最大实体数量
            multihop: 是否启用多跳检索（默认 True）

        Returns:
            [(chunk, score), ...] 图谱检索结果，score 基于跳数衰减
        """
        if not self.graph_store:
            return []
        try:
            if multihop and hasattr(self.graph_store, "search_entities_multihop"):
                entities = self.graph_store.search_entities_multihop(query, limit=limit, hops=2)
            else:
                entities = self.graph_store.search_entities(query, limit)

            results = []
            for i, e in enumerate(entities):
                # 将三元组组合为可读文本
                path_info = e.get("path", "")
                if path_info:
                    # 多跳路径，包含完整路径信息
                    text = f"{path_info} [证据: {e.get('evidence', '')[:200]}]"
                else:
                    evidence = e.get("evidence", "")
                    text = f"{e.get('source', '')} {e.get('relation', '')} {e.get('target', '')}"
                    if evidence:
                        text += f" [证据: {evidence[:200]}]"
                if text.strip():
                    # 第一跳 score 高，第二跳 score 低（衰减）
                    score = 0.5 if not path_info else 0.3
                    results.append(({"text": text, "source": "neo4j", "type": "graph"}, score))
            return results
        except Exception as exc:
            print(f"[RETRIEVER] 图谱检索异常: {exc}")
            return []

    def _weighted_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """加权融合检索 -- BM25 + 向量 + 图谱 并行执行，加权融合。

        流程:
          1. 通过线程池并行执行三种检索
          2. 对各路结果进行 Min-Max 归一化
          3. 按权重加权融合分数
          4. 去重并按融合分数排序返回 top_k 结果

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            [(chunk, score), ...] 融合后的检索结果
        """
        # 读取各检索器的权重配置
        bm25_weight = getattr(self.config, 'BM25_WEIGHT', 0.4)
        vec_weight = getattr(self.config, 'VEC_WEIGHT', 0.6)
        graph_weight = getattr(self.config, 'GRAPH_WEIGHT', 0.3)
        faiss_k = self.config.FAISS_TOP_K
        bm25_k = self.config.BM25_TOP_K
        graph_k = getattr(self.config, 'NEO4J_TOP_K', 5)

        vec_results = []
        bm25_results = []
        graph_results = []

        # --- 并行执行三种检索 ---
        print(f"[RETRIEVER] 加权检索启动并行任务: vec_top={faiss_k}, "
              f"bm25_top={bm25_k}, graph_top={graph_k}")

        executor = self._get_executor()
        futures = {
            executor.submit(self.vector_store.search, query, faiss_k): "vec",
        }
        if self.bm25:
            futures[executor.submit(self.bm25.search, query, bm25_k)] = "bm25"
        if self.graph_store:
            futures[executor.submit(self._graph_search, query, graph_k)] = "graph"

        for future in as_completed(futures):
            tag = futures[future]
            try:
                result = future.result()
                if tag == "vec":
                    vec_results = result
                elif tag == "bm25":
                    bm25_results = result
                else:
                    graph_results = result
            except Exception as exc:
                print(f"[RETRIEVER] 并行检索任务异常 [{tag}]: {exc}")

        # [RETRIEVER] 并行检索完成，各路结果数量
        print(f"[RETRIEVER] 并行检索完成: 向量结果={len(vec_results)}条, "
              f"BM25结果={len(bm25_results)}条, "
              f"图谱结果={len(graph_results)}条")

        # --- 分数归一化 ---
        vec_norm = normalize_scores(vec_results)
        bm25_norm = normalize_scores(bm25_results)
        graph_norm = normalize_scores(graph_results)

        # --- 加权融合 ---
        # 使用多维度去重键：文本前 150 字符 + 来源类型
        all_scores = {}
        all_chunks_map = {}
        all_sources = {}  # 记录每个 key 的来源类型

        def _merge_chunk(key, chunk, score, weight, source_type):
            """合并检索结果，累加加权分数，记录来源类型。"""
            if key in all_scores:
                all_scores[key] += score * weight
                # 保留分数更高的来源信息
                if source_type not in all_sources.get(key, ""):
                    all_sources[key] = all_sources.get(key, "") + f"+{source_type}"
            else:
                all_scores[key] = score * weight
                all_sources[key] = source_type
            all_chunks_map[key] = chunk

        for chunk, score in bm25_norm:
            key = chunk["text"][:150]
            _merge_chunk(key, chunk, score, bm25_weight, "bm25")

        for chunk, score in vec_norm:
            key = chunk["text"][:150]
            _merge_chunk(key, chunk, score, vec_weight, "vec")

        for chunk, score in graph_norm:
            key = chunk["text"][:150]
            _merge_chunk(key, chunk, score, graph_weight, "graph")

        # 按融合分数降序排序，取 top_k
        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        final_results = [(all_chunks_map[key], score) for key, score in ranked if key in all_chunks_map]

        # 统计各来源的贡献
        source_counts = {"bm25": 0, "vec": 0, "graph": 0, "multi": 0}
        for key, sources in all_sources.items():
            if "+" in sources:
                source_counts["multi"] += 1
            elif "bm25" in sources:
                source_counts["bm25"] += 1
            elif "vec" in sources:
                source_counts["vec"] += 1
            elif "graph" in sources:
                source_counts["graph"] += 1

        # [RETRIEVER] 融合结果日志
        print(f"[RETRIEVER] 加权融合完成: 去重后候选数={len(all_scores)}, "
              f"最终返回数={len(final_results)}, "
              f"权重: vec={vec_weight}, bm25={bm25_weight}, graph={graph_weight}, "
              f"来源分布: bm25={source_counts['bm25']}, vec={source_counts['vec']}, "
              f"graph={source_counts['graph']}, 多源融合={source_counts['multi']}")

        return final_results
