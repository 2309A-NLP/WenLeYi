"""Reranker 精排模块 -- 使用本地 cross-encoder 模型对检索结果进行重排序。"""
import time
from typing import List, Dict, Tuple


class Reranker:
    """基于 cross-encoder 的重排序器。

    功能说明:
        1. 接收查询(query)和初步检索结果(results)
        2. 使用 cross-encoder 模型对每一对 (query, document) 进行相关性打分
        3. 按分数降序排列, 返回 top_k 个最相关的结果
    """

    def __init__(self, model_path: str):
        """初始化重排序器。

        Args:
            model_path: cross-encoder 模型的路径(本地路径或 HuggingFace 模型名)
        """
        self.model_path = model_path
        self._model = None
        # [RERANKER] 初始化时打印模型路径
        print(f"[RERANKER] 初始化重排序器, model_path={self.model_path}")

    def _get_model(self):
        """懒加载 cross-encoder 模型。

        首次调用时加载模型, 后续调用直接返回已加载的模型实例。
        记录模型加载耗时。
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder
            # [RERANKER] 模型加载开始, 记录起始时间
            load_start = time.time()
            print(f"[RERANKER] 模型加载开始: {self.model_path}")
            self._model = CrossEncoder(self.model_path)
            # [RERANKER] 模型加载完成, 记录耗时
            load_end = time.time()
            load_elapsed = load_end - load_start
            print(f"[RERANKER] 模型加载完成, 耗时={load_elapsed:.2f}s, 模型={self.model_path}")
        return self._model

    def rerank(
        self, query: str, results: List[Tuple[Dict, float]], top_k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """对检索结果进行重排序。

        Args:
            query: 用户查询文本
            results: 初步检索结果列表, 每个元素为 (chunk_dict, original_score) 的元组
                     chunk_dict 需包含 "text" 字段
            top_k: 返回的最相关结果数量, 默认为 3

        Returns:
            重排序后的结果列表, 按分数降序排列, 每个元素为 (chunk_dict, rerank_score)
        """
        # [RERANKER] 打印输入结果数量
        print(f"[RERANKER] 开始重排序, 输入结果数={len(results)}, top_k={top_k}")

        # 如果输入结果为空, 直接返回空列表
        if not results:
            print("[RERANKER] 输入结果为空, 跳过重排序")
            return []

        # 记录重排序开始时间
        rerank_start = time.time()

        # 加载模型(懒加载, 首次会触发模型下载/加载)
        model = self._get_model()

        # 构建 query-document 对, 用于 cross-encoder 打分
        # cross-encoder 需要输入 (query, document) 的配对
        pairs = [(query, chunk["text"]) for chunk, _ in results]

        # 使用 cross-encoder 模型对每一对进行相关性打分
        scores = model.predict(pairs)

        # 将原始结果与重排序分数组合, 便于排序
        scored = [(results[i][0], float(scores[i])) for i in range(len(results))]

        # 按分数降序排列
        scored.sort(key=lambda x: x[1], reverse=True)

        # 取前 top_k 个结果
        top_results = scored[:top_k]

        # 记录重排序完成时间
        rerank_end = time.time()
        rerank_elapsed = rerank_end - rerank_start

        # [RERANKER] 打印重排序结果和耗时
        print(f"[RERANKER] 重排序完成, 耗时={rerank_elapsed:.2f}s, 输出结果数={len(top_results)}")
        for idx, (chunk, score) in enumerate(top_results):
            # 显示每个结果的分数, 截断文本前50个字符作为预览
            text_preview = chunk["text"][:50].replace("\n", " ")
            print(
                f"[RERANKER]   Top-{idx + 1}: score={score:.4f}, text_preview=\"{text_preview}...\""
            )

        return top_results
