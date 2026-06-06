"""Milvus 向量存储 -- 远程 Milvus 服务。

本模块提供两个核心类:
- VectorStore: 基于 Milvus 的文档向量存储，支持索引构建、检索、增量更新等
- QuestionStore: 基于 FAISS 的历史问答缓存，用于快速命中高频问题
"""
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# VectorStore: 基于 Milvus 的文档向量存储
# ---------------------------------------------------------------------------
class VectorStore:
    """Milvus 向量存储管理器。

    负责将文档切片(chunk)编码为向量并存入 Milvus, 同时提供语义检索能力。
    """

    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        collection_name: str = "rag_chunks",
        embedding_model_path: str = "",
        embedding_batch_size: int = 32,
        insert_batch_size: int = 200,
    ):
        """初始化向量存储, 记录所有连接参数。

        Args:
            milvus_host: Milvus 服务器地址
            milvus_port: Milvus 服务器端口
            collection_name: Milvus 集合名称
            embedding_model_path: SentenceTransformer 模型路径或名称
        """
        print("[VECTOR] 初始化 VectorStore")
        print(f"[VECTOR]   milvus_host = {milvus_host}")
        print(f"[VECTOR]   milvus_port = {milvus_port}")
        print(f"[VECTOR]   collection_name = {collection_name}")
        print(f"[VECTOR]   embedding_model_path = {embedding_model_path or '(默认: shibing624/text2vec-base-chinese)'}")

        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self._embedder = None
        self._embedding_model_path = embedding_model_path
        self._client = None
        self._chunks = []
        self.embedding_batch_size = max(1, int(embedding_batch_size or 32))
        self.insert_batch_size = max(1, int(insert_batch_size or 200))

    def _get_client(self):
        """懒加载 Milvus 客户端。

        首次调用时创建 MilvusClient 实例, 后续复用同一连接。
        """
        if self._client is None:
            from pymilvus import MilvusClient

            uri = f"http://{self.milvus_host}:{self.milvus_port}"
            print(f"[VECTOR] 尝试连接 Milvus: {uri}")
            t0 = time.time()
            try:
                self._client = MilvusClient(uri=uri)
                elapsed = time.time() - t0
                print(f"[VECTOR] Milvus 连接成功 ({elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"[VECTOR] Milvus 连接失败 ({elapsed:.2f}s): {e}")
                raise
        return self._client

    def _get_embedder(self):
        """懒加载 Embedding 模型。

        首次调用时加载 SentenceTransformer 模型, 后续复用同一实例。
        """
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            model_path = self._embedding_model_path or "shibing624/text2vec-base-chinese"
            print(f"[VECTOR] 加载 Embedding 模型: {model_path}")
            t0 = time.time()
            self._embedder = SentenceTransformer(model_path)
            elapsed = time.time() - t0
            print(f"[VECTOR] Embedding 模型加载完成 ({elapsed:.2f}s)")
        return self._embedder

    def _ensure_collection(self, dim: int):
        """确保 Milvus collection 存在, 不存在则创建。

        根据向量维度自动创建集合, 使用 FLAT 索引 + COSINE 相似度。

        Args:
            dim: 向量维度
        """
        print(f"[VECTOR] 检查 collection '{self.collection_name}' 是否存在")
        from pymilvus import DataType

        client = self._get_client()

        if client.has_collection(self.collection_name):
            print(f"[VECTOR] collection '{self.collection_name}' 已存在, 跳过创建")
            return

        print(f"[VECTOR] collection '{self.collection_name}' 不存在, 开始创建 (维度={dim})")
        t0 = time.time()

        # 定义 schema: id, embedding, text, source, chunk_id
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_id", DataType.INT64)

        # 配置索引参数: FLAT 类型, COSINE 相似度
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type="COSINE",
        )

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

        elapsed = time.time() - t0
        print(f"[VECTOR] collection 创建完成 ({elapsed:.2f}s)")

    def build_index(self, chunks: List[Dict]):
        """从文本块构建 Milvus 向量索引。

        流程: 提取文本 -> 编码为向量 -> 确保集合存在 -> 批量插入
        每批 200 条记录, 打印每批进度。

        Args:
            chunks: 文本块列表, 每个元素需包含 text, source, chunk_id 字段
        """
        if not chunks:
            print("[VECTOR] 没有文本块, 跳过索引构建")
            return

        texts = [c["text"] for c in chunks]
        embedder = self._get_embedder()

        print(f"[VECTOR] 开始编码 {len(texts)} 个文本块...")
        t0 = time.time()
        vectors = embedder.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        vectors = np.array(vectors, dtype=np.float32)
        dim = vectors.shape[1]
        elapsed = time.time() - t0
        print(f"[VECTOR] 编码完成: {len(texts)} 个文本块, 维度 {dim}, 耗时 {elapsed:.2f}s")

        # 确保 Milvus 中的 collection 存在
        self._ensure_collection(dim)

        # 批量插入向量到 Milvus, 每批 200 条
        client = self._get_client()
        batch_size = self.insert_batch_size
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        t_insert = time.time()

        for batch_idx, i in enumerate(range(0, len(chunks), batch_size), 1):
            batch = chunks[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size].tolist()
            data = [
                {
                    "embedding": vec,
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                }
                for vec, chunk in zip(batch_vectors, batch)
            ]
            client.insert(collection_name=self.collection_name, data=data)
            print(
                f"[VECTOR] 批量插入进度: {batch_idx}/{total_batches} "
                f"(已插入 {min(i + batch_size, len(chunks))}/{len(chunks)})"
            )

        insert_elapsed = time.time() - t_insert
        print(f"[VECTOR] 索引构建完成: {len(chunks)} 条向量, 维度 {dim}, 插入耗时 {insert_elapsed:.2f}s")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """向量语义检索。

        将查询文本编码后在 Milvus 中进行最近邻搜索, 返回最相似的 top_k 个结果。

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            列表, 每个元素为 (chunk_dict, score) 的元组
        """
        print(f"[VECTOR] 搜索查询: '{query[:80]}...' (top_k={top_k})" if len(query) > 80 else f"[VECTOR] 搜索查询: '{query}' (top_k={top_k})")

        t0 = time.time()

        # 编码查询文本
        embedder = self._get_embedder()
        q_vec = embedder.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype=np.float32).tolist()[0]

        # 在 Milvus 中执行搜索
        client = self._get_client()
        results = client.search(
            collection_name=self.collection_name,
            data=[q_vec],
            limit=top_k,
            output_fields=["text", "source", "chunk_id"],
        )

        # 整理搜索结果
        output = []
        for hit in results[0]:
            chunk = {
                "text": hit["entity"]["text"],
                "source": hit["entity"]["source"],
                "chunk_id": hit["entity"]["chunk_id"],
            }
            output.append((chunk, float(hit["distance"])))

        elapsed = time.time() - t0
        # 打印搜索结果概要, 包含返回数量和前3个最高分
        top_scores = [f"{score:.4f}" for _, score in output[:3]]
        print(
            f"[VECTOR] 搜索完成: 返回 {len(output)} 条结果, "
            f"最高分 = [{', '.join(top_scores)}], 耗时 {elapsed:.2f}s"
        )

        return output

    def load_chunks(self) -> List[Dict]:
        """从 Milvus 加载所有 chunk 元数据 (用于 BM25 初始化等)。

        查询 collection 中的所有记录, 提取 text, source, chunk_id 字段。

        Returns:
            chunk 字典列表
        """
        print("[VECTOR] 加载所有 chunks 元数据...")
        try:
            client = self._get_client()
            if not client.has_collection(self.collection_name):
                print("[VECTOR] collection 不存在, 返回空列表")
                return []

            t0 = time.time()
            res = client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["text", "source", "chunk_id"],
                limit=10000,
            )
            self._chunks = [
                {"text": r["text"], "source": r["source"], "chunk_id": r["chunk_id"]}
                for r in res
            ]
            elapsed = time.time() - t0
            print(f"[VECTOR] load_chunks 完成: 共加载 {len(self._chunks)} 个 chunks ({elapsed:.2f}s)")
            return self._chunks
        except Exception as e:
            print(f"[VECTOR] load_chunks 失败: {e}")
            return []

    def build_index_incremental(self, chunks: List[Dict]):
        """增量更新: 仅添加新文件的向量, 跳过已存在的文件。

        流程: 查询已有 source -> 筛选新 chunk -> 编码新 chunk -> 批量插入

        Args:
            chunks: 完整的文本块列表
        """
        print("[VECTOR] 开始增量索引更新...")

        # 查询 Milvus 中已存在的 source 集合
        existing_sources = set()
        try:
            client = self._get_client()
            if client.has_collection(self.collection_name):
                res = client.query(
                    collection_name=self.collection_name,
                    filter="",
                    output_fields=["source"],
                    limit=10000,
                )
                existing_sources = {r["source"] for r in res}
        except Exception:
            pass

        print(f"[VECTOR] 已有 source 数量: {len(existing_sources)}")

        # 筛选出新 chunk (不在已有 source 中的)
        new_chunks = [c for c in chunks if c.get("source", "") not in existing_sources]
        skipped = len(chunks) - len(new_chunks)

        if not new_chunks:
            print(f"[VECTOR] 没有新文档需要索引 (全部 {len(chunks)} 个已存在, 跳过 {skipped} 个)")
            return

        print(f"[VECTOR] 增量更新: 新增 {len(new_chunks)} 个 chunks, 跳过 {skipped} 个已存在 chunks")

        # 编码新 chunk
        texts = [c["text"] for c in new_chunks]
        embedder = self._get_embedder()
        print(f"[VECTOR] 开始编码 {len(texts)} 个新文本块...")
        t0 = time.time()
        vectors = embedder.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        vectors = np.array(vectors, dtype=np.float32)
        dim = vectors.shape[1]
        elapsed = time.time() - t0
        print(f"[VECTOR] 编码完成: {len(texts)} 个文本块, 维度 {dim}, 耗时 {elapsed:.2f}s")

        # 确保 collection 存在
        self._ensure_collection(dim)

        # 批量插入新向量
        client = self._get_client()
        batch_size = self.insert_batch_size
        total_batches = (len(new_chunks) + batch_size - 1) // batch_size
        t_insert = time.time()

        for batch_idx, i in enumerate(range(0, len(new_chunks), batch_size), 1):
            batch = new_chunks[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size].tolist()
            data = [
                {
                    "embedding": vec,
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                }
                for vec, chunk in zip(batch_vectors, batch)
            ]
            client.insert(collection_name=self.collection_name, data=data)
            print(
                f"[VECTOR] 增量插入进度: {batch_idx}/{total_batches} "
                f"(已插入 {min(i + batch_size, len(new_chunks))}/{len(new_chunks)})"
            )

        insert_elapsed = time.time() - t_insert
        print(f"[VECTOR] 增量更新完成: 新增 {len(new_chunks)} 条向量, 插入耗时 {insert_elapsed:.2f}s")

    @property
    def is_ready(self) -> bool:
        """检查 collection 是否存在且包含数据。

        Returns:
            True 表示 collection 存在且行数 > 0
        """
        try:
            client = self._get_client()
            if not client.has_collection(self.collection_name):
                print(f"[VECTOR] is_ready 检查: collection '{self.collection_name}' 不存在 -> False")
                return False
            stats = client.get_collection_stats(self.collection_name)
            row_count = int(stats.get("row_count", 0))
            result = row_count > 0
            print(f"[VECTOR] is_ready 检查: collection '{self.collection_name}' 行数={row_count} -> {result}")
            return result
        except Exception as e:
            print(f"[VECTOR] is_ready 检查异常: {e} -> False")
            return False


# ---------------------------------------------------------------------------
# QuestionStore: 基于 FAISS 的历史问答缓存
# ---------------------------------------------------------------------------
class QuestionStore:
    """基于 Milvus 的历史问答缓存存储。

    将用户历史问答对编码为向量并存储在 Milvus 的 qa_history collection 中,
    当新问题到来时先检查历史库, 若相似度足够高则直接返回历史答案。
    """

    def __init__(self, milvus_host: str, milvus_port: int, collection_name: str, embedding_model_path: str = ""):
        """初始化 QuestionStore (Milvus 版)。"""
        print("[VECTOR] 初始化 QuestionStore (Milvus)")
        print(f"[VECTOR]   milvus_host = {milvus_host}")
        print(f"[VECTOR]   milvus_port = {milvus_port}")
        print(f"[VECTOR]   collection_name = {collection_name}")

        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self._embedding_model_path = embedding_model_path
        self._embedder = None
        self._client = None
        self._dim = 0

    def _get_client(self):
        """懒加载 Milvus 客户端。"""
        if self._client is None:
            from pymilvus import MilvusClient
            uri = f"http://{self.milvus_host}:{self.milvus_port}"
            print(f"[VECTOR] QuestionStore 连接 Milvus: {uri}")
            self._client = MilvusClient(uri=uri)
        return self._client

    def _get_embedder(self):
        """懒加载 Embedding 模型。"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            model_path = self._embedding_model_path or "shibing624/text2vec-base-chinese"
            print(f"[VECTOR] QuestionStore 加载 Embedding 模型: {model_path}")
            t0 = time.time()
            self._embedder = SentenceTransformer(model_path)
            print(f"[VECTOR] QuestionStore Embedding 模型加载完成 ({time.time() - t0:.2f}s)")
        return self._embedder

    def _ensure_collection(self, dim: int):
        """确保 qa_history collection 存在。"""
        from pymilvus import DataType
        client = self._get_client()
        if client.has_collection(self.collection_name):
            return
        print(f"[VECTOR] QuestionStore 创建 collection: {self.collection_name}, 维度={dim}")
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("question", DataType.VARCHAR, max_length=2048)
        schema.add_field("answer", DataType.VARCHAR, max_length=8192)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")
        client.create_collection(collection_name=self.collection_name, schema=schema, index_params=index_params)
        print(f"[VECTOR] QuestionStore collection 创建完成")

    def _question_filter(self, question: str) -> str:
        """Build a Milvus scalar filter for an exact question match."""
        return f"question == {json.dumps(question, ensure_ascii=False)}"

    def get_exact_answer(self, question: str) -> Optional[str]:
        """Return the cached answer for an exact question match without vector search."""
        question = question.strip()
        if not question:
            return None

        try:
            client = self._get_client()
            if not client.has_collection(self.collection_name):
                print("[VECTOR] QuestionStore exact lookup skipped: collection missing")
                return None

            rows = client.query(
                collection_name=self.collection_name,
                filter=self._question_filter(question),
                output_fields=["answer"],
                limit=1,
            )
            if not rows:
                print("[VECTOR] QuestionStore exact lookup miss")
                return None

            print("[VECTOR] QuestionStore exact lookup hit")
            return rows[0].get("answer")
        except Exception as e:
            print(f"[VECTOR] QuestionStore exact lookup failed: {e}")
            return None

    def delete_question(self, question: str) -> int:
        """Delete exact question cache rows from the QA history collection."""
        question = question.strip()
        if not question:
            return 0

        try:
            client = self._get_client()
            if not client.has_collection(self.collection_name):
                print("[VECTOR] QuestionStore delete skipped: collection missing")
                return 0

            rows = client.query(
                collection_name=self.collection_name,
                filter=self._question_filter(question),
                output_fields=["id"],
                limit=100,
            )
            if not rows:
                print("[VECTOR] QuestionStore delete miss")
                return 0

            client.delete(
                collection_name=self.collection_name,
                filter=self._question_filter(question),
            )
            print(f"[VECTOR] QuestionStore deleted question cache: count={len(rows)}")
            return len(rows)
        except Exception as e:
            print(f"[VECTOR] QuestionStore delete failed: {e}")
            return 0

    @property
    def is_ready(self) -> bool:
        """检查 collection 是否有数据。"""
        try:
            client = self._get_client()
            if not client.has_collection(self.collection_name):
                print("[VECTOR] QuestionStore is_ready: False (collection 不存在)")
                return False
            stats = client.get_collection_stats(self.collection_name)
            count = int(stats.get("row_count", 0))
            print(f"[VECTOR] QuestionStore is_ready: {count > 0} (total={count})")
            return count > 0
        except Exception as e:
            print(f"[VECTOR] QuestionStore is_ready 异常: {e}")
            return False

    def search(self, question: str, top_k: int = 1) -> List[Tuple[Dict, float]]:
        """在历史问答库中搜索最相似的问题。"""
        try:
            client = self._get_client()
            if not client.has_collection(self.collection_name):
                print("[VECTOR] QuestionStore collection 不存在, 跳过搜索")
                return []
            stats = client.get_collection_stats(self.collection_name)
            if int(stats.get("row_count", 0)) == 0:
                print("[VECTOR] QuestionStore collection 为空, 跳过搜索")
                return []
        except Exception as e:
            print(f"[VECTOR] QuestionStore 连接失败: {e}")
            return []

        print(f"[VECTOR] QuestionStore 搜索: '{question[:60]}...' (top_k={top_k})" if len(question) > 60 else f"[VECTOR] QuestionStore 搜索: '{question}' (top_k={top_k})")
        t0 = time.time()
        embedder = self._get_embedder()
        q_vec = embedder.encode([question], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype=np.float32).tolist()[0]

        client = self._get_client()
        results = client.search(
            collection_name=self.collection_name,
            data=[q_vec],
            limit=top_k,
            output_fields=["question", "answer"],
        )

        output = []
        for hit in results[0]:
            meta = {"question": hit["entity"]["question"], "answer": hit["entity"]["answer"]}
            output.append((meta, float(hit["distance"])))

        elapsed = time.time() - t0
        top_scores_str = [f"{s:.4f}" for _, s in output[:3]]
        print(f"[VECTOR] QuestionStore 搜索完成: {len(output)} 条结果, 最高分=[{', '.join(top_scores_str)}], 耗时 {elapsed:.2f}s")
        return output

    def add_qa(self, question: str, answer: str):
        """将一条问答对添加到历史库。"""
        question = question.strip()
        print(f"[VECTOR] QuestionStore 添加问答对: '{question[:40]}...'")
        t0 = time.time()

        if self.get_exact_answer(question) is not None:
            print("[VECTOR] QuestionStore exact question already exists, skip insert")
            return

        embedder = self._get_embedder()
        q_vec = embedder.encode([question], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype=np.float32)
        dim = q_vec.shape[1]

        self._ensure_collection(dim)

        client = self._get_client()
        data = [{
            "embedding": q_vec.tolist()[0],
            "question": question,
            "answer": answer,
        }]
        client.insert(collection_name=self.collection_name, data=data)

        elapsed = time.time() - t0
        stats = client.get_collection_stats(self.collection_name)
        total = int(stats.get("row_count", 0))
        print(f"[VECTOR] QuestionStore 问答对添加完成, 当前总数: {total} ({elapsed:.2f}s)")
