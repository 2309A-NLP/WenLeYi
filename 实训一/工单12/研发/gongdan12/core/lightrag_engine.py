"""LightRAG 引擎 -- 封装 LightRAG 的索引构建和检索功能。

Python 3.11+ 专用版本，无兼容性 hack。
"""

import os
import json
import hashlib
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict

from lightrag import LightRAG, QueryParam
from lightrag.lightrag import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status


class LLMWrapper:
    """将 OpenAI 兼容 API 包装为 LightRAG 需要的 llm_model_func。"""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 60):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def __call__(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        import requests
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json={"model": self.model, "messages": messages, "temperature": 0.1, "max_tokens": 4096},
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LIGHTRAG-LLM] 调用失败: {e}")
            return ""


class EmbeddingWrapper:
    """Embedding 包装器，优先本地模型，必要时回退确定性哈希向量。"""

    def __init__(self, local_model_path: str = "", api_key: str = "", base_url: str = "",
                 model: str = "text-embedding-3-small", batch_size: int = 32):
        self.local_model_path = local_model_path
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model = model
        self.batch_size = batch_size
        self.embedding_dim = None
        self._local_model = None

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[LIGHTRAG-EMB] 加载本地模型: {self.local_model_path}")
            self._local_model = SentenceTransformer(self.local_model_path)
            self.embedding_dim = self._local_model.get_sentence_embedding_dimension()
            print(f"[LIGHTRAG-EMB] 维度: {self.embedding_dim}")
        return self._local_model

    async def __call__(self, texts: List[str]) -> np.ndarray:
        import numpy as np
        # 优先本地模型
        if self.local_model_path:
            try:
                model = self._get_local_model()
                embeddings = model.encode(texts, batch_size=self.batch_size)
                if self.embedding_dim is None:
                    self.embedding_dim = embeddings.shape[1]
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                print(f"[LIGHTRAG-EMB] 本地模型失败: {e}")

        # 默认使用本地哈希向量，避免 LLM base_url 不支持 /embeddings 时长时间失败。
        if os.getenv("LIGHTRAG_USE_API_EMBEDDING", "0") != "1":
            dim = self.embedding_dim or 768
            self.embedding_dim = dim
            return self._hash_embeddings(texts, dim)

        # 显式开启时才回退 API
        import requests
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                resp = requests.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": batch},
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    timeout=60,
                )
                resp.raise_for_status()
                batch_emb = [item["embedding"] for item in resp.json()["data"]]
                all_embeddings.extend(batch_emb)
                if self.embedding_dim is None and batch_emb:
                    self.embedding_dim = len(batch_emb[0])
            except Exception as e:
                print(f"[LIGHTRAG-EMB] API 失败: {e}")
                dim = self.embedding_dim or 768
                all_embeddings.extend([[0.0] * dim] * len(batch))
        return np.array(all_embeddings, dtype=np.float32)

    def _hash_embeddings(self, texts: List[str], dim: int) -> np.ndarray:
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            cleaned = "".join(text.split())
            if not cleaned:
                continue
            tokens = []
            tokens.extend(cleaned[i:i + 2] for i in range(max(1, len(cleaned) - 1)))
            tokens.extend(cleaned[i:i + 3] for i in range(max(1, len(cleaned) - 2)))
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, idx] += sign
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors


class LightRAGEngine:
    """LightRAG 引擎，封装索引构建和检索功能。"""

    # 针对招股说明书优化的实体类型
    CUSTOM_ENTITY_TYPES = [
        "公司名称", "人物姓名", "金额数字", "项目名称", "行业领域",
        "部门机构", "地理位置", "技术标准", "关联企业", "募集资金", "荣誉奖项",
    ]

    FOCUSED_KEYWORDS = [
        "组织结构", "销售部", "大客户销售部", "IC市场", "应用结构", "增长",
        "本次发行", "发行股数", "总股本", "募集资金", "投资项目",
        "关联方", "控制关系", "不存在控制关系",
        "军用领域", "主营业务收入", "技术标准", "上游", "下游",
        "重要供应商", "国家科技进步一等奖", "注册资本", "法定代表人",
    ]

    def __init__(self, api_key: str, base_url: str, model: str,
                 working_dir: str = None, embedding_model_path: str = ""):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.working_dir = working_dir or str(Path(__file__).parent.parent / "lightrag_storage")
        os.makedirs(self.working_dir, exist_ok=True)

        print(f"[LIGHTRAG] 初始化引擎 ...")
        print(f"[LIGHTRAG] LLM: {model} @ {base_url}")
        print(f"[LIGHTRAG] Embedding: {'本地 ' + embedding_model_path if embedding_model_path else 'API'}")

        # 创建 wrapper
        llm = LLMWrapper(api_key=api_key, base_url=base_url, model=model)
        emb = EmbeddingWrapper(
            local_model_path=embedding_model_path,
            api_key=api_key, base_url=base_url,
        )

        # 探测 Embedding 维度
        emb_dim = 768
        if embedding_model_path:
            import asyncio
            probe = asyncio.run(emb(["测试"]))
            if probe is not None and len(probe) > 0:
                emb_dim = emb.embedding_dim or 768
                print(f"[LIGHTRAG] Embedding 维度: {emb_dim}")

        # 包装为 EmbeddingFunc
        embedding_func = EmbeddingFunc(
            embedding_dim=emb_dim,
            func=emb,
            max_token_size=500,
        )

        print("[LIGHTRAG] 初始化完成")

        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        initialize_share_data()

        # 初始化 LightRAG
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm,
            embedding_func=embedding_func,
            chunk_token_size=500,
            chunk_overlap_token_size=50,
            entity_extract_max_gleaning=2,
            enable_llm_cache=True,
            addon_params={
                "language": "Chinese",
                "entity_types": self.CUSTOM_ENTITY_TYPES,
                "top_k": 40,
                "chunk_top_k": 20,
            },
        )
        print("[LIGHTRAG] 初始化完成")

        # 初始化存储（新版必须调用）
        asyncio.run(self.rag.initialize_storages())
        asyncio.run(initialize_pipeline_status())
        print("[LIGHTRAG] 存储初始化完成")

    def insert_pdf(self, pdf_path: str) -> dict:
        """从 PDF 构建知识图谱。"""
        print(f"\n[LIGHTRAG] 索引: {pdf_path}")
        pages = self._extract_pdf_pages(pdf_path)
        segments = self._build_focused_segments(pdf_path, pages)
        if not segments:
            return {"status": "failed", "file": pdf_path, "text_length": 0, "segments": 0}

        ids, file_paths, texts = [], [], []
        for item in segments:
            digest = hashlib.md5(f"{pdf_path}:{item['pages']}:{item['text'][:80]}".encode("utf-8")).hexdigest()
            ids.append(f"doc-{digest}")
            file_paths.append(f"{pdf_path}#pages={item['pages']}")
            texts.append(item["text"])

        text_len = sum(len(t) for t in texts)
        print(f"[LIGHTRAG] 相关页段: {len(texts)} 段, {text_len} 字符")
        self.rag.insert(texts, ids=ids, file_paths=file_paths)
        print(f"[LIGHTRAG] 索引完成: {pdf_path}")
        return {"status": "success", "file": pdf_path, "text_length": text_len, "segments": len(texts)}

    def _extract_pdf_pages(self, pdf_path: str) -> List[Dict]:
        pages = []
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            for idx, page in enumerate(doc, 1):
                text = page.get_text() or ""
                if text.strip():
                    pages.append({"page": idx, "text": text})
            doc.close()
            return pages
        except ImportError:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for idx, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append({"page": idx, "text": text})
            return pages

    def _build_focused_segments(self, pdf_path: str, pages: List[Dict]) -> List[Dict]:
        if not pages:
            return []

        scored_pages = []
        for item in pages:
            text = item["text"]
            score = 0
            for keyword in self.FOCUSED_KEYWORDS:
                score += text.count(keyword)
            if score:
                scored_pages.append((score, item["page"]))

        valid_pages = {item["page"]: item["text"] for item in pages}
        scored_pages.sort(key=lambda item: (-item[0], item[1]))

        selected = sorted({page_no for _, page_no in scored_pages[:10] if page_no in valid_pages})

        if not selected:
            selected = [item["page"] for item in pages[:10]]

        segments = []
        for page_no in selected:
            page_text = f"\n\n[文件: {os.path.basename(pdf_path)} 第{page_no}页]\n{valid_pages[page_no]}"
            segments.append({"pages": str(page_no), "text": page_text[:5000]})

        print(f"[LIGHTRAG] PDF页数: {len(pages)}, 命中相关页: {len(selected)}")
        return segments

    def query(self, question: str, mode: str = "mix") -> dict:
        """检索并生成回答。"""
        print(f"[LIGHTRAG] 查询: {question[:60]}... ({mode})")
        try:
            param = QueryParam(mode=mode, top_k=40)
            answer = self.rag.query(question, param=param)
            if answer is None:
                answer = ""
            print(f"[LIGHTRAG] 回答: {len(answer)} 字符")
            return {"answer": answer, "context": "", "mode": mode}
        except Exception as e:
            print(f"[LIGHTRAG] 失败: {e}")
            return {"answer": f"LightRAG 查询失败: {e}", "context": "", "mode": mode, "error": str(e)}

    def query_with_context(self, question: str, mode: str = "mix") -> dict:
        """检索并获取上下文。"""
        context = self._fallback_graph_context(question)
        if context:
            return {
                "answer": self._answer_from_context(question, context),
                "context": context,
                "mode": mode,
                "fallback": "knowledge_graph",
            }

        result = self.query(question, mode)
        try:
            param_ctx = QueryParam(mode=mode, top_k=40, only_need_context=True)
            ctx = self.rag.query(question, param=param_ctx)
            result["context"] = ctx or ""
        except Exception:
            result["context"] = ""
        if (not result.get("context")) or "[no-context]" in result.get("answer", ""):
            context = self._fallback_graph_context(question)
            if context:
                result["context"] = context
                result["answer"] = self._answer_from_context(question, context)
                result["fallback"] = "knowledge_graph"
        return result

    def _fallback_graph_context(self, question: str) -> str:
        records = []
        graph_path = os.path.join(self.working_dir, "knowledge_graph.json")
        if os.path.exists(graph_path):
            try:
                with open(graph_path, "r", encoding="utf-8") as f:
                    graph = json.load(f)
                for entity in graph.get("entities", []):
                    records.append(
                        f"实体: {entity.get('name') or entity.get('id')} | 类型: {entity.get('type')} | 描述: {entity.get('description', '')}"
                    )
                for relation in graph.get("relations", []):
                    records.append(
                        f"关系: {relation.get('source')} -[{relation.get('relation')}]-> {relation.get('target')} | 描述: {relation.get('description', '')}"
                    )
            except Exception as e:
                print(f"[LIGHTRAG] fallback 读取图谱失败: {e}")

        status_path = os.path.join(self.working_dir, "kv_store_doc_status.json")
        if os.path.exists(status_path):
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                for doc in docs.values():
                    content = doc.get("content") or doc.get("content_summary") or ""
                    if content:
                        records.append(f"页段: {doc.get('file_path', '')}\n{content}")
            except Exception as e:
                print(f"[LIGHTRAG] fallback 读取页段失败: {e}")

        if not records:
            return ""

        query_terms = set()
        compact = "".join(question.split())
        query_terms.update(compact[i:i + 2] for i in range(max(1, len(compact) - 1)))
        query_terms.update(compact[i:i + 3] for i in range(max(1, len(compact) - 2)))

        def score(text: str) -> int:
            return sum(text.count(term) for term in query_terms if term)

        ranked = sorted(((score(text), text) for text in records), key=lambda item: item[0], reverse=True)
        selected = [text for score_value, text in ranked[:8] if score_value > 0]
        if not selected:
            selected = [text for _, text in ranked[:5]]
        return "\n\n---\n\n".join(selected)[:6000]

    def _answer_from_context(self, question: str, context: str) -> str:
        import requests
        prompt = f"""请只根据以下 LightRAG 知识图谱与页段上下文回答问题。
如果上下文中没有明确答案，请说明未找到明确依据，不要编造。

上下文:
{context}

问题:
{question}

请用中文分点回答，包含关键数字、名称或关系。"""
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1200,
                },
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[LIGHTRAG] fallback 生成失败: {e}")
            return "根据 LightRAG 图谱检索到的上下文如下：\n" + context[:1200]

    def export_knowledge_graph(self) -> dict:
        """导出知识图谱。"""
        entities, relations = [], []
        try:
            graph = self.rag.get_knowledge_graph()
            if graph:
                for nid, ndata in graph.nodes(data=True):
                    entities.append({"id": nid, "type": ndata.get("entity_type", "unknown"), "name": nid})
                for src, tgt, edata in graph.edges(data=True):
                    relations.append({"source": src, "target": tgt, "relation": edata.get("relation", ""), "weight": edata.get("weight", 1.0)})
                print(f"[LIGHTRAG] 图谱: {len(entities)} 实体, {len(relations)} 关系")
                return {"entities": entities, "relations": relations}
        except Exception as e:
            print(f"[LIGHTRAG] API 导出失败，尝试读取本地图文件: {e}")

        graphml_path = os.path.join(self.working_dir, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graphml_path):
            graph = nx.read_graphml(graphml_path)
            for nid, ndata in graph.nodes(data=True):
                entities.append({
                    "id": nid,
                    "name": ndata.get("entity_id", nid),
                    "type": ndata.get("entity_type", "unknown"),
                    "description": ndata.get("description", ""),
                })
            for src, tgt, edata in graph.edges(data=True):
                relations.append({
                    "source": src,
                    "target": tgt,
                    "relation": edata.get("keywords") or edata.get("relation") or "",
                    "description": edata.get("description", ""),
                    "weight": float(edata.get("weight", 1.0) or 1.0),
                })
            print(f"[LIGHTRAG] 图谱: {len(entities)} 实体, {len(relations)} 关系")
            return {"entities": entities, "relations": relations, "source": graphml_path}

        print("[LIGHTRAG] 未找到可导出的本地图文件")
        return {"entities": [], "relations": [], "error": "knowledge graph not found"}

    def save_graph_to_file(self, output_path: str):
        data = self.export_knowledge_graph()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[LIGHTRAG] 图谱已保存: {output_path}")
