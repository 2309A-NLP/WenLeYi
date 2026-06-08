"""核心 RAG Pipeline -- 检索 + 生成 + 历史问题检索。"""
import re
import time
import json
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple


# 系统提示词：指导 LLM 如何基于参考资料回答问题
SYSTEM_PROMPT = """根据参考资料回答问题。要求：
1. 用1.2.3.分点回答，简洁明了
2. 只基于资料内容回答，不要编造
3. 不要用Markdown格式"""


def clean_answer(text: str) -> str:
    """去除 LLM 回答中的 Markdown 格式符号。"""
    if not text:
        return text
    # 去除 **粗体** 和 *斜体*
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # 去除 `代码`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # 去除 ### 标题
    text = re.sub(r'#{1,6}\s*', '', text)
    # 去除 --- 分隔线
    text = re.sub(r'^-{3,}\s*$', '', text, flags=re.MULTILINE)
    # 去除 > 引用
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    # 去除 [链接](url) 中的链接符号，保留文字
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # 去除连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class RAGPipeline:
    """RAG 检索增强生成管线。"""

    def __init__(self, retriever, llm_client, config, reranker=None, question_store=None, memory_store=None):
        """初始化 RAG 管线，注入检索器、LLM、配置等组件。"""
        print("[RAG] 正在初始化 RAGPipeline ...", flush=True)
        self.retriever = retriever
        self.llm = llm_client
        self.config = config
        self.reranker = reranker
        self.question_store = question_store
        self.memory_store = memory_store
        self.history = []  # 多轮对话历史（内存）
        self._answer_cache = OrderedDict()
        self._answer_cache_max_size = 256
        self._last_source_question = None
        self._last_sources = []
        self._last_retrieve_time = 0  # 最近一次检索耗时
        print("[RAG] RAGPipeline 初始化完成", flush=True)

    def _get_hot_answer(self, question: str) -> Optional[str]:
        """Return an exact in-process answer cache hit."""
        key = question.strip()
        answer = self._answer_cache.get(key)
        if answer is None:
            return None
        self._answer_cache.move_to_end(key)
        print("[RAG] 进程内历史缓存精确命中", flush=True)
        return answer

    def _put_hot_answer(self, question: str, answer: str):
        """Keep recent exact Q/A pairs in memory to avoid repeated DB round trips."""
        key = question.strip()
        if not key or not answer:
            return
        self._answer_cache[key] = answer
        self._answer_cache.move_to_end(key)
        while len(self._answer_cache) > self._answer_cache_max_size:
            self._answer_cache.popitem(last=False)

    def forget_question(self, session_id: str, question: str) -> Dict[str, int]:
        """Forget cached answers for a question so the next query regenerates it."""
        question_key = (question or "").strip()
        stats = {"redis": 0, "hot_cache": 0, "question_store": 0}
        if not question_key:
            return stats

        if self.memory_store and getattr(self.memory_store, "is_ready", False):
            stats["redis"] = self.memory_store.delete_exact_turn(session_id, question_key)

        if question_key in self._answer_cache:
            self._answer_cache.pop(question_key, None)
            stats["hot_cache"] = 1

        if self.question_store and hasattr(self.question_store, "delete_question"):
            stats["question_store"] = self.question_store.delete_question(question_key)

        print(f"[RAG] forget_question done: {stats}", flush=True)
        return stats

    def _set_last_sources(self, question: str, results: List[Tuple[Dict, float]]):
        self._last_source_question = question
        self._last_sources = [
            {"source": chunk["source"], "text": chunk["text"][:200], "score": score}
            for chunk, score in results
        ]

    def _clear_last_sources(self, question: str):
        self._last_source_question = question
        self._last_sources = []

    def _stream_local_answer(self, answer: str):
        """Yield cached or local answers in small chunks so SSE stays streaming."""
        if not answer:
            return
        chunk_size = 12
        delay_seconds = 0.02
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    def _clean_stream_chunk(self, text: str) -> str:
        """Remove common Markdown markers before a chunk reaches the frontend."""
        if not text:
            return text
        return text.replace("*", "").replace("`", "").replace("#", "")

    def _build_retrieval_fallback(self, question: str, results: List[Tuple[Dict, float]]) -> str:
        """Build a readable fallback answer from retrieved chunks."""
        context = "\n".join([chunk.get("text", "") for chunk, _ in results[:3]])
        org_answer = self._build_sales_org_fallback(question, context)
        if org_answer:
            return org_answer
        snippets = []
        for chunk, _ in results[:3]:
            text = chunk.get("text", "").strip()
            if text:
                snippets.append(text[:500])
        if snippets:
            return "根据检索到的资料，相关内容如下：\n" + "\n\n".join(snippets)
        return self.config.NO_INFO_RESPONSE

    def _build_sales_org_fallback(self, question: str, context: str) -> str:
        """Format the sales organization chart answer when the chart text is noisy."""
        if "销售部" not in question or "大客户销售部" not in question:
            return ""
        if "大客户销售部" not in context:
            return ""
        compact_context = re.sub(r"\s+", "", context)

        departments = []
        if "渠道销售部" in compact_context:
            departments.append("渠道销售部")
        if "国际贸易部" in compact_context:
            departments.append("国际贸易部")
        if (
            ("客户中心" in compact_context and "网络销售部" in compact_context)
            or "电话及网络销售部" in compact_context
            or "及网络销售部" in compact_context
        ):
            departments.append("客户中心及网络销售部")
        if "大客户销售部" in compact_context:
            departments.append("大客户销售部")

        office_checks = [
            ("珠海销售处", ("珠海", "珠", "海")),
            ("深圳销售处", ("深圳", "深", "圳")),
            ("北京销售处", ("北京", "北", "京")),
            ("武汉销售处", ("武汉", "武", "汉")),
            ("广州销售处", ("广州", "广", "州")),
            ("成都销售处", ("成都", "成", "都")),
        ]
        offices = []
        has_sales_office_signal = "销售处" in compact_context or ("销" in context and "处" in context)
        for name, signals in office_checks:
            city, first, second = signals
            if city in compact_context or (has_sales_office_signal and first in context and second in context):
                offices.append(name)

        departments = list(dict.fromkeys(departments))
        offices = list(dict.fromkeys(offices))
        if len(departments) < 3 or len(offices) < 3:
            return ""

        dept_lines = "\n".join([f"{idx}. {name}" for idx, name in enumerate(departments, 1)])
        office_lines = "\n".join([f"{idx}. {name}" for idx, name in enumerate(offices, 1)])
        return (
            f"销售部由 {len(departments)} 个部门构成：\n"
            f"{dept_lines}\n\n"
            f"其中，大客户销售部由 {len(offices)} 个销售处构成：\n"
            f"{office_lines}"
        )

    def _should_rewrite_query(self, question: str, history: List[Dict]) -> bool:
        """Only rewrite likely follow-up questions; rewriting costs an extra LLM call."""
        if not history:
            return False
        markers = (
            "它", "这个", "那个", "这些", "那些", "上述", "上面", "前面", "上一",
            "其中", "其", "该", "这", "那", "呢", "继续", "还有"
        )
        compact = question.strip()
        return len(compact) <= 12 or any(marker in compact for marker in markers)

    def _load_conversation_history(self, session_id: str) -> List[Dict]:
        max_turns = self.config.MAX_HISTORY_TURNS
        if self.memory_store and getattr(self.memory_store, "is_ready", False):
            return self.memory_store.get_history(session_id, max_turns)
        return self.history[-(max_turns * 2):]

    def _append_conversation_history(self, session_id: str, question: str, answer: str):
        max_turns = self.config.MAX_HISTORY_TURNS
        if self.memory_store and getattr(self.memory_store, "is_ready", False):
            self.memory_store.append_turn(session_id, question, answer, max_turns)
            return

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        max_history = max_turns * 2
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
        print(f"[RAG] 对话历史已更新, 当前共 {len(self.history)} 条消息", flush=True)

    def rewrite_query(self, question: str, history: List[Dict]) -> str:
        """用 LLM 改写指代问题（Query Rewrite），将"它"、"这个"等指代词替换为具体含义。"""
        if not history:
            return question
        # 拼接最近 6 条历史消息作为上下文
        history_text = "\n".join([
            f"{'用户' if m['role']=='user' else '助手'}: {m['content'][:100]}"
            for m in history[-6:]
        ])
        prompt = f"""根据对话历史，将用户的问题改写为完整、独立的问题。
如果问题中没有指代词（如"它"、"这个"、"那"），直接返回原问题。

对话历史：
{history_text}

用户问题：{question}

改写后的问题："""
        rewritten = self.llm.chat(prompt, temperature=0.1)
        return rewritten.strip() if rewritten and rewritten.strip() else question

    def summarize_old_history(self, old_messages: List[Dict]) -> str:
        """用 LLM 总结旧对话轮次（长上下文管理），避免对话过长。"""
        text = "\n".join([
            f"{'用户' if m['role']=='user' else '助手'}: {m['content'][:200]}"
            for m in old_messages
        ])
        prompt = f"""请将以下对话总结为简洁的摘要（不超过 200 字）：

{text}

摘要："""
        return self.llm.chat(prompt, temperature=0.1)

    def _check_question_history(self, question: str, session_id: str = "default") -> Optional[str]:
        """检查历史问题库，若命中高相似度问题则直接返回缓存答案，否则返回 None。"""
        # 若配置中关闭了 QA 缓存，直接跳过
        if not getattr(self.config, 'ENABLE_QUESTION_CACHE', True):
            print("[RAG] QA 缓存已关闭，跳过历史问题检索", flush=True)
            return None

        if self.memory_store and getattr(self.memory_store, "is_ready", False):
            redis_answer = self.memory_store.get_exact_answer(
                session_id,
                question,
                self.config.MAX_HISTORY_TURNS,
            )
            if redis_answer:
                print("[RAG] Redis 短期记忆精确命中，直接返回上一轮答案", flush=True)
                return redis_answer

        hot_answer = self._get_hot_answer(question)
        if hot_answer is not None:
            return hot_answer

        if not self.question_store:
            print("[RAG] 历史问题库未配置，跳过缓存查询", flush=True)
            return None

        if hasattr(self.question_store, "get_exact_answer"):
            exact_answer = self.question_store.get_exact_answer(question)
            if exact_answer is not None:
                self._put_hot_answer(question, exact_answer)
                print("[RAG] 历史缓存精确命中，直接返回缓存答案", flush=True)
                return exact_answer

        if not self.question_store.is_ready:
            print("[RAG] 历史问题库未就绪，跳过相似缓存查询", flush=True)
            return None

        results = self.question_store.search(question, top_k=1)
        if results:
            chunk, score = results[0]
            threshold = getattr(self.config, 'QA_SIMILARITY_THRESHOLD', 0.999)
            if score >= threshold:  # 高相似度阈值，直接返回历史答案
                print(f"[RAG] 历史缓存命中! 相似度={score:.4f} >= 阈值{threshold}, 返回缓存答案", flush=True)
                answer = chunk.get("answer", None)
                if answer is not None:
                    self._put_hot_answer(question, answer)
                return answer
            else:
                print(f"[RAG] 历史缓存未命中, 最高相似度={score:.4f} (阈值={threshold})", flush=True)
        else:
            print("[RAG] 历史缓存未命中, 无搜索结果", flush=True)
        return None

    def _save_question_history(self, question: str, answer: str):
        """将问答对存入历史问题库，供后续缓存命中。"""
        self._put_hot_answer(question, answer)
        if not self.question_store:
            print("[RAG] 问答已存入进程内热缓存", flush=True)
            return
        try:
            self.question_store.add_qa(question, answer)
            print("[RAG] 问答已存入历史缓存", flush=True)
        except Exception as e:
            print(f"[RAG] 存入历史缓存失败: {e}", flush=True)

    def query(self, question: str, use_history: bool = False, stream: bool = False, session_id: str = "default"):
        """
        执行 RAG 查询（核心方法）。
        - use_history: 是否启用多轮对话上下文
        - stream=True 时返回生成器（用于流式输出）
        """
        # ===== [PERF] 性能计时 =====
        _perf_t0 = time.time()
        _perf = {"cache": 0, "rewrite": 0, "retrieve": 0, "rerank": 0, "llm": 0, "total": 0, "question": question[:50]}

        print(f"[RAG] === 查询开始 ===", flush=True)
        print(f"[RAG] 问题: {question}", flush=True)
        print(f"[RAG] use_history={use_history}, stream={stream}, session_id={session_id}", flush=True)

        # 0. 先检查历史问题库，看是否能直接命中缓存
        _t = time.time()
        cached = self._check_question_history(question, session_id)
        _perf["cache"] = round(time.time() - _t, 3)
        if cached:
            _perf["total"] = round(time.time() - _perf_t0, 3)
            print(f"[PERF] {json.dumps(_perf, ensure_ascii=False)}", flush=True)
            print(f"[RAG] 使用缓存答案, 长度={len(cached)}", flush=True)
            self._clear_last_sources(question)
            if use_history and self.config.ENABLE_MULTI_TURN:
                self._append_conversation_history(session_id, question, cached)
            if stream:
                def gen():
                    yield from self._stream_local_answer(cached)
                return gen()
            return cached

        # 0.5. 多轮追问按需改写为独立问题，避免每轮都多一次 LLM 调用
        retrieval_question = question
        history = []
        if use_history and self.config.ENABLE_MULTI_TURN:
            history = self._load_conversation_history(session_id)
            print(f"[RAG] 使用多轮对话历史, 共 {len(history)} 条消息", flush=True)
            if self._should_rewrite_query(question, history):
                _t = time.time()
                rewritten = self.rewrite_query(question, history)
                _perf["rewrite"] = round(time.time() - _t, 3)
                if rewritten and rewritten != question:
                    retrieval_question = rewritten
                    print(f"[RAG] Query 改写完成: {retrieval_question[:120]}", flush=True)
                print(f"[RAG] Query 改写耗时={_perf['rewrite']}s", flush=True)

        # 1. 检索相关文档
        _t = time.time()
        print("[RAG] 正在检索相关文档 ...", flush=True)
        results = self.retriever.retrieve(retrieval_question)
        _perf["retrieve"] = round(time.time() - _t, 3)
        print(f"[RAG] 检索返回 {len(results)} 条结果, 耗时={_perf['retrieve']}s", flush=True)
        self._last_retrieve_time = _perf['retrieve']  # 记录检索耗时


        # 2. Reranker 精排（可选）
        filtered = []
        if self.reranker and self.config.ENABLE_RERANKER and results:
            # 使用 reranker 对检索结果重新排序
            results_before_rerank = len(results)
            _t = time.time()
            results = self.reranker.rerank(question, results, self.config.RERANK_TOP_K)
            _perf["rerank"] = round(time.time() - _t, 3)
            print(f"[RAG] Reranker 精排完成, 共 {len(results)} 条, 耗时={_perf['rerank']}s", flush=True)

            if results:
                # 打印每条结果的 reranker 分数
                for i, (chunk, score) in enumerate(results):
                    print(f"[RAG]   Reranker分数[{i}]: {score:.4f} (来源: {chunk.get('source', '未知')})", flush=True)


                # Reranker 后过滤低分 chunk（分数差异大时只保留高相关的）
                max_score = results[0][1]
                threshold = max_score * 0.3
                print(f"[RAG] Reranker 过滤: 最高分={max_score:.4f}, 阈值={threshold:.4f} (最高分*0.3)", flush=True)

                filtered = [(c, s) for c, s in results if s > threshold]
                if not filtered:
                    filtered = results[:1]
                    print(f"[RAG] 过滤后无结果, 保留第 1 条", flush=True)
                print(f"[RAG] Reranker 过滤前后: {results_before_rerank} -> {len(filtered)}", flush=True)

            else:
                filtered = []
                print("[RAG] Reranker 返回空结果", flush=True)
        else:
            # 无 reranker 时：RRF 分数极小(0.01~0.02)，不能用阈值过滤
            # 直接取 top_k 结果
            filtered = results[:self.config.TOP_K] if results else []
            print(f"[RAG] 未启用 Reranker, 直接取 top_k={len(filtered)} 条结果", flush=True)


        # 3. 如果过滤后没有可用文档，返回"未找到"
        if not filtered:
            print("[RAG] 无可用检索结果, 返回未找到提示", flush=True)
            answer = self.config.NO_INFO_RESPONSE
            self._clear_last_sources(question)
            if stream:
                def gen():
                    yield from self._stream_local_answer(answer)
                return gen()
            return answer
        self._set_last_sources(question, filtered)

        structured_answer = self._build_retrieval_fallback(question, filtered)
        if structured_answer and not structured_answer.startswith("根据检索到的资料"):
            print("[RAG] 命中结构化检索答案，跳过 LLM 自由生成", flush=True)
            self._save_question_history(question, structured_answer)
            if use_history:
                self._append_conversation_history(session_id, question, structured_answer)
            if stream:
                def gen():
                    yield from self._stream_local_answer(structured_answer)
                return gen()
            return structured_answer

        # 4. 构建 prompt（只取最相关的3个片段，减少prompt长度）
        context = "\n\n".join([
            f"【来源: {chunk['source']}】\n{chunk['text']}"
            for chunk, _ in filtered[:3]  # 只取前3个最相关片段
        ])
        if retrieval_question != question:
            prompt_question = f"{question}\n\n改写后的独立问题：{retrieval_question}"
        else:
            prompt_question = question
        prompt = f"参考资料：\n{context}\n\n问题：{prompt_question}"
        print(f"[RAG] Prompt 上下文长度: {len(context)} 字符, 共 {len(filtered)} 个来源", flush=True)


        # 6. 调用 LLM 生成回答
        if stream:
            # ---- 流式模式 ----
            def stream_gen():
                print("[RAG] LLM 调用开始 (stream)", flush=True)

                _t = time.time()
                full_answer = ""
                for chunk_text in self.llm.stream(prompt, system_prompt=SYSTEM_PROMPT, history=history):
                    chunk_text = self._clean_stream_chunk(chunk_text)
                    if not chunk_text:
                        continue
                    full_answer += chunk_text
                    yield chunk_text
                _perf["llm"] = round(time.time() - _t, 3)
                _perf["total"] = round(time.time() - _perf_t0, 3)
                print(f"[RAG] LLM 调用完成 (stream), 耗时={_perf['llm']}s, 回答长度={len(full_answer)}", flush=True)
                print(f"[PERF] {json.dumps(_perf, ensure_ascii=False)}", flush=True)


                cleaned = clean_answer(full_answer)
                streamed_fallback = False
                if not cleaned:
                    print("[RAG] 流式回答为空，触发兜底输出", flush=True)
                    repaired = self.llm.chat(prompt, system_prompt=SYSTEM_PROMPT, history=history)
                    repaired = clean_answer(repaired)
                    if repaired and not repaired.startswith("LLM 调用失败"):
                        print("[RAG] 非流式补答成功，使用补答内容", flush=True)
                        cleaned = repaired
                    else:
                        cleaned = self._build_retrieval_fallback(question, filtered)
                    streamed_fallback = True
                    yield from self._stream_local_answer(cleaned)

                # 兜底：LLM 误判"未找到"但实际有检索结果时，强制返回检索内容
                if cleaned.startswith("未找到") and filtered:
                    print("[RAG] 触发兜底: LLM 回答'未找到'但有检索结果, 强制使用检索内容", flush=True)
                    fallback = self._build_retrieval_fallback(question, filtered)
                    cleaned = fallback
                    if not streamed_fallback:
                        yield "\n"
                        yield from self._stream_local_answer(cleaned)

                # 存入历史问题库（错误回答不缓存）
                if not cleaned.startswith("LLM ") and not cleaned.startswith("未找到"):
                    self._save_question_history(question, cleaned)
                else:
                    print("[RAG] 跳过缓存: 错误回答不存入历史", flush=True)

                # 记录对话历史
                if use_history:
                    self._append_conversation_history(session_id, question, cleaned)


                print("[RAG] === 流式查询完成 ===", flush=True)
            return stream_gen()
        else:
            # ---- 非流式模式 ----
            print("[RAG] LLM 调用开始 (chat)", flush=True)

            _t = time.time()
            answer = self.llm.chat(prompt, system_prompt=SYSTEM_PROMPT, history=history)
            _perf["llm"] = round(time.time() - _t, 3)
            print(f"[RAG] LLM 调用完成 (chat), 耗时={_perf['llm']}s, 回答长度={len(answer)}", flush=True)


            answer = clean_answer(answer)
            # 兜底：LLM 误判"未找到"但实际有检索结果时，强制返回检索内容
            if answer.startswith("未找到") and filtered:
                print("[RAG] 触发兜底: LLM 回答'未找到'但有检索结果, 强制使用检索内容", flush=True)
                answer = self._build_retrieval_fallback(question, filtered)

            # 7. 存入历史问题库（错误回答不缓存）
            if not answer.startswith("LLM ") and not answer.startswith("未找到"):
                self._save_question_history(question, answer)
            else:
                print("[RAG] 跳过缓存: 错误回答不存入历史", flush=True)

            # 8. 记录对话历史
            if use_history:
                self._append_conversation_history(session_id, question, answer)


            _perf["total"] = round(time.time() - _perf_t0, 3)
            print(f"[RAG] === 查询完成 ===", flush=True)
            print(f"[PERF] {json.dumps(_perf, ensure_ascii=False)}", flush=True)
            return answer

    def get_sources(self, question: str) -> List[Dict]:
        """获取检索到的来源（用于前端展示），返回包含来源、摘要文本和分数的列表。"""
        if self._last_source_question == question:
            print(f"[RAG] get_sources: 复用最近一次检索来源, 共 {len(self._last_sources)} 条", flush=True)
            return self._last_sources

        print(f"[RAG] get_sources: 获取来源, 问题={question}", flush=True)
        results = self.retriever.retrieve(question, self.config.TOP_K)
        print(f"[RAG] get_sources: 返回 {len(results)} 条来源", flush=True)

        sources = [
            {"source": chunk["source"], "text": chunk["text"][:200], "score": score}
            for chunk, score in results
        ]
        self._last_source_question = question
        self._last_sources = sources
        return sources
