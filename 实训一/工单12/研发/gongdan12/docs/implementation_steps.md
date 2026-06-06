# RAG 与 LightRAG 招股说明书问答实验过程

## 1. 数据与问题集

- 原始文档放在 `documents/`：
  - `招股说明书1.pdf`
  - `招股说明书2.pdf`
- 测试问题放在 `documents/test_questions.json`，已整理为验收口径的 14 个问题。

## 2. 构建传统 RAG 索引

```bash
D:\an10-1\envs\nlp_1\python.exe scripts\build_index.py --rebuild
```

传统 RAG 使用 `core/document_processor.py` 解析 PDF，使用 `core/vector_store.py` 构建向量索引，检索阶段由 `core/retriever.py` 融合 BM25 与向量检索结果。

## 3. 构建 LightRAG 知识图谱

```bash
D:\an10-1\envs\nlp_1\python.exe scripts\build_lightrag.py
```

如果 `lightrag_storage/` 中存在旧的 `processing` 状态或导出的图谱为空，使用清理重建：

```bash
D:\an10-1\envs\nlp_1\python.exe scripts\build_lightrag.py --reset
```

LightRAG 的实体类型在 `core/lightrag_engine.py` 中按招股说明书优化，包括公司名称、人物姓名、金额数字、项目名称、行业领域、部门机构、地理位置、技术标准、关联企业、募集资金、荣誉奖项。

构建产物位于 `lightrag_storage/`，其中：

- `graph_chunk_entity_relation.graphml`：LightRAG 知识图谱
- `knowledge_graph.json`：导出的实体与关系 JSON
- `vdb_entities.json`、`vdb_relationships.json`：实体与关系向量数据

## 4. 对比检索与 RAGAS 指标

```bash
D:\an10-1\envs\nlp_1\python.exe scripts\compare_rag_lightrag.py
```

如需强制重建 LightRAG 索引：

```bash
D:\an10-1\envs\nlp_1\python.exe scripts\compare_rag_lightrag.py --build-index
```

如果需要从空目录重建 LightRAG：

```bash
D:\an10-1\envs\nlp_1\python.exe scripts\compare_rag_lightrag.py --reset-lightrag --build-index
```

评估脚本会生成：

- `evaluation_results/rag_results.json`
- `evaluation_results/lightrag_results.json`
- `evaluation_results/metrics_comparison.json`
- `evaluation_results/comparison_report.md`

指标包含 Faithfulness、Answer Relevancy、Context Precision、Context Recall。当前脚本使用 LLM-as-Judge 方式实现这四项 RAGAS 口径指标。

## 5. 验收检查

1. `documents/test_questions.json` 保持 14 个测试问题。
2. `lightrag_storage/knowledge_graph.json` 能看到实体、关系数量不为 0。
3. `evaluation_results/comparison_report.md` 包含逐题 RAG 与 LightRAG 回答对比。
4. `evaluation_results/metrics_comparison.json` 包含 RAG 与 LightRAG 四项指标平均分。
