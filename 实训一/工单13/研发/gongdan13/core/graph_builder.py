"""Build a lightweight Neo4j knowledge graph from document chunks."""
import re
from collections import OrderedDict
from typing import Dict, List, Optional


class GraphBuilder:
    """Extract simple entity-relation triples from chunks and write them to Neo4j."""

    RELATIONS = (
        "负责", "使用", "用于", "属于", "包含", "包括", "支持", "提供", "管理",
        "存储", "部署", "运行", "配置", "连接", "依赖", "影响", "导致", "组成",
        "位于", "调用", "生成", "检索", "写入", "读取", "关联", "适用",
    )

    ENTITY_PATTERN = r"[\u4e00-\u9fa5A-Za-z0-9_（）()《》、·.\-]{2,40}"

    def __init__(self, graph_store, max_triples_per_chunk: int = 8):
        self.graph_store = graph_store
        self.max_triples_per_chunk = max(1, max_triples_per_chunk)
        relation_expr = "|".join(map(re.escape, self.RELATIONS))
        self._relation_pattern = re.compile(
            rf"(?P<source>{self.ENTITY_PATTERN})\s*(?P<relation>{relation_expr})\s*(?P<target>{self.ENTITY_PATTERN})"
        )
        self._colon_pattern = re.compile(
            rf"(?P<source>{self.ENTITY_PATTERN})\s*[:：]\s*(?P<target>{self.ENTITY_PATTERN})"
        )

    def _clean_entity(self, text: str) -> str:
        text = re.sub(r"\s+", "", text or "")
        text = text.strip("，。；;：:、,.()（）[]【】《》<>\"'")
        prefixes = ("其中", "因此", "所以", "并且", "同时", "以及", "根据", "对于", "关于")
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
        return text[:40]

    def _sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text or "")
        return [s.strip() for s in re.split(r"[。！？!?；;\n]", text) if s.strip()]

    def extract_triples(self, chunk: Dict) -> List[Dict]:
        text = chunk.get("text", "")
        source_doc = chunk.get("source") or chunk.get("metadata", {}).get("source", "")
        triples = []

        for sentence in self._sentences(text):
            for match in self._relation_pattern.finditer(sentence):
                source = self._clean_entity(match.group("source"))
                relation = self._clean_entity(match.group("relation"))
                target = self._clean_entity(match.group("target"))
                if self._valid_triple(source, relation, target):
                    triples.append({
                        "source": source,
                        "relation": relation,
                        "target": target,
                        "source_doc": source_doc,
                        "evidence": sentence[:500],
                    })
                if len(triples) >= self.max_triples_per_chunk:
                    return triples

            for match in self._colon_pattern.finditer(sentence):
                source = self._clean_entity(match.group("source"))
                target = self._clean_entity(match.group("target"))
                if self._valid_triple(source, "描述", target):
                    triples.append({
                        "source": source,
                        "relation": "描述",
                        "target": target,
                        "source_doc": source_doc,
                        "evidence": sentence[:500],
                    })
                if len(triples) >= self.max_triples_per_chunk:
                    return triples

        return triples

    def _valid_triple(self, source: str, relation: str, target: str) -> bool:
        if not source or not relation or not target:
            return False
        if source == target:
            return False
        if len(source) < 2 or len(target) < 2:
            return False
        if len(relation) > 12:
            return False
        return True

    def build_from_chunks(
        self,
        chunks: List[Dict],
        clear: bool = False,
        max_chunks: Optional[int] = None,
    ) -> Dict:
        if not self.graph_store or not getattr(self.graph_store, "is_ready", False):
            print("[GRAPH-BUILD] Neo4j graph store is not ready, skip graph build")
            return {"chunks": 0, "triples": 0, "inserted": 0}

        if clear:
            self.graph_store.clear_project_graph()

        selected_chunks = chunks[:max_chunks] if max_chunks else chunks
        dedup = OrderedDict()
        for chunk in selected_chunks:
            for triple in self.extract_triples(chunk):
                key = (triple["source"], triple["relation"], triple["target"])
                dedup[key] = triple

        triples = list(dedup.values())
        inserted = self.graph_store.upsert_relationships(triples)
        print(
            f"[GRAPH-BUILD] Graph build done: chunks={len(selected_chunks)}, "
            f"triples={len(triples)}, inserted={inserted}"
        )
        return {"chunks": len(selected_chunks), "triples": len(triples), "inserted": inserted}
