"""Optional Neo4j graph retrieval backend."""
from typing import Dict, List


class Neo4jGraphStore:
    """Read lightweight relationship triples from Neo4j for graph augmentation."""

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._enabled = False

        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            self._enabled = True
            print(f"[GRAPH] Neo4jGraphStore ready: {uri}")
        except Exception as e:
            print(f"[GRAPH] Neo4jGraphStore disabled: {e}")

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._driver is not None

    def search_entities(self, query: str, limit: int = 5) -> List[Dict]:
        """单跳检索：匹配查询中出现的实体关系三元组。"""
        if not self.is_ready or not query:
            return []

        cypher = """
        MATCH (a)-[r]->(b)
        WITH
            coalesce(a.name, a.title, a.id, toString(id(a))) AS source,
            coalesce(r.relation, type(r)) AS relation,
            coalesce(b.name, b.title, b.id, toString(id(b))) AS target,
            coalesce(r.evidence, '') AS evidence,
            coalesce(r.source_doc, '') AS source_doc,
            (
                coalesce(a.name, '') + ' ' +
                coalesce(a.title, '') + ' ' +
                coalesce(a.text, '') + ' ' +
                coalesce(a.content, '') + ' ' +
                coalesce(b.name, '') + ' ' +
                coalesce(b.title, '') + ' ' +
                coalesce(b.text, '') + ' ' +
                coalesce(b.content, '') + ' ' +
                coalesce(r.relation, type(r)) + ' ' +
                coalesce(r.evidence, '') + ' ' +
                coalesce(r.source_doc, '')
            ) AS haystack
        WHERE toLower(haystack) CONTAINS toLower($query)
        RETURN source, relation, target, evidence, source_doc
        LIMIT $limit
        """
        try:
            with self._driver.session() as session:
                rows = session.run(cypher, query=query, limit=int(limit))
                return [
                    {
                        "source": row["source"],
                        "relation": row["relation"],
                        "target": row["target"],
                        "evidence": row["evidence"],
                        "source_doc": row["source_doc"],
                    }
                    for row in rows
                ]
        except Exception as e:
            print(f"[GRAPH] Neo4j search failed: {e}")
            return []

    def search_entities_multihop(self, query: str, limit: int = 5, hops: int = 2) -> List[Dict]:
        """多跳检索：先匹配查询中出现的实体，再沿关系展开 1~2 跳邻居。

        流程:
          1. 单跳匹配：找出与查询直接相关的三元组
          2. 提取这些三元组中的实体名（source / target）
          3. 沿关系展开这些实体的邻居（最多 hops 跳）
          4. 去重合并，返回完整结果列表
        """
        if not self.is_ready or not query:
            return []

        # 第一步：单跳匹配（与 search_entities 相同逻辑）
        first_hop = self.search_entities(query, limit=limit)
        if not first_hop:
            return []

        # 提取第一跳中出现的所有实体名
        seed_entities = set()
        for triple in first_hop:
            seed_entities.add(triple.get("source", ""))
            seed_entities.add(triple.get("target", ""))
        seed_entities.discard("")

        if not seed_entities or hops < 2:
            return first_hop

        # 第二步：沿种子实体展开第二跳
        second_hop = []
        try:
            with self._driver.session() as session:
                for entity_name in seed_entities:
                    cypher_2hop = """
                    MATCH (a)-[r1]->(b)-[r2]->(c)
                    WHERE a.name = $entity_name
                    RETURN
                        coalesce(a.name, '') AS e1,
                        coalesce(r1.relation, type(r1)) AS rel1,
                        coalesce(b.name, '') AS e2,
                        coalesce(r2.relation, type(r2)) AS rel2,
                        coalesce(c.name, '') AS e3,
                        coalesce(r1.evidence, '') AS evidence1,
                        coalesce(r2.evidence, '') AS evidence2,
                        coalesce(r1.source_doc, '') AS source_doc
                    LIMIT 10
                    """
                    rows = session.run(cypher_2hop, entity_name=entity_name)
                    for row in rows:
                        e2 = row["e2"]
                        e3 = row["e3"]
                        if not e2 or not e3:
                            continue
                        # 构造两条路径的文本
                        path1 = f"{row['e1']} {row['rel1']} {e2}"
                        path2 = f"{e2} {row['rel2']} {e3}"
                        evidence = row["evidence2"] or row["evidence1"] or ""
                        second_hop.append({
                            "source": e2,
                            "relation": row["rel2"],
                            "target": e3,
                            "evidence": evidence[:500],
                            "source_doc": row["source_doc"] or "",
                            "path": f"{path1} -> {path2}",
                        })
                        # 同时添加原始三元组
                        second_hop.append({
                            "source": row["e1"],
                            "relation": row["rel1"],
                            "target": e2,
                            "evidence": row["evidence1"][:500] if row["evidence1"] else "",
                            "source_doc": row["source_doc"] or "",
                        })
        except Exception as e:
            print(f"[GRAPH] 多跳检索异常: {e}")

        # 合并去重
        seen = set()
        merged = []
        for triple in first_hop + second_hop:
            key = (triple.get("source", ""), triple.get("relation", ""), triple.get("target", ""))
            if key not in seen:
                seen.add(key)
                merged.append(triple)

        print(f"[GRAPH] 多跳检索: 第一跳={len(first_hop)}, 第二跳={len(second_hop)}, 去重后={len(merged)}")
        return merged

    def clear_project_graph(self):
        """Delete graph nodes created by this RAG project."""
        if not self.is_ready:
            return
        cypher = """
        MATCH (n:RAGEntity)
        DETACH DELETE n
        """
        try:
            with self._driver.session() as session:
                session.run(cypher)
            print("[GRAPH] Cleared RAG graph")
        except Exception as e:
            print(f"[GRAPH] Clear graph failed: {e}")

    def upsert_relationships(self, triples: List[Dict]) -> int:
        """Merge extracted triples into Neo4j."""
        if not self.is_ready or not triples:
            return 0

        cypher = """
        UNWIND $rows AS row
        MERGE (s:RAGEntity {name: row.source})
        MERGE (t:RAGEntity {name: row.target})
        MERGE (s)-[r:RAG_RELATION {relation: row.relation}]->(t)
        SET
            s.updated_at = timestamp(),
            t.updated_at = timestamp(),
            r.source_doc = row.source_doc,
            r.evidence = row.evidence,
            r.updated_at = timestamp()
        """
        rows = [
            {
                "source": t.get("source", "").strip(),
                "relation": t.get("relation", "").strip(),
                "target": t.get("target", "").strip(),
                "source_doc": t.get("source_doc", "").strip(),
                "evidence": t.get("evidence", "").strip()[:500],
            }
            for t in triples
            if t.get("source") and t.get("relation") and t.get("target")
        ]
        if not rows:
            return 0

        try:
            with self._driver.session() as session:
                session.run(cypher, rows=rows)
            print(f"[GRAPH] Upserted relationships: {len(rows)}")
            return len(rows)
        except Exception as e:
            print(f"[GRAPH] Upsert relationships failed: {e}")
            return 0

    def close(self):
        if self._driver:
            self._driver.close()
