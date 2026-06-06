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
