"""检查 Neo4j 中实际的节点和关系属性结构。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import Config
config = Config()
from neo4j import GraphDatabase
driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
with driver.session() as s:
    r = s.run('MATCH (n) RETURN labels(n) AS labels, keys(n) AS props, count(n) AS cnt ORDER BY cnt DESC LIMIT 10')
    print('=== 节点 ===')
    for rec in r:
        print(f'  labels={rec["labels"]}, props={rec["props"]}, count={rec["cnt"]}')
    r = s.run('MATCH ()-[r]->() RETURN type(r) AS rtype, keys(r) AS props, count(r) AS cnt ORDER BY cnt DESC LIMIT 10')
    print('=== 关系 ===')
    for rec in r:
        print(f'  type={rec["rtype"]}, props={rec["props"]}, count={rec["cnt"]}')
    r = s.run('MATCH (a)-[r]->(b) RETURN properties(a) AS pa, type(r) AS rt, properties(r) AS pr, properties(b) AS pb LIMIT 3')
    print('=== 样例 ===')
    for rec in r:
        print(f'  a={rec["pa"]}, rel={rec["rt"]}, r={rec["pr"]}, b={rec["pb"]}')
driver.close()
