"""LightRAG 知识图谱可视化脚本。

功能说明:
    1. 从 LightRAG 存储中读取知识图谱数据
    2. 生成交互式 HTML 可视化文件（基于 pyvis）
    3. 导出实体和关系统计信息

使用方法:
    python scripts/visualize_graph.py
    python scripts/visualize_graph.py --output knowledge_graph.html
"""

import sys
import os
import json
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from core.config import Config


def load_graph_data(storage_dir: str) -> dict:
    """从 LightRAG 存储目录加载图谱数据。"""
    entities = []
    relations = []

    # 尝试从 json_graph 子目录加载（LightRAG NetworkX 格式）
    json_graph_dir = os.path.join(storage_dir, "json_graph")
    if os.path.exists(json_graph_dir):
        for fname in os.listdir(json_graph_dir):
            fpath = os.path.join(json_graph_dir, fname)
            if fname.endswith(".json"):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # NetworkX node_link_data 格式
                    if "nodes" in data:
                        for node in data["nodes"]:
                            entities.append({
                                "id": node.get("id", ""),
                                "label": node.get("id", ""),
                                "type": node.get("entity_type", "unknown"),
                                "description": node.get("description", ""),
                            })
                    if "links" in data:
                        for link in data["links"]:
                            relations.append({
                                "source": link.get("source", ""),
                                "target": link.get("target", ""),
                                "relation": link.get("relation", ""),
                                "weight": link.get("weight", 1.0),
                            })
                except Exception as e:
                    print(f"[可视化] 加载 {fname} 失败: {e}")

    # 备选：从 knowledge_graph.json 加载
    if not entities:
        kg_path = os.path.join(storage_dir, "knowledge_graph.json")
        if os.path.exists(kg_path):
            with open(kg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entities = data.get("entities", [])
            relations = data.get("relations", [])

    # 备选：从 kv_store.json 实体数据加载
    if not entities:
        kv_path = os.path.join(storage_dir, "kv_store.json")
        if os.path.exists(kv_path):
            with open(kv_path, "r", encoding="utf-8") as f:
                kv_data = json.load(f)
            # 解析 LightRAG 的 KV 存储格式
            for key, val in kv_data.items():
                if isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        if isinstance(sub_val, dict) and "entity_type" in sub_val:
                            entities.append({
                                "id": sub_key,
                                "label": sub_key,
                                "type": sub_val.get("entity_type", "unknown"),
                                "description": sub_val.get("description", ""),
                            })

    return {"entities": entities, "relations": relations}


def generate_html_visualization(graph_data: dict, output_path: str):
    """生成交互式 HTML 知识图谱可视化。"""
    entities = graph_data["entities"]
    relations = graph_data["relations"]

    if not entities:
        print("[可视化] 无实体数据，无法生成可视化")
        return

    # 按实体类型分配颜色
    type_colors = {
        "公司名称": "#e74c3c",
        "人物姓名": "#3498db",
        "金额数字": "#2ecc71",
        "项目名称": "#f39c12",
        "行业领域": "#9b59b6",
        "部门机构": "#1abc9c",
        "地理位置": "#e67e22",
        "产品名称": "#34495e",
        "技术标准": "#95a5a6",
        "关联企业": "#c0392b",
        "Person": "#3498db",
        "Organization": "#e74c3c",
        "Location": "#e67e22",
        "Concept": "#9b59b6",
        "unknown": "#bdc3c7",
    }

    # 构建节点和边的数据
    nodes_js = []
    for e in entities:
        eid = e.get("id", e.get("name", ""))
        etype = e.get("type", "unknown")
        color = type_colors.get(etype, "#bdc3c7")
        nodes_js.append({
            "id": eid,
            "label": eid[:20],
            "title": f"{eid}\n类型: {etype}\n{e.get('description', '')[:100]}",
            "color": color,
            "size": 25,
            "font": {"size": 14, "color": "#333333"},
        })

    edges_js = []
    for r in relations:
        edges_js.append({
            "from": r.get("source", ""),
            "to": r.get("target", ""),
            "label": r.get("relation", "")[:15],
            "title": f"{r.get('source', '')} --{r.get('relation', '')}--> {r.get('target', '')}",
            "arrows": "to",
            "color": {"color": "#888888"},
            "font": {"size": 10, "align": "middle"},
        })

    # 统计信息
    type_counts = {}
    for e in entities:
        t = e.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    stats_html = ""
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        color = type_colors.get(t, "#bdc3c7")
        stats_html += f'<span style="display:inline-block;margin:2px 6px;padding:2px 8px;background:{color};color:white;border-radius:4px;font-size:12px;">{t}: {c}</span>'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LightRAG 知识图谱可视化</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 0; background: #1a1a2e; color: #eee; }}
        #header {{ padding: 15px 20px; background: #16213e; border-bottom: 2px solid #0f3460; }}
        #header h1 {{ margin: 0; font-size: 20px; color: #e94560; }}
        #header .subtitle {{ font-size: 13px; color: #aaa; margin-top: 4px; }}
        #stats {{ padding: 8px 20px; background: #16213e; border-bottom: 1px solid #0f3460; font-size: 12px; }}
        #graph {{ width: 100%; height: calc(100vh - 120px); }}
        #legend {{ position: absolute; bottom: 20px; left: 20px; background: rgba(22,33,62,0.9); padding: 12px; border-radius: 8px; border: 1px solid #0f3460; }}
        #legend h3 {{ margin: 0 0 8px 0; font-size: 13px; color: #e94560; }}
        .legend-item {{ display: flex; align-items: center; margin: 3px 0; font-size: 11px; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>LightRAG 知识图谱可视化</h1>
        <div class="subtitle">基于招股说明书构建 | 实体: {len(entities)} | 关系: {len(relations)}</div>
    </div>
    <div id="stats">{stats_html}</div>
    <div id="graph"></div>
    <div id="legend">
        <h3>实体类型图例</h3>
        {''.join(f'<div class="legend-item"><div class="legend-dot" style="background:{type_colors.get(t, "#bdc3c7")}"></div>{t} ({c})</div>' for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))}
    </div>
    <script>
        var nodes = new vis.DataSet({json.dumps(nodes_js, ensure_ascii=False)});
        var edges = new vis.DataSet({json.dumps(edges_js, ensure_ascii=False)});
        var container = document.getElementById('graph');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            physics: {{
                enabled: true,
                barnesHut: {{ gravitationalConstant: -3000, centralGravity: 0.3, springLength: 150 }},
                stabilization: {{ iterations: 200 }}
            }},
            interaction: {{ hover: true, tooltipDelay: 200, zoomView: true, dragView: true }},
            nodes: {{ shape: 'dot', borderWidth: 2 }},
            edges: {{ smooth: {{ type: 'continuous' }}, width: 1.5 }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[可视化] HTML 文件已生成: {output_path}")


def export_statistics(graph_data: dict, output_dir: str):
    """导出图谱统计信息。"""
    entities = graph_data["entities"]
    relations = graph_data["relations"]

    # 实体类型统计
    type_counts = {}
    for e in entities:
        t = e.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    # 关系类型统计
    rel_counts = {}
    for r in relations:
        rel = r.get("relation", "unknown")
        rel_counts[rel] = rel_counts.get(rel, 0) + 1

    # 度数最高的实体（中心节点）
    degree = {}
    for r in relations:
        src = r.get("source", "")
        tgt = r.get("target", "")
        degree[src] = degree.get(src, 0) + 1
        degree[tgt] = degree.get(tgt, 0) + 1
    top_nodes = sorted(degree.items(), key=lambda x: -x[1])[:20]

    stats = {
        "total_entities": len(entities),
        "total_relations": len(relations),
        "entity_type_distribution": type_counts,
        "relation_type_distribution": rel_counts,
        "top_central_nodes": top_nodes,
    }

    stats_path = os.path.join(output_dir, "graph_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[统计] 图谱统计信息:")
    print(f"  实体总数: {len(entities)}")
    print(f"  关系总数: {len(relations)}")
    print(f"  实体类型分布: {type_counts}")
    print(f"  关系类型分布: {rel_counts}")
    print(f"  中心节点 Top5: {top_nodes[:5]}")
    print(f"  统计文件: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="LightRAG 知识图谱可视化")
    parser.add_argument("--storage-dir", type=str, default=None, help="LightRAG 存储目录")
    parser.add_argument("--output", type=str, default="knowledge_graph.html", help="输出 HTML 文件名")
    args = parser.parse_args()

    config = Config()
    storage_dir = args.storage_dir or os.path.join(str(PROJECT_DIR), "lightrag_storage")
    output_html = os.path.join(str(PROJECT_DIR), args.output)
    output_dir = os.path.join(str(PROJECT_DIR), "evaluation_results")

    print(f"[可视化] LightRAG 存储目录: {storage_dir}")

    # 加载图谱数据
    graph_data = load_graph_data(storage_dir)
    print(f"[可视化] 加载完成: {len(graph_data['entities'])} 实体, {len(graph_data['relations'])} 关系")

    if not graph_data["entities"]:
        print("[可视化] 未找到图谱数据。请先运行: python scripts/compare_rag_lightrag.py --build-index")
        return

    # 生成 HTML 可视化
    generate_html_visualization(graph_data, output_html)

    # 导出统计信息
    os.makedirs(output_dir, exist_ok=True)
    export_statistics(graph_data, output_dir)

    print(f"\n[可视化] 完成! 用浏览器打开 {output_html} 查看交互式图谱")


if __name__ == "__main__":
    main()
