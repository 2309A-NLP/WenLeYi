"""从 Neo4j 导出知识图谱并生成可视化 HTML（带分类、点击查看详情）。"""
import os, sys, json, argparse, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import Config

# 实体类型分类规则
ENTITY_RULES = [
    ("公司", ["公司", "集团", "银行", "保险", "证券", "基金", "科技", "股份", "有限", "控股", "投资"]),
    ("金融产品", ["贷款", "存款", "理财", "债券", "基金", "保险", "信用卡", "融资"]),
    ("财务指标", ["收入", "利润", "资产", "负债", "增长", "率", "比", "总额", "净利", "营收", "拨备", "覆盖"]),
    ("业务领域", ["零售", "对公", "同业", "资金", "渠道", "网银", "支付", "信贷", "风控"]),
    ("地区", ["北京", "上海", "深圳", "广州", "珠海", "武汉", "成都", "杭州", "南京", "重庆"]),
    ("部门", ["部", "处", "中心", "委员会", "董事会", "监事会"]),
    ("风险", ["风险", "不良", "逾期", "坏账", "违约", "合规", "监管"]),
    ("技术", ["系统", "平台", "数据", "模型", "算法", "人工智能", "大数据", "区块链"]),
]

TYPE_COLORS = {
    "公司": "#e74c3c",
    "金融产品": "#3498db",
    "财务指标": "#f39c12",
    "业务领域": "#2ecc71",
    "地区": "#9b59b6",
    "部门": "#e91e63",
    "风险": "#ff5722",
    "技术": "#00bcd4",
    "其他": "#607d8b",
}


def classify_entity(name):
    """根据实体名称关键词判断类型。"""
    for etype, keywords in ENTITY_RULES:
        for kw in keywords:
            if kw in name:
                return etype
    return "其他"


def export_graph(config):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

    nodes = []
    edges = []
    seen_nodes = set()

    with driver.session() as s:
        result = s.run("""
            MATCH (a)-[r]->(b)
            RETURN
                coalesce(a.name, toString(id(a))) AS source,
                labels(a) AS source_labels,
                coalesce(b.name, toString(id(b))) AS target,
                labels(b) AS target_labels,
                type(r) AS rel_type,
                coalesce(r.relation, type(r)) AS relation,
                coalesce(r.evidence, '') AS evidence,
                coalesce(r.source_doc, '') AS source_doc
        """)
        for rec in result:
            src, tgt = rec["source"], rec["target"]
            if src not in seen_nodes:
                seen_nodes.add(src)
                nodes.append({"id": src, "label": src, "type": classify_entity(src)})
            if tgt not in seen_nodes:
                seen_nodes.add(tgt)
                nodes.append({"id": tgt, "label": tgt, "type": classify_entity(tgt)})
            edges.append({
                "from": src, "to": tgt,
                "label": rec["relation"],
                "evidence": (rec["evidence"] or "")[:300],
                "source_doc": rec["source_doc"] or "",
            })

    driver.close()
    print(f"导出: {len(nodes)} 节点, {len(edges)} 关系")
    return nodes, edges


def generate_html(nodes, edges, output_path):
    # 统计各类型数量
    type_counts = {}
    for n in nodes:
        t = n["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    vis_nodes = [{
        "id": n["id"], "label": n["label"], "group": n["type"],
        "color": {"background": TYPE_COLORS.get(n["type"], "#607d8b"),
                  "border": TYPE_COLORS.get(n["type"], "#607d8b"),
                  "highlight": {"background": "#fff", "border": "#e94560"}},
        "font": {"size": 12, "color": "#ccc", "face": "Microsoft YaHei"},
        "shape": "dot", "size": 18,
    } for n in nodes]

    vis_edges = [{
        "from": e["from"], "to": e["to"], "label": e["label"],
        "arrows": "to",
        "font": {"size": 9, "align": "middle", "color": "#888", "face": "Microsoft YaHei"},
        "color": {"color": "#555", "highlight": "#e94560", "hover": "#aaa"},
        "smooth": {"type": "continuous"},
        "evidence": e.get("evidence", ""),
        "source_doc": e.get("source_doc", ""),
    } for e in edges]

    legend = "".join([
        f'<span class="legend-item"><span class="dot" style="background:{TYPE_COLORS.get(t,"#607d8b")}"></span>{t}({c})</span>'
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1])
    ])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>知识图谱 - Graph RAG</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Microsoft YaHei',Arial,sans-serif;background:#0f0f23;color:#eee;}}
#header{{background:linear-gradient(135deg,#16213e,#0f3460);padding:12px 20px;display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #e94560;}}
#header h1{{font-size:18px;color:#fff;}}
#header .stats{{font-size:12px;color:#aaa;}}
#legend{{background:#16213e;padding:8px 20px;border-bottom:1px solid #333;font-size:12px;display:flex;flex-wrap:wrap;gap:4px;}}
.legend-item{{display:inline-flex;align-items:center;margin:2px 8px;cursor:pointer;opacity:0.9;}}
.legend-item:hover{{opacity:1;}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;}}
#container{{display:flex;height:calc(100vh - 90px);}}
#graph{{flex:1;}}
#sidebar{{width:320px;background:#16213e;border-left:1px solid #333;overflow-y:auto;padding:16px;display:none;}}
#sidebar.active{{display:block;}}
#sidebar h3{{color:#e94560;margin-bottom:10px;font-size:14px;}}
#sidebar .close-btn{{float:right;cursor:pointer;color:#888;font-size:18px;}}
#sidebar .close-btn:hover{{color:#fff;}}
.detail-section{{margin-bottom:12px;}}
.detail-section .label{{color:#888;font-size:11px;margin-bottom:2px;}}
.detail-section .value{{color:#eee;font-size:13px;line-height:1.5;}}
.relation-item{{background:#1a1a3e;border-radius:6px;padding:8px 10px;margin-bottom:6px;border-left:3px solid #e94560;}}
.relation-item .rel-name{{color:#e94560;font-size:13px;font-weight:bold;}}
.relation-item .rel-target{{color:#3498db;font-size:12px;}}
.relation-item .rel-evidence{{color:#888;font-size:11px;margin-top:4px;line-height:1.4;}}
.relation-item .rel-doc{{color:#666;font-size:10px;margin-top:2px;}}
#search{{background:#1a1a3e;border:1px solid #333;color:#eee;padding:6px 12px;border-radius:4px;font-size:12px;width:200px;}}
#search::placeholder{{color:#666;}}
.vis-tooltip{{background:#16213e!important;color:#eee!important;border:1px solid #e94560!important;border-radius:6px!important;padding:8px 12px!important;font-size:12px!important;max-width:350px!important;white-space:pre-wrap!important;}}
</style>
</head>
<body>
<div id="header">
  <h1>📊 知识图谱可视化</h1>
  <div style="display:flex;align-items:center;gap:12px;">
    <input id="search" placeholder="🔍 搜索节点..." />
    <div class="stats">{len(nodes)} 节点 | {len(edges)} 关系</div>
  </div>
</div>
<div id="legend">{legend}</div>
<div id="container">
  <div id="graph"></div>
  <div id="sidebar">
    <span class="close-btn" onclick="closeSidebar()">✕</span>
    <h3 id="sb-title">节点详情</h3>
    <div id="sb-content"></div>
  </div>
</div>
<script>
var allNodes = {json.dumps(vis_nodes, ensure_ascii=False)};
var allEdges = {json.dumps(vis_edges, ensure_ascii=False)};
var nodes = new vis.DataSet(allNodes);
var edges = new vis.DataSet(allEdges);
var container = document.getElementById('graph');
var network = new vis.Network(container, {{nodes:nodes, edges:edges}}, {{
  physics: {{
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{gravitationalConstant:-60,centralGravity:0.008,springLength:120,springConstant:0.06,damping:0.4}},
    stabilization: {{iterations:150}},
  }},
  interaction: {{hover:true,tooltipDelay:200,zoomView:true,dragView:true,navigationButtons:true,keyboard:true}},
  nodes: {{font:{{face:'Microsoft YaHei'}},borderWidth:2}},
  edges: {{font:{{face:'Microsoft YaHei',strokeWidth:0}},smooth:{{type:'continuous'}},width:1}},
}});

// 点击节点 → 高亮邻居 + 侧边栏显示详情
network.on("click", function(params) {{
  if (params.nodes.length > 0) {{
    var nodeId = params.nodes[0];
    highlightNode(nodeId);
    showSidebar(nodeId);
  }} else {{
    resetHighlight();
    closeSidebar();
  }}
}});

function highlightNode(nodeId) {{
  var connectedNodes = new Set([nodeId]);
  var connectedEdges = new Set();
  allEdges.forEach(function(e, i) {{
    if (e.from === nodeId || e.to === nodeId) {{
      connectedNodes.add(e.from);
      connectedNodes.add(e.to);
      connectedEdges.add(i);
    }}
  }});
  // 淡化未连接的节点
  var updates = [];
  allNodes.forEach(function(n) {{
    var dim = !connectedNodes.has(n.id);
    updates.push({{id:n.id, opacity: dim?0.1:1, font:{{size: n.id===nodeId?16:12, color: n.id===nodeId?'#fff':'#ccc'}}}});
  }});
  nodes.update(updates);
  // 高亮连接边
  var edgeUpdates = [];
  allEdges.forEach(function(e, i) {{
    edgeUpdates.push({{id:i, color:{{color: connectedEdges.has(i)?'#e94560':'#222'}}, width: connectedEdges.has(i)?2:0.5}});
  }});
  edges.update(edgeUpdates);
}}

function resetHighlight() {{
  var updates = allNodes.map(function(n) {{return {{id:n.id, opacity:1, font:{{size:12,color:'#ccc'}}}};}});
  nodes.update(updates);
  var edgeUpdates = allEdges.map(function(e,i) {{return {{id:i, color:{{color:'#555'}}, width:1}};}});
  edges.update(edgeUpdates);
}}

function showSidebar(nodeId) {{
  var sb = document.getElementById('sidebar');
  sb.classList.add('active');
  document.getElementById('sb-title').textContent = nodeId;
  // 找出所有相关关系
  var related = [];
  allEdges.forEach(function(e) {{
    if (e.from === nodeId || e.to === nodeId) {{
      related.push(e);
    }}
  }});
  var html = '<div class="detail-section"><div class="label">连接数</div><div class="value">' + related.length + ' 条关系</div></div>';
  html += '<div class="detail-section"><div class="label">相关关系</div>';
  related.forEach(function(e) {{
    var dir = e.from === nodeId ? '→ ' + e.to : '← ' + e.from;
    var evi = e.evidence ? '<div class="rel-evidence">' + e.evidence + '</div>' : '';
    var doc = e.source_doc ? '<div class="rel-doc">📄 ' + e.source_doc + '</div>' : '';
    html += '<div class="relation-item"><div class="rel-name">' + e.label + '</div><div class="rel-target">' + dir + '</div>' + evi + doc + '</div>';
  }});
  html += '</div>';
  document.getElementById('sb-content').innerHTML = html;
}}

function closeSidebar() {{
  document.getElementById('sidebar').classList.remove('active');
}}

// 搜索功能
document.getElementById('search').addEventListener('input', function(e) {{
  var keyword = e.target.value.toLowerCase();
  if (!keyword) {{ resetHighlight(); return; }}
  var matchIds = new Set();
  allNodes.forEach(function(n) {{
    if (n.label.toLowerCase().indexOf(keyword) >= 0) matchIds.add(n.id);
  }});
  var updates = allNodes.map(function(n) {{
    return {{id:n.id, opacity: matchIds.has(n.id)?1:0.05, font:{{size: matchIds.has(n.id)?16:10}}}};
  }});
  nodes.update(updates);
}});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"可视化页面已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Neo4j 知识图谱可视化")
    parser.add_argument("-o", "--output", default="", help="输出 HTML 路径")
    args = parser.parse_args()
    config = Config()
    if not config.ENABLE_NEO4J:
        print("[ERROR] Neo4j 未启用")
        sys.exit(1)
    output = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "graph.html"
    )
    nodes, edges = export_graph(config)
    if not nodes:
        print("[WARN] Neo4j 中没有数据")
        sys.exit(1)
    generate_html(nodes, edges, output)


if __name__ == "__main__":
    main()
