"""
查看长期记忆（Milvus）工具脚本。

用途：
- 查看 long_term_memory 集合中的最近记录
- 支持按 user_id / role_id 过滤
"""

from __future__ import annotations

import argparse
import os
from typing import List, Dict, Any

from pymilvus import connections, Collection
from dotenv import load_dotenv


load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.18.128")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))
MEMORY_COLLECTION = "long_term_memory"


def connect_milvus() -> None:
    """连接 Milvus。"""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)


def fetch_memories(limit: int = 20, user_id: int | None = None, role_id: int | None = None) -> List[Dict[str, Any]]:
    """查询长期记忆记录。"""
    col = Collection(MEMORY_COLLECTION)

    expr_parts = []
    if user_id is not None:
        expr_parts.append(f"user_id == {user_id}")
    if role_id is not None:
        expr_parts.append(f"role_id == {role_id}")
    expr = " && ".join(expr_parts) if expr_parts else "id >= 0"

    rows = col.query(
        expr=expr,
        output_fields=["id", "user_id", "role_id", "memory_text", "summary", "timestamp"],
        limit=limit,
    )
    # 按时间倒序展示
    rows.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
    return rows


def print_memories(rows: List[Dict[str, Any]]) -> None:
    """格式化打印查询结果。"""
    if not rows:
        print("未查询到长期记忆记录。")
        return

    print(f"共查询到 {len(rows)} 条长期记忆：\n")
    for i, row in enumerate(rows, start=1):
        print(f"[{i}] id={row.get('id')} user={row.get('user_id')} role={row.get('role_id')} ts={row.get('timestamp')}")
        summary = (row.get("summary") or "").strip()
        memory = (row.get("memory_text") or "").strip()
        if summary:
            print(f"summary: {summary}")
        print(f"memory: {memory[:500]}{'...' if len(memory) > 500 else ''}")
        print("-" * 80)


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="查看 Milvus 长期记忆")
    parser.add_argument("--limit", type=int, default=20, help="最多返回记录数")
    parser.add_argument("--user-id", type=int, default=None, help="按 user_id 过滤")
    parser.add_argument("--role-id", type=int, default=None, help="按 role_id 过滤")
    args = parser.parse_args()

    connect_milvus()
    rows = fetch_memories(limit=args.limit, user_id=args.user_id, role_id=args.role_id)
    print_memories(rows)


if __name__ == "__main__":
    main()
