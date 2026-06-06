"""修复 Python 3.9 下所有包的 X | Y 类型注解兼容问题。
运行方式: D:\an10-1\envs\nlp_1\python.exe scripts\patch_lightrag_39.py
"""
import os
import sys
import subprocess

site_dir = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
MARKER = "from __future__ import annotations"

# 需要修复的包
PACKAGES = ["lightrag", "ascii_colors", "pipmaster"]

print(f"site-packages: {site_dir}\n")

# [1] 重新安装恢复原始文件
print("[1/2] 重新安装依赖恢复原始文件 ...")
for pkg in ["lightrag-hku", "ascii_colors", "pipmaster"]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", pkg],
        capture_output=True, text=True, timeout=120
    )
    status = "OK" if result.returncode == 0 else f"FAIL: {result.stderr[:100]}"
    print(f"  {pkg}: {status}")

# [2] 扫描所有 .py 文件，加 future import
print("\n[2/2] 扫描并修复所有包的 .py 文件 ...")

# 找到所有需要修复的包目录
pkg_dirs = []
for pkg_name in PACKAGES:
    for candidate in [pkg_name, pkg_name.replace("-", "_")]:
        d = os.path.join(site_dir, candidate)
        if os.path.isdir(d):
            pkg_dirs.append((pkg_name, d))
            break

# 也扫描 site-packages 根目录下的单文件 .py
# （有些包是直接一个 .py 文件）

fixed = 0
skipped = 0

for pkg_name, pkg_dir in pkg_dirs:
    print(f"\n  === {pkg_name} ({pkg_dir}) ===")
    for root, dirs, files in os.walk(pkg_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            if MARKER in content:
                skipped += 1
                continue

            # 检查文件是否有 X | Y 语法（简单启发式）
            import re
            has_pipe_type = bool(re.search(r'\b(str|int|float|bool|list|dict|set|tuple|Any|None)\s*\|\s*(str|int|float|bool|list|dict|set|tuple|Any|None)', content))
            has_type_hint = bool(re.search(r'def\s+\w+\(.*:\s*\w+', content))

            if not has_pipe_type and not has_type_hint:
                skipped += 1
                continue

            # 在文件开头插入 future import
            lines = content.split("\n")
            pos = 0
            if lines and lines[0].startswith("#!"):
                pos = 1
            if pos < len(lines) and "coding" in lines[pos].lower():
                pos += 1

            lines.insert(pos, MARKER)
            new_content = "\n".join(lines)

            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_content)
            fixed += 1
            rel = os.path.relpath(fpath, site_dir)
            print(f"    修复: {rel}")

print(f"\n修复完成: {fixed} 个文件已修复, {skipped} 个已跳过")
print("\n现在运行: python scripts\\compare_rag_lightrag.py --build-index")
