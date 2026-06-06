"""修复 lightrag 1.3.9 的 pipeline_status history_messages KeyError。
直接在 lightrag.py 里加安全检查。
运行方式: D:\an10-1\envs\nlp_1\python.exe scripts\fix_lightrag_bug.py
"""
import os
import sys

site_dir = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
lightrag_py = os.path.join(site_dir, "lightrag", "lightrag.py")

if not os.path.exists(lightrag_py):
    print(f"找不到: {lightrag_py}")
    sys.exit(1)

with open(lightrag_py, "r", encoding="utf-8") as f:
    lines = f.readlines()

fixed = 0
for i, line in enumerate(lines):
    # 修复所有 pipeline_status["history_messages"] 的直接访问
    # 在 .append() 和 del 之前加安全检查
    stripped = line.lstrip()
    indent = line[:len(line) - len(stripped)]

    # 修复 del pipeline_status["history_messages"][:]
    if 'del pipeline_status["history_messages"][:]' in stripped:
        lines[i] = f'{indent}if "history_messages" in pipeline_status:\n{indent}    del pipeline_status["history_messages"][:]\n'
        fixed += 1

    # 修复 pipeline_status["history_messages"].append(...)
    elif 'pipeline_status["history_messages"].append(' in stripped:
        # 检查上一行是否已有安全检查
        if i > 0 and 'history_messages' in lines[i-1] and 'if' in lines[i-1]:
            continue
        lines[i] = f'{indent}if "history_messages" in pipeline_status:\n{indent}    {stripped}'
        fixed += 1

    # 修复 pipeline_status["history_messages"][:-5000]
    elif 'pipeline_status["history_messages"][:-5000]' in stripped:
        lines[i] = f'{indent}if "history_messages" in pipeline_status:\n{indent}    {stripped}'
        fixed += 1

    # 修复 len(pipeline_status["history_messages"])
    elif 'len(pipeline_status["history_messages"])' in stripped and 'if' not in stripped:
        lines[i] = f'{indent}if "history_messages" in pipeline_status:\n{indent}    {stripped}'
        fixed += 1

with open(lightrag_py, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"修复完成: {fixed} 处")
print(f"文件: {lightrag_py}")
