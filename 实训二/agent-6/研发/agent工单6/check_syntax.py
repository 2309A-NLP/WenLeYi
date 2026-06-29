import py_compile
import sys

files = ['config.py', 'tool_client.py', 'orchestrator.py', 'app.py']
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f'OK {f}')
    except py_compile.PyCompileError as e:
        print(f'FAIL {f}: {e}')
        sys.exit(1)
print('ALL OK')
