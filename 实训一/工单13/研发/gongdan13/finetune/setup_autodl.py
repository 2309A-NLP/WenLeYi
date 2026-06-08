"""AutoDL环境部署脚本。

在AutoDL实例上运行此脚本，自动配置训练环境。

使用方法:
    python setup_autodl.py
"""

import os
import subprocess
import sys


def run_cmd(cmd, desc=""):
    """运行命令并打印结果。"""
    print(f"\n>>> {desc}")
    print(f"    {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    失败: {result.stderr}")
        return False
    print(f"    成功")
    return True


def setup_autodl():
    """配置AutoDL训练环境。"""
    print("=" * 60)
    print("AutoDL训练环境配置")
    print("=" * 60)

    # 1. 更新pip
    run_cmd("pip install --upgrade pip", "更新pip")

    # 2. 安装依赖
    deps = [
        "sentence-transformers>=2.2",
        "datasets",
        "torch>=2.0",
        "pymupdf",
        "numpy",
        "tqdm",
    ]
    for dep in deps:
        run_cmd(f"pip install {dep}", f"安装 {dep}")

    # 3. 创建目录结构
    dirs = [
        "/root/autodl-tmp/models",
        "/root/autodl-tmp/finetune",
        "/root/autodl-tmp/data",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"创建目录: {d}")

    # 4. 检查GPU
    print("\n" + "=" * 40)
    print("环境检查")
    print("=" * 40)

    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("警告: GPU不可用")

    print("\n环境配置完成！")
    print("\n下一步:")
    print("1. 上传m3e-base模型到 /root/autodl-tmp/models/m3e-base/")
    print("2. 上传 finetune/ 目录到 /root/autodl-tmp/finetune/")
    print("3. 运行: python run_autodl.py")


if __name__ == "__main__":
    setup_autodl()
