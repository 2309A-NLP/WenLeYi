# AutoDL部署指南

## 一、准备工作

### 1. 本地打包代码
在你的Windows电脑上，把以下文件打包成zip：
- `D:\桌面\gongdanzuoye\RAG测试2.0版本\finetune\` 整个目录

### 2. 准备模型文件
把 `D:\桌面\模型\m3e-base\` 整个目录打包成zip

### 3. 准备PDF文档
把 `D:\桌面\gongdanzuoye\RAG测试2.0版本\documents\` 目录打包成zip

## 二、AutoDL操作步骤

### 步骤1：创建实例
1. 登录 AutoDL (https://www.autodl.com/)
2. 创建实例，推荐配置：
   - GPU: RTX 3090 或 A100
   - 系统: Ubuntu 20.04
   - Python: 3.10+
   - 数据盘: 50GB+

### 步骤2：上传文件
使用AutoDL的文件上传功能或JupyterLab上传：
1. 上传 `finetune.zip` 到 `/root/autodl-tmp/`
2. 上传 `m3e-base.zip` 到 `/root/autodl-tmp/`
3. 上传 `documents.zip` 到 `/root/autodl-tmp/`

### 步骤3：解压文件
打开终端，执行：
```bash
cd /root/autodl-tmp
unzip finetune.zip
unzip m3e-base.zip -d models/
unzip documents.zip -d finetune/
```

### 步骤4：配置环境
```bash
cd /root/autodl-tmp/finetune
python setup_autodl.py
```

### 步骤5：设置API密钥
```bash
export MIMO_SK_KEY="你的sk-开头的密钥"
```

### 步骤6：运行微调
```bash
cd /root/autodl-tmp/finetune
python run_autodl.py
```

## 三、文件结构

上传后的目录结构应该是：
```
/root/autodl-tmp/
├── models/
│   └── m3e-base/          # 基础模型
├── finetune/
│   ├── run_autodl.py      # 主入口（AutoDL版）
│   ├── finetune_autodl.py # 微调脚本（AutoDL版）
│   ├── dataset_generator.py
│   ├── evaluate.py
│   ├── setup_autodl.py
│   └── documents/         # PDF文档（从zip解压）
└── data/                  # 数据盘
```

## 四、常见问题

### Q: 模型从哪里下载？
A: 在AutoDL终端执行：
```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('moka-ai/m3e-base', local_dir='/root/autodl-tmp/models/m3e-base')"
```

### Q: 训练中途断了怎么办？
A: 模型会定期保存到 `/root/autodl-tmp/finetune/m3e-finetuned/`，可以从断点继续。

### Q: 如何把训练好的模型下载回来？
A: 在AutoDL的文件管理界面，找到 `/root/autodl-tmp/finetune/m3e-finetuned/`，右键下载。

### Q: GPU显存不够怎么办？
A: 编辑 `finetune_autodl.py`，修改：
```python
BATCH_SIZE = 16  # 减小批量大小
```

### Q: 训练太慢怎么办？
A: 检查是否使用了GPU：
```python
import torch
print(torch.cuda.is_available())  # 应该输出True
```

## 五、把微调模型下载回本地

训练完成后，从AutoDL下载 `m3e-finetuned` 目录，放到：
- Windows: `D:\桌面\模型\m3e-finetuned\`
- 替换项目中的 m3e-base 路径即可使用
