# 文生图智能体 - 面部旋转+扩图

## 工单信息

- **工单编号**: 人工智能NLP-Agent 数字人项目-文生图智能体任务
- **功能**: 输入一张面部图片，生成三张不同角度的图片（右转、左转、端正），并进行扩图处理
- **技术栈**: Stable Diffusion + ControlNet + IP-Adapter + Real-ESRGAN

## 项目结构

```
agent工单3/
├── main.py              # 主程序入口
├── face_detector.py     # 面部检测和关键点提取
├── pose_generator.py    # 姿态控制图生成
├── image_generator.py   # 图像生成（IP-Adapter + ControlNet）
├── outpainter.py        # 扩图模块
├── super_resolution.py  # 超分辨率增强
├── utils.py             # 工具函数
├── requirements.txt     # 依赖包列表
└── README.md            # 本文件
```

## 环境要求

- Python >= 3.8
- CUDA >= 11.7（需要GPU）
- 内存 >= 8GB
- 显存 >= 8GB

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备输入图片

将面部图片放入输入目录：
```bash
mkdir -p /root/autodl-tmp/input
# 复制图片到该目录
cp your_face.jpg /root/autodl-tmp/input/
```

### 2. 运行程序

```bash
python main.py
```

### 3. 查看输出

输出文件保存在 `/root/autodl-tmp/output/` 目录：

```
output/
├── view_right.jpg        # 右转30°
├── view_front.jpg        # 端正
├── view_left.jpg         # 左转30°
├── expanded_right.jpg    # 右转+扩图
├── expanded_front.jpg    # 端正+扩图
├── expanded_left.jpg     # 左转+扩图
├── enhanced_right.jpg    # 右转+扩图+超分
├── enhanced_front.jpg    # 端正+扩图+超分
└── enhanced_left.jpg     # 左转+扩图+超分
```

## 技术说明

### 核心技术

1. **IP-Adapter FaceID**: 保持面部特征一致性（五官、肤色、发色）
2. **ControlNet + OpenPose**: 精确控制面部旋转角度（±30°）
3. **Stable Diffusion Inpainting**: 高质量扩图
4. **Real-ESRGAN**: 4倍超分辨率增强

### 验收标准覆盖

| 验收要求 | 技术方案 | 状态 |
|----------|----------|------|
| 面部特征保持 | IP-Adapter FaceID | ✅ |
| 角度±30°以内 | ControlNet + OpenPose | ✅ |
| 图像清晰度 | Real-ESRGAN超分 | ✅ |
| 扩图内容一致性 | Inpainting | ✅ |
| 扩图无拼接痕迹 | 渐变遮罩融合 | ✅ |
| 图像质量提升 | 超分辨率+增强 | ✅ |

## 常见问题

### Q: 显存不足怎么办？

A: 修改 `main.py` 中的配置：
```python
# 跳过超分辨率模型
pipeline.load_models(skip_sr=True)
```

### Q: 没有GPU可以运行吗？

A: 可以，但速度很慢。程序会自动检测并使用CPU：
```python
# utils.py 中的 get_device() 会自动选择
```

### Q: 如何调整旋转角度？

A: 修改 `main.py` 中的调用参数，或直接修改 `pose_generator.py` 中的角度列表。

## 更新日志

### v1.0.0 (2025-01)
- 初始版本
- 实现面部三视图生成
- 实现扩图功能
- 实现超分辨率增强

## 许可证

本项目为课程作业，仅供学习参考。
