"""
ArcFace推理模块
本模块提供人脸特征提取的推理功能，用于验证训练好的模型。

使用方法：
    python inference.py --network r50 --weight model.pth --img face.jpg

功能：
1. 加载训练好的骨干网络模型
2. 预处理输入人脸图像（缩放、归一化）
3. 提取人脸特征向量
"""
import argparse

import cv2
import numpy as np
import torch

from backbones import get_model  # 从backbones模块导入模型工厂函数


@torch.no_grad()  # 推理时不需要计算梯度，节省显存
def inference(weight, name, img):
    """
    人脸特征提取推理函数
    
    参数:
        weight (str): 模型权重文件路径
        name (str): 骨干网络名称（如'r50'）
        img (str): 输入人脸图像路径，为None时使用随机图像
    """
    if img is None:
        # 如果未提供图像，使用随机图像进行测试
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)       # 读取图像
        img = cv2.resize(img, (112, 112))  # 缩放到112x112（ArcFace标准输入尺寸）

    # 图像预处理
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    img = np.transpose(img, (2, 0, 1))          # HWC -> CHW
    img = torch.from_numpy(img).unsqueeze(0).float()  # 转为Tensor并添加batch维度
    img.div_(255).sub_(0.5).div_(0.5)  # 归一化到[-1, 1]

    # 加载模型并推理
    net = get_model(name, fp16=False)  # 创建骨干网络
    net.load_state_dict(torch.load(weight))  # 加载权重
    net.eval()  # 设置为评估模式
    feat = net(img).numpy()  # 提取特征向量
    print(feat)  # 打印特征向量


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')  # 骨干网络类型
    parser.add_argument('--weight', type=str, default='')  # 模型权重路径
    parser.add_argument('--img', type=str, default=None)    # 输入图像路径
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)
