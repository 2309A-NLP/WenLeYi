# torchalign 模块的核心API文件
# 提供 FacialLandmarkDetector 类，用于人脸关键点检测
# 该类集成了骨干网络(backbone)和热图头(heatmap_head)，实现端到端的关键点预测

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

# 导入 backbone 和 heatmap_head 子模块
from . import backbone, heatmap_head


# 定义该模块对外公开的类
__all__ = [ 'FacialLandmarkDetector' ]


class FacialLandmarkDetector(nn.Module):
    """FacialLandmarkDetector - 人脸关键点检测器
    
    该类是 torchalign 库的核心检测器，负责：
    1. 加载配置文件和预训练模型
    2. 构建骨干网络（如HRNet）和热图预测头
    3. 对输入图像进行预处理（裁剪、缩放、归一化）
    4. 通过前向传播预测人脸关键点坐标
    5. 支持水平翻转增强来提升预测精度
    """
    def __init__(self, root, pretrained=True):
        """初始化人脸关键点检测器
        
        参数:
            root: 模型目录路径，包含 config.yaml 配置文件和 model.pth 预训练权重
            pretrained: 是否加载预训练权重，默认为 True
        """
        super(FacialLandmarkDetector, self).__init__()
        # 从配置文件中加载模型配置
        self.config = self.config_from_file(os.path.join(root, 'config.yaml'))
        # 根据配置创建骨干网络（特征提取器），如 hrnet18
        self.backbone = backbone.__dict__[self.config.BACKBONE.ARCH](pretrained=False)
        # 根据配置创建热图预测头，用于将特征图转换为关键点热图
        self.heatmap_head = heatmap_head.__dict__[self.config.HEATMAP.ARCH](self.config)
        # 构建图像预处理流水线：缩放 -> 转为张量 -> ImageNet标准化
        self.transform = transforms.Compose([
            transforms.Resize(self.config.INPUT.SIZE),  # 缩放到配置指定的输入尺寸
            transforms.ToTensor(),  # 将PIL图像转为PyTorch张量，像素值归一化到[0,1]
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # ImageNet均值和标准差归一化
        ])
        if pretrained:
            # 加载预训练模型权重
            self.load_state_dict(torch.load(os.path.join(root, 'model.pth')))
        
    def config_from_file(self, filename):
        """从YAML文件加载配置
        
        参数:
            filename: 配置文件路径
        返回:
            合并了文件配置的全局配置对象
        """
        from .cfg import cfg
        if os.path.isfile(filename):
            # 将文件中的配置与默认配置合并（覆盖默认值）
            cfg.merge_from_file(filename)
        return cfg
        
    def resized_crop(self, img, bbox):
        """对图像进行裁剪和缩放预处理
        
        根据人脸边界框(bbox)裁剪图像区域，并缩放到模型输入尺寸。
        如果没有提供边界框，则使用整张图像。
        
        参数:
            img: PIL格式的输入图像
            bbox: 人脸边界框坐标 [N, 4]，格式为 [x1, y1, x2, y2]
                  如果为 None，则使用整张图像
        返回:
            data: 处理后的图像张量 [N, C, H, W]
            rect: 裁剪区域的坐标 [N, 4]
        """
        # 创建覆盖整张图像的矩形区域作为默认值
        rect = torch.Tensor([[0, 0, img.width, img.height]])
        if bbox is not None:
            # 计算边界框的宽高，取最大值并乘以缩放因子（扩大裁剪区域）
            wh = (bbox[:,2:] - bbox[:,:2] + 1).max(1)[0] * self.config.INPUT.SCALE
            # 计算中心点位置
            xy = (bbox[:,:2] + bbox[:,2:] - wh.unsqueeze(1) + 1) / 2.0
            # 构建新的裁剪矩形 [左上角x, 左上角y, 右下角x, 右下角y]
            rect = torch.cat([xy, xy+wh.unsqueeze(1)], 1)
        # 对每个裁剪区域进行预处理变换，并堆叠成批量张量
        data = torch.stack([self.transform(img.crop(x.tolist())) for x in rect])
        return data, rect
        
    def resized_crop_inverse(self, landmark, rect):
        """将归一化坐标逆变换回原始图像坐标系
        
        模型输出的关键点坐标是在裁剪后的图像坐标系中，
        需要通过逆变换映射回原始图像的坐标系。
        
        参数:
            landmark: 模型输出的关键点坐标 [N, K, 2]，K为关键点数量
            rect: 裁剪区域坐标 [N, 4]
        返回:
            映射到原始图像坐标的关键点坐标 [N, K, 2]
        """
        # 计算缩放比例：输入尺寸 / 裁剪区域尺寸
        scale = torch.stack([
            self.config.INPUT.SIZE[0] / (rect[:,2]-rect[:,0]),  # 宽度缩放比
            self.config.INPUT.SIZE[1] / (rect[:,3]-rect[:,1])   # 高度缩放比
        ]).t()
        # 逆缩放 + 加上裁剪偏移量，得到原始坐标
        return landmark / scale[:,None,:] + rect[:,None,:2]
        
    def flip_landmark(self, landmark, img_width):
        """水平翻转关键点坐标
        
        用于水平翻转增强（test-time augmentation），
        将翻转后预测的关键点坐标恢复到原始方向。
        
        参数:
            landmark: 翻转后的关键点坐标 [N, K, 2]
            img_width: 图像宽度
        返回:
            恢复到原始方向的关键点坐标
        """
        # 水平翻转x坐标
        landmark[..., 0] = img_width - 1 -landmark[...,0]
        # 按照预定义的翻转索引顺序重排关键点
        return landmark[...,self.config.INPUT.FLIP_ORDER,:]

    def forward(self, img, bbox=None, device=None):
        """前向传播 - 执行人脸关键点检测
        
        完整的检测流程：
        1. 裁剪并预处理图像
        2. 通过骨干网络提取特征
        3. 通过热图头预测关键点
        4. 可选的翻转增强
        5. 逆变换回原始坐标
        
        参数:
            img: PIL格式的输入图像
            bbox: 人脸边界框 [N, 4]，可选
            device: 目标设备（如 'cuda:0'），可选
        返回:
            landmark: 预测的关键点坐标 [N, K, 2]
        """
        # 裁剪并预处理图像
        data, rect = self.resized_crop(img, bbox)
        if device is not None:
            # 将数据移动到指定设备（GPU/CPU）
            data, rect = data.to(device), rect.to(device)
        # 骨干网络提取特征 -> 热图头预测关键点坐标
        landmark = self.heatmap_head(self.backbone(data))
        if self.config.INPUT.FLIP:
            # 水平翻转增强：对翻转后的图像也进行预测，然后取平均值
            data = data.flip(dims=[-1])
            landmark_ = self.heatmap_head(self.backbone(data))
            # 将翻转后的预测坐标翻转回来
            landmark_ = self.flip_landmark(landmark_, data.shape[-1])
            # 原始预测和翻转预测取平均，提升精度
            landmark = (landmark + landmark_) / 2.0
        # 将坐标从裁剪图像空间映射回原始图像空间
        landmark = self.resized_crop_inverse(landmark, rect)
        return landmark
