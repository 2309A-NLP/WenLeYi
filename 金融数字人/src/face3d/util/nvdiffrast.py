"""可微分渲染器模块

该脚本是 Deep3DFaceRecon_pytorch 的可微分渲染器。
基于 PyTorch3D 库实现网格渲染功能。
注意：当前版本缺少抗锯齿步骤。
"""
import pytorch3d.ops                # PyTorch3D 操作模块
import torch                        # PyTorch 深度学习框架
import torch.nn.functional as F     # PyTorch 函数式接口
import kornia                       # 可微分计算机视觉库
from kornia.geometry.camera import pixel2cam  # 像素到相机坐标转换
import numpy as np                  # NumPy 数值计算库
from typing import List             # 类型提示
from scipy.io import loadmat        # MATLAB 文件读取
from torch import nn                # PyTorch 神经网络模块

# PyTorch3D 渲染相关模块
from pytorch3d.structures import Meshes  # 3D 网格数据结构
from pytorch3d.renderer import (
    look_at_view_transform,         # 视图变换
    FoVPerspectiveCameras,          # 透视相机
    DirectionalLights,              # 方向光源
    RasterizationSettings,          # 光栅化设置
    MeshRenderer,                   # 网格渲染器
    MeshRasterizer,                 # 网格光栅化器
    SoftPhongShader,                # 软 Phong 着色器
    TexturesUV,                     # UV 纹理
)


class MeshRenderer(nn.Module):
    """自定义网格渲染器。

    该渲染器基于 PyTorch3D 的光栅化管线，支持：
    - 遮罩（mask）渲染：标记可见的三角面片像素
    - 深度（depth）渲染：输出每个像素的深度值
    - 特征（feature）渲染：通过重心坐标插值渲染顶点特征

    渲染管线：
    1. 3D 顶点 → 齐次坐标
    2. 透视投影 + 光栅化
    3. 生成遮罩、深度和特征图
    """

    def __init__(self,
                rasterize_fov,
                znear=0.1,
                zfar=10, 
                rasterize_size=224):
        """初始化网格渲染器。

        参数:
            rasterize_fov (float): 光栅化的视场角（度）
            znear (float): 近裁剪面距离，默认 0.1
            zfar (float): 远裁剪面距离，默认 10
            rasterize_size (int): 渲染图像的大小（像素），默认 224
        """
        super(MeshRenderer, self).__init__()

        self.rasterize_size = rasterize_size  # 渲染尺寸
        self.fov = rasterize_fov              # 视场角
        self.znear = znear                    # 近裁剪面
        self.zfar = zfar                      # 远裁剪面

        self.rasterizer = None                # 光栅化器（延迟初始化）
    
    def forward(self, vertex, tri, feat=None):
        """执行可微分渲染。

        参数:
            vertex (Tensor): 3D 顶点坐标 (B, N, 3) 或 (B, N, 4)
            tri (Tensor): 三角面片索引 (B, M, 3) 或 (M, 3)
            feat (Tensor, optional): 顶点特征 (B, N, C)，如果提供则进行特征插值渲染

        返回:
            mask (Tensor): 可见性遮罩 (B, 1, H, W)，值为 0 或 1
            depth (Tensor): 深度图 (B, 1, H, W)
            image (Tensor or None): 渲染的特征图 (B, C, H, W)，feat 为 None 时返回 None
        """
        device = vertex.device
        rsize = int(self.rasterize_size)

        # 如果顶点是3D坐标，转换为齐次坐标（添加第4维为1）
        # 同时将 x 轴取反以匹配 OpenGL 约定
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 0] = -vertex[..., 0]  # x 轴镜像

        # 延迟初始化光栅化器（在正确的 GPU 设备上）
        if self.rasterizer is None:
            self.rasterizer = MeshRasterizer()
            print("create rasterizer on device cuda:%d" % device.index)
        
        # 确保三角面片索引为 int32 类型
        tri = tri.type(torch.int32).contiguous()

        # 创建透视相机
        cameras = FoVPerspectiveCameras(
            device=device,
            fov=self.fov,       # 视场角
            znear=self.znear,   # 近裁剪面
            zfar=self.zfar,     # 远裁剪面
        )

        # 设置光栅化参数
        raster_settings = RasterizationSettings(
            image_size=rsize    # 输出图像大小
        )

        # 构建 PyTorch3D Meshes 对象
        # vertex: 取前3个坐标 (x, y, z)
        # tri: 扩展三角面片以匹配批次大小
        mesh = Meshes(vertex.contiguous()[...,:3], tri.unsqueeze(0).repeat((vertex.shape[0],1,1)))

        # 执行光栅化
        fragments = self.rasterizer(mesh, cameras=cameras, raster_settings=raster_settings)
        rast_out = fragments.pix_to_face.squeeze(-1)  # 每个像素对应的三角面片索引
        depth = fragments.zbuf                        # 深度缓冲

        # 渲染深度图
        depth = depth.permute(0, 3, 1, 2)             # 调整维度顺序为 (B, 1, H, W)
        mask = (rast_out > 0).float().unsqueeze(1)     # 可见区域遮罩（>0 表示可见）
        depth = mask * depth                            # 只保留可见区域的深度值
        

        image = None
        if feat is not None:
            # 特征插值渲染：使用重心坐标在三角面片上插值顶点特征
            attributes = feat.reshape(-1, 3)[mesh.faces_packed()]  # 获取每个面的顶点特征
            image = pytorch3d.ops.interpolate_face_attributes(
                fragments.pix_to_face,        # 像素对应的面索引
                fragments.bary_coords,        # 重心坐标
                attributes                     # 顶点特征
            )
            image = image.squeeze(-2).permute(0, 3, 1, 2)  # 调整维度
            image = mask * image  # 只保留可见区域的特征值
        
        return mask, depth, image
