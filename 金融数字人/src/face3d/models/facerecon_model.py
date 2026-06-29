"""Deep3DFaceRecon_pytorch的面部重建模型定义脚本

该脚本定义了面部重建模型，包含网络定义、损失计算、参数优化等功能。
用于从单张图片重建3D人脸模型。
"""

import numpy as np
import torch
from src.face3d.models.base_model import BaseModel
from src.face3d.models import networks
from src.face3d.models.bfm import ParametricFaceModel
from src.face3d.models.losses import perceptual_loss, photo_loss, reg_loss, reflectance_loss, landmark_loss
from src.face3d.util import util 
from src.face3d.util.nvdiffrast import MeshRenderer

import trimesh
from scipy.io import savemat

class FaceReconModel(BaseModel):
    """面部重建模型类
    
    继承自BaseModel，实现了面部3D重建的完整流程。
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        """配置CUT模型的特定选项
        
        Args:
            parser: 命令行参数解析器
            is_train: 是否为训练模式
        
        Returns:
            parser: 修改后的参数解析器
        """
        # 网络结构和参数配置
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
        # ResNet预训练权重路径
        parser.add_argument('--init_path', type=str, default='./checkpoints/init_model/resnet50-0676ba61.pth')
        # 是否使用最后的全连接层
        parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
        # BFM模型文件夹路径
        parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
        # BFM模型文件名
        parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

        # 渲染器参数配置
        parser.add_argument('--focal', type=float, default=1015.)  # 焦距
        parser.add_argument('--center', type=float, default=112.)  # 图像中心点
        parser.add_argument('--camera_d', type=float, default=10.)  # 相机距离
        parser.add_argument('--z_near', type=float, default=5.)  # 近裁剪面
        parser.add_argument('--z_far', type=float, default=15.)  # 远裁剪面

        if is_train:
            # 训练参数配置
            # 人脸识别网络结构
            parser.add_argument('--net_recog', type=str, default='r50', choices=['r18', 'r43', 'r50'], help='face recog network structure')
            # 人脸识别模型预训练权重路径
            parser.add_argument('--net_recog_path', type=str, default='checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')
            # 是否使用裁剪后的面部掩码计算光照损失
            parser.add_argument('--use_crop_face', type=util.str2bool, nargs='?', const=True, default=False, help='use crop mask for photo loss')
            # 是否使用预定义的变换矩阵M
            parser.add_argument('--use_predef_M', type=util.str2bool, nargs='?', const=True, default=False, help='use predefined M for predicted face')

            
            # 数据增强参数
            parser.add_argument('--shift_pixs', type=float, default=10., help='shift pixels')  # 像素偏移量
            parser.add_argument('--scale_delta', type=float, default=0.1, help='delta scale factor')  # 缩放因子变化量
            parser.add_argument('--rot_angle', type=float, default=10., help='rot angles, degree')  # 旋转角度

            # 各项损失函数的权重配置
            parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')  # 特征损失权重
            parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')  # 颜色损失权重
            parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')  # 正则化损失权重
            parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')  # 身份正则化损失权重
            parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')  # 表情正则化损失权重
            parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')  # 纹理正则化损失权重
            parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')  # 光照参数损失权重
            parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')  # 关键点损失权重
            parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')  # 反射率损失权重

        # 解析已知参数
        opt, _ = parser.parse_known_args()
        # 设置默认参数值
        parser.set_defaults(
                focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
            )
        if is_train:
            parser.set_defaults(
                use_crop_face=True, use_predef_M=False
            )
        return parser

    def __init__(self, opt):
        """初始化面部重建模型
        
        Args:
            opt: 训练/测试配置选项
        """
        # 调用父类初始化方法
        BaseModel.__init__(self, opt) 
        
        # 定义可视化名称列表
        self.visual_names = ['output_vis']
        # 定义模型名称列表
        self.model_names = ['net_recon']
        # 定义需要并行处理的模型名称列表
        self.parallel_names = self.model_names + ['renderer']

        # 初始化参数化面部模型（BFM）
        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model
        )
        
        # 计算视场角（FOV）
        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        # 初始化网格渲染器
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center)
        )

        if self.isTrain:
            # 定义损失名称列表
            self.loss_names = ['all', 'feat', 'color', 'lm', 'reg', 'gamma', 'reflc']

            # 初始化人脸识别网络（用于特征损失计算）
            self.net_recog = networks.define_net_recog(
                net_recog=opt.net_recog, pretrained_path=opt.net_recog_path
                )
            # 损失函数名称映射：(compute_%s_loss) % loss_name
            self.compute_feat_loss = perceptual_loss  # 特征损失
            self.comupte_color_loss = photo_loss  # 颜色/光照损失
            self.compute_lm_loss = landmark_loss  # 关键点损失
            self.compute_reg_loss = reg_loss  # 正则化损失
            self.compute_reflc_loss = reflectance_loss  # 反射率损失

            # 初始化Adam优化器
            self.optimizer = torch.optim.Adam(self.net_recon.parameters(), lr=opt.lr)
            self.optimizers = [self.optimizer]
            self.parallel_names += ['net_recog']
        # 我们的程序会自动调用<model.setup>来定义学习率调度器、加载网络和打印网络信息

    def set_input(self, input):
        """从数据加载器解包输入数据并执行必要的预处理
        
        Args:
            input: 包含数据及其元信息的字典
        """
        # 将输入数据移动到指定设备
        self.input_img = input['imgs'].to(self.device) 
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None

    def forward(self, output_coeff, device):
        """前向传播，计算3D面部模型并渲染
        
        Args:
            output_coeff: 面部系数向量
            device: 计算设备
        """
        # 将面部模型移动到指定设备
        self.facemodel.to(device)
        # 根据系数计算顶点、纹理、颜色和关键点
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(output_coeff)
        # 使用渲染器渲染面部图像
        self.pred_mask, _, self.pred_face = self.renderer(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)
        
        # 分割系数字典
        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)


    def compute_losses(self):
        """计算各项损失函数
        
        在每个训练迭代中调用，计算特征损失、颜色损失、正则化损失等
        """
        # 确保人脸识别网络处于评估模式
        assert self.net_recog.training == False
        trans_m = self.trans_m
        if not self.opt.use_predef_M:
            # 估计归一化变换矩阵
            trans_m = estimate_norm_torch(self.pred_lm, self.input_img.shape[-2])

        # 计算特征损失（感知损失）
        pred_feat = self.net_recog(self.pred_face, trans_m)
        gt_feat = self.net_recog(self.input_img, self.trans_m)
        self.loss_feat = self.opt.w_feat * self.compute_feat_loss(pred_feat, gt_feat)

        face_mask = self.pred_mask
        if self.opt.use_crop_face:
            # 使用前脸掩码计算光照损失
            face_mask, _, _ = self.renderer(self.pred_vertex, self.facemodel.front_face_buf)
        
        # 分离掩码梯度
        face_mask = face_mask.detach()
        # 计算颜色/光照损失
        self.loss_color = self.opt.w_color * self.comupte_color_loss(
            self.pred_face, self.input_img, self.atten_mask * face_mask)
        
        # 计算正则化损失和光照参数损失
        loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict, self.opt)
        self.loss_reg = self.opt.w_reg * loss_reg
        self.loss_gamma = self.opt.w_gamma * loss_gamma

        # 计算关键点损失
        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(self.pred_lm, self.gt_lm)

        # 计算反射率损失
        self.loss_reflc = self.opt.w_reflc * self.compute_reflc_loss(self.pred_tex, self.facemodel.skin_mask)

        # 计算总损失
        self.loss_all = self.loss_feat + self.loss_color + self.loss_reg + self.loss_gamma \
                        + self.loss_lm + self.loss_reflc
            

    def optimize_parameters(self, isTrain=True):
        """优化网络参数
        
        在每个训练迭代中调用，执行前向传播、损失计算和反向传播
        
        Args:
            isTrain: 是否为训练模式
        """
        self.forward()               
        self.compute_losses()
        """更新网络权重；在每个训练迭代中调用"""
        if isTrain:
            # 清零梯度
            self.optimizer.zero_grad()  
            # 反向传播计算梯度
            self.loss_all.backward()         
            # 更新网络参数
            self.optimizer.step()       

    def compute_visuals(self):
        """计算可视化结果
        
        生成输入图像、渲染图像和带关键点标注的图像的对比图
        """
        with torch.no_grad():
            # 将输入图像转换为numpy数组，范围0-255
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            # 将渲染结果与原始图像混合
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            
            if self.gt_lm is not None:
                # 获取真实关键点和预测关键点
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                # 在渲染图像上绘制关键点（蓝色为真实，红色为预测）
                output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
                output_vis_numpy = util.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')
            
                # 拼接输入图像、渲染图像和带关键点的图像
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw, output_vis_numpy), axis=-2)
            else:
                # 只拼接输入图像和渲染图像
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw), axis=-2)

            # 将numpy数组转换回torch张量
            self.output_vis = torch.tensor(
                    output_vis_numpy / 255., dtype=torch.float32
                ).permute(0, 3, 1, 2).to(self.device)

    def save_mesh(self, name):
        """保存重建的3D面部网格模型
        
        Args:
            name: 输出文件名
        """
        # 获取重建的顶点坐标
        recon_shape = self.pred_vertex  # get reconstructed shape
        # 将坐标从相机空间转换到世界空间
        recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        # 获取顶点颜色
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        # 获取三角形面片索引
        tri = self.facemodel.face_buf.cpu().numpy()
        # 创建trimesh网格对象并导出
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
        mesh.export(name)

    def save_coeff(self,name):
        """保存预测的面部系数
        
        Args:
            name: 输出文件名（.mat格式）
        """
        # 将系数字典转换为numpy数组
        pred_coeffs = {key:self.pred_coeffs_dict[key].cpu().numpy() for key in self.pred_coeffs_dict}
        # 获取预测的关键点
        pred_lm = self.pred_lm.cpu().numpy()
        # 将关键点坐标转换为图像坐标系（y轴翻转）
        pred_lm = np.stack([pred_lm[:,:,0],self.input_img.shape[2]-1-pred_lm[:,:,1]],axis=2) # transfer to image coordinate
        # 添加关键点到系数字典
        pred_coeffs['lm68'] = pred_lm
        # 保存为mat文件
        savemat(name,pred_coeffs)


