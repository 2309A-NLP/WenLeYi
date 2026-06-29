"""
ONNX 模型辅助工具

该模块提供了 ArcFace ONNX 模型的加载、验证和推理功能。
主要用于：
1. 检查 ONNX 模型的合法性（输入输出格式、模型大小、推理速度等）
2. 支持多种竞赛赛道（CFAT、MS1M、Glint、Unconstrained）
3. 批量和单张图像的推理接口
4. 模型性能基准测试

该模块是 IJB（IARPA Janus Benchmark）评测流程的一部分。
"""

from __future__ import division
import datetime
import os
import os.path as osp
import glob
import numpy as np
import cv2
import sys
import onnxruntime
import onnx
import argparse
from onnx import numpy_helper
from insightface.data import get_image

class ArcFaceORT:
    """
    ArcFace ONNX Runtime 推理类
    
    封装了 ONNX Runtime 的推理接口，提供模型加载、验证和推理功能。
    支持 CPU 和 GPU 两种执行后端。
    
    属性:
        model_path (str): ONNX 模型目录路径
        providers (list): ONNX Runtime 执行后端列表
        session: ONNX Runtime 推理会话
        input_name (str): 模型输入节点名称
        output_names (list): 模型输出节点名称列表
        image_size (tuple): 模型输入图像尺寸 (W, H)
        input_mean (float): 输入图像均值（用于预处理）
        input_std (float): 输入图像标准差（用于预处理）
        feat_dim (int): 输出特征向量维度
        cost_ms (float): 单次推理耗时（毫秒）
        model_size_mb (float): 模型文件大小（MB）
    """
    def __init__(self, model_path, cpu=False):
        """
        初始化 ArcFaceORT 实例
        
        参数:
            model_path (str): ONNX 模型文件或目录路径
            cpu (bool): 是否强制使用 CPU 推理（默认 False 使用 GPU）
        """
        self.model_path = model_path
        # providers = None 时自动选择可用的最佳后端
        # 对于 onnxruntime-gpu，会自动选择 "CUDAExecutionProvider"
        self.providers = ['CPUExecutionProvider'] if cpu else None

    #input_size is (w,h), return error message, return None if success
    def check(self, track='cfat', test_img = None):
        """
        检查并验证 ONNX 模型的合法性
        
        对模型进行全面检查，包括：
        - 模型文件是否存在和可加载
        - 输入形状是否为 4D（N, C, H, W）
        - 输出节点数量是否为 1
        - 模型大小是否在限制范围内
        - 特征维度是否合法
        - 推理速度是否达标
        - 权重数据类型是否有效
        - 批量推理是否正常
        
        参数:
            track (str): 竞赛赛道名称（'cfat', 'ms1m', 'glint', 'unconstrained'）
            test_img: 测试图像（None 时使用默认测试图像）
            
        返回:
            str: 错误信息（成功时返回 None）
        """
        # 根据赛道设置不同的限制参数
        #default is cfat
        max_model_size_mb=1024
        max_feat_dim=512
        max_time_cost=15
        if track.startswith('ms1m'):
            max_model_size_mb=1024
            max_feat_dim=512
            max_time_cost=10
        elif track.startswith('glint'):
            max_model_size_mb=1024
            max_feat_dim=1024
            max_time_cost=20
        elif track.startswith('cfat'):
            max_model_size_mb = 1024
            max_feat_dim = 512
            max_time_cost = 15
        elif track.startswith('unconstrained'):
            max_model_size_mb=1024
            max_feat_dim=1024
            max_time_cost=30
        else:
            return "track not found"

        # 检查模型路径是否存在
        if not os.path.exists(self.model_path):
            return "model_path not exists"
        if not os.path.isdir(self.model_path):
            return "model_path should be directory"
        
        # 查找目录中的 ONNX 模型文件
        onnx_files = []
        for _file in os.listdir(self.model_path):
            if _file.endswith('.onnx'):
                onnx_files.append(osp.join(self.model_path, _file))
        if len(onnx_files)==0:
            return "do not have onnx files"
        
        # 选择最新的 ONNX 文件（按文件名排序取最后一个）
        self.model_file = sorted(onnx_files)[-1]
        print('use onnx-model:', self.model_file)
        
        # 尝试加载 ONNX 模型
        try:
            session = onnxruntime.InferenceSession(self.model_file, providers=self.providers)
        except:
            return "load onnx failed"
        
        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        print('input-shape:', input_shape)
        
        # 检查输入形状是否为 4D
        if len(input_shape)!=4:
            return "length of input_shape should be 4"
        
        # 如果第一维不是动态维度（字符串 'None'），则修改模型支持动态批量
        if not isinstance(input_shape[0], str):
            #return "input_shape[0] should be str to support batch-inference"
            print('reset input-shape[0] to None')
            model = onnx.load(self.model_file)
            model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
            new_model_file = osp.join(self.model_path, 'zzzzrefined.onnx')
            onnx.save(model, new_model_file)
            self.model_file = new_model_file
            print('use new onnx-model:', self.model_file)
            try:
                session = onnxruntime.InferenceSession(self.model_file, providers=self.providers)
            except:
                return "load onnx failed"
            input_cfg = session.get_inputs()[0]
            input_shape = input_cfg.shape
            print('new-input-shape:', input_shape)

        # 从输入形状中提取图像尺寸 (H, W) -> (W, H)
        self.image_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        
        # 检查输出节点数量是否为 1
        if len(output_names)!=1:
            return "number of output nodes should be 1"
        self.session = session
        self.input_name = input_name
        self.output_names = output_names
        
        # 加载 ONNX 模型图以检查节点数量
        model = onnx.load(self.model_file)
        graph = model.graph
        if len(graph.node)<8:
            return "too small onnx graph"

        input_size = (112,112)
        self.crop = None
        
        # CFAT 赛道特殊处理：读取裁剪参数
        if track=='cfat':
            crop_file = osp.join(self.model_path, 'crop.txt')
            if osp.exists(crop_file):
                lines = open(crop_file,'r').readlines()
                if len(lines)!=6:
                    return "crop.txt should contain 6 lines"
                lines = [int(x) for x in lines]
                self.crop = lines[:4]      # 裁剪区域 [x1, y1, x2, y2]
                input_size = tuple(lines[4:6])  # 输入尺寸
        if input_size!=self.image_size:
            return "input-size is inconsistant with onnx model input, %s vs %s"%(input_size, self.image_size)

        # 检查模型文件大小
        self.model_size_mb = os.path.getsize(self.model_file) / float(1024*1024)
        if self.model_size_mb > max_model_size_mb:
            return "max model size exceed, given %.3f-MB"%self.model_size_mb

        # 检测输入预处理方式（均值和标准差）
        input_mean = None
        input_std = None
        if track=='cfat':
            # CFAT 赛道从配置文件读取像素归一化参数
            pn_file = osp.join(self.model_path, 'pixel_norm.txt')
            if osp.exists(pn_file):
                lines = open(pn_file,'r').readlines()
                if len(lines)!=2:
                    return "pixel_norm.txt should contain 2 lines"
                input_mean = float(lines[0])
                input_std = float(lines[1])
        if input_mean is not None or input_std is not None:
            if input_mean is None or input_std is None:
                return "please set input_mean and input_std simultaneously"
        else:
            # 通过分析模型图的前几个节点判断预处理方式
            find_sub = False
            find_mul = False
            for nid, node in enumerate(graph.node[:8]):
                print(nid, node.name)
                if node.name.startswith('Sub') or node.name.startswith('_minus'):
                    find_sub = True
                if node.name.startswith('Mul') or node.name.startswith('_mul') or node.name.startswith('Div'):
                    find_mul = True
            if find_sub and find_mul:
                print("find sub and mul")
                # 如果模型包含 Sub 和 Mul 节点，说明预处理已内置在模型中
                #mxnet arcface model
                input_mean = 0.0
                input_std = 1.0
            else:
                # 标准预处理：(pixel - 127.5) / 127.5
                input_mean = 127.5
                input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        
        # 检查所有权重参数的数据类型（必须是 float32 或更高精度）
        for initn in graph.initializer:
            weight_array = numpy_helper.to_array(initn)
            dt = weight_array.dtype
            if dt.itemsize<4:
                return 'invalid weight type - (%s:%s)' % (initn.name, dt.name)
        
        # 使用测试图像验证模型推理
        if test_img is None:
            test_img = get_image('Tom_Hanks_54745')  # 默认使用 Tom Hanks 的测试图像
            test_img = cv2.resize(test_img, self.image_size)
        else:
            test_img = cv2.resize(test_img, self.image_size)
        
        # 执行基准测试获取特征和推理耗时
        feat, cost = self.benchmark(test_img)
        
        # 测试批量推理的数值稳定性
        batch_result = self.check_batch(test_img)
        batch_result_sum = float(np.sum(batch_result))
        if batch_result_sum in [float('inf'), -float('inf')] or batch_result_sum != batch_result_sum:
            print(batch_result)
            print(batch_result_sum)
            return "batch result output contains NaN!"

        # 验证特征维度
        if len(feat.shape) < 2:
           return "the shape of the feature must be two, but get {}".format(str(feat.shape))

        if feat.shape[1] > max_feat_dim:
            return "max feat dim exceed, given %d"%feat.shape[1]
        self.feat_dim = feat.shape[1]
        
        # 验证推理速度
        cost_ms = cost*1000
        if cost_ms>max_time_cost:
            return "max time cost exceed, given %.4f"%cost_ms
        self.cost_ms = cost_ms
        print('check stat:, model-size-mb: %.4f, feat-dim: %d, time-cost-ms: %.4f, input-mean: %.3f, input-std: %.3f'%(self.model_size_mb, self.feat_dim, self.cost_ms, self.input_mean, self.input_std))
        return None

    def check_batch(self, img):
        """
        测试批量推理功能
        
        使用 32 张相同图像进行批量推理，检测模型是否支持批量推理
        以及输出数值是否正常（无 NaN/Inf）。
        
        参数:
            img: 输入图像（单张或列表）
            
        返回:
            net_out: 模型输出特征
        """
        if not isinstance(img, list):
            imgs = [img, ] * 32
        # 如果设置了裁剪区域，对每张图像进行裁剪
        if self.crop is not None:
            nimgs = []
            for img in imgs:
                nimg = img[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2], :]
                if nimg.shape[0] != self.image_size[1] or nimg.shape[1] != self.image_size[0]:
                    nimg = cv2.resize(nimg, self.image_size)
                nimgs.append(nimg)
            imgs = nimgs
        # 使用 OpenCV DNN 模块进行图像预处理和批量推理
        blob = cv2.dnn.blobFromImages(
            images=imgs, scalefactor=1.0 / self.input_std, size=self.image_size,
            mean=(self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out


    def meta_info(self):
        """
        返回模型元信息字典
        
        返回:
            dict: 包含模型大小、特征维度和推理耗时的字典
        """
        return {'model-size-mb':self.model_size_mb, 'feature-dim':self.feat_dim, 'infer': self.cost_ms}


    def forward(self, imgs):
        """
        对输入图像执行前向推理，提取人脸嵌入特征
        
        支持单张图像（自动转换为列表）和批量图像输入。
        
        参数:
            imgs: 输入图像（单张 numpy 数组或图像列表）
            
        返回:
            net_out: 模型输出的特征向量
        """
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.image_size
        # 如果设置了裁剪区域，对每张图像进行裁剪
        if self.crop is not None:
            nimgs = []
            for img in imgs:
                nimg = img[self.crop[1]:self.crop[3],self.crop[0]:self.crop[2],:]
                if nimg.shape[0]!=input_size[1] or nimg.shape[1]!=input_size[0]:
                    nimg = cv2.resize(nimg, input_size)
                nimgs.append(nimg)
            imgs = nimgs
        # 图像预处理：缩放、减均值、通道转换（BGR->RGB）
        blob = cv2.dnn.blobFromImages(imgs, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name : blob})[0]
        return net_out

    def benchmark(self, img):
        """
        模型推理性能基准测试
        
        对同一张图像执行 50 次推理，取第 6 次的结果作为稳定耗时
        （前几次可能包含 JIT 编译等初始化开销）。
        
        参数:
            img: 测试图像
            
        返回:
            net_out: 模型输出特征
            cost: 单次推理耗时（秒）
        """
        input_size = self.image_size
        if self.crop is not None:
            nimg = img[self.crop[1]:self.crop[3],self.crop[0]:self.crop[2],:]
            if nimg.shape[0]!=input_size[1] or nimg.shape[1]!=input_size[0]:
                nimg = cv2.resize(nimg, input_size)
            img = nimg
        blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        costs = []
        for _ in range(50):
            ta = datetime.datetime.now()
            net_out = self.session.run(self.output_names, {self.input_name : blob})[0]
            tb = datetime.datetime.now()
            cost = (tb-ta).total_seconds()
            costs.append(cost)
        costs = sorted(costs)
        cost = costs[5]  # 取第6次的耗时（跳过前5次可能的不稳定值）
        return net_out, cost


if __name__ == '__main__':
    # 命令行入口：用于检查提交的 ONNX 模型是否符合竞赛要求
    parser = argparse.ArgumentParser(description='')
    # general
    parser.add_argument('workdir', help='submitted work dir', type=str)
    parser.add_argument('--track', help='track name, for different challenge', type=str, default='cfat')
    args = parser.parse_args()
    handler = ArcFaceORT(args.workdir)
    err = handler.check(args.track)
    print('err:', err)
