"""
PyTorch 到 ONNX 模型转换工具

该模块提供了将 PyTorch 训练的 ArcFace 人脸识别模型转换为 ONNX 格式的功能。
ONNX（Open Neural Network Exchange）是一种开放的模型格式，支持跨框架部署。

主要功能：
1. 加载 PyTorch 模型权重
2. 使用随机输入执行模型导出为 ONNX 格式
3. 修改动态批量维度以支持可变批量大小推理
4. 可选使用 onnxsim 进行模型简化优化

使用方法：
    python torch2onnx.py <input_pth_path> --output <output_dir> --network <backbone_name>
"""

import numpy as np
import onnx
import torch


def convert_onnx(net, path_module, output, opset=11, simplify=False):
    """
    将 PyTorch 模型转换为 ONNX 格式
    
    转换流程：
    1. 生成随机输入图像（112×112×3）
    2. 对输入进行标准预处理（归一化到 [-1, 1]）
    3. 加载 PyTorch 模型权重
    4. 使用 torch.onnx.export 导出 ONNX 模型
    5. 修改输入维度为动态（支持可变批量大小）
    6. 可选进行模型简化
    
    参数:
        net (torch.nn.Module): PyTorch 网络模型实例
        path_module (str): PyTorch 模型权重文件路径（.pth）
        output (str): 输出 ONNX 文件路径
        opset (int): ONNX opset 版本，默认 11
        simplify (bool): 是否使用 onnxsim 简化模型，默认 False
    """
    assert isinstance(net, torch.nn.Module)
    
    # 生成随机测试图像（112×112×3，RGB）
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float)
    # 标准预处理：(pixel / 255 - 0.5) / 0.5，归一化到 [-1, 1]
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    # HWC -> CHW 转换，并添加 batch 维度
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    # 加载模型权重并设置为评估模式
    weight = torch.load(path_module)
    net.load_state_dict(weight)
    net.eval()
    
    # 导出 ONNX 模型
    torch.onnx.export(net, img, output, keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    
    # 加载导出的模型并修改输入维度为动态
    model = onnx.load(output)
    graph = model.graph
    # 将 batch 维度设置为 'None'（动态维度），支持可变批量大小推理
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    
    # 可选：使用 onnxsim 简化模型（移除冗余操作、融合算子等）
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)

    
if __name__ == '__main__':
    # 命令行入口：将 PyTorch 模型转换为 ONNX 格式
    import os
    import argparse
    from backbones import get_model

    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('input', type=str, help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    parser.add_argument('--network', type=str, default=None, help='backbone network')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    args = parser.parse_args()
    
    # 处理输入路径（支持文件或目录）
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "backbone.pth")
    assert os.path.exists(input_file)
    
    # 从目录名自动推断网络类型
    # 目录名格式示例：ms1mv3_arcface_r50
    model_name = os.path.basename(os.path.dirname(input_file)).lower()
    params = model_name.split("_")
    if len(params) >= 3 and params[1] in ('arcface', 'cosface'):
        if args.network is None:
            args.network = params[2]  # 从目录名提取网络名称（如 r50, r100）
    assert args.network is not None
    print(args)
    
    # 创建骨干网络模型实例
    backbone_onnx = get_model(args.network, dropout=0)

    # 设置输出路径
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'onnx')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    assert os.path.isdir(output_path)
    output_file = os.path.join(output_path, "%s.onnx" % model_name)
    
    # 执行转换
    convert_onnx(backbone_onnx, input_file, output_file, simplify=args.simplify)
