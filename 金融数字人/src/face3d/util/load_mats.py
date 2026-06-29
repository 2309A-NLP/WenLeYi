"""3D人脸模型加载模块

该脚本用于加载和转换 3D 人脸模型数据（BFM - Basel Face Model），
为 Deep3DFaceRecon_pytorch 提供3D形体和纹理基函数。
"""
import numpy as np                  # NumPy 数值计算库
from PIL import Image               # PIL 图像处理库
from scipy.io import loadmat, savemat  # MATLAB 文件读写
from array import array             # 高效数组模块
import os.path as osp               # 路径处理模块


# 加载表情基函数
def LoadExpBasis(bfm_folder='BFM'):
    """从二进制文件加载表情基函数。

    该函数从 Exp_Pca.bin 文件中读取表情的 PCA 基函数，
    包括平均表情和主成分。

    参数:
        bfm_folder (str): BFM 模型文件夹路径

    返回:
        expPC (ndarray): 表情主成分基函数矩阵 (3*n_vertex, n_exp)
        expEV (ndarray): 表情对应的特征值（标准差）
    """
    n_vertex = 53215  # BFM 模型的总顶点数
    # 打开二进制文件读取表情数据
    Expbin = open(osp.join(bfm_folder, 'Exp_Pca.bin'), 'rb')
    exp_dim = array('i')             # 整数数组，存储表情维度
    exp_dim.fromfile(Expbin, 1)      # 读取表情维度（主成分数）
    expMU = array('f')               # 浮点数组，存储平均表情
    expPC = array('f')               # 浮点数组，存储主成分
    expMU.fromfile(Expbin, 3 * n_vertex)          # 读取平均表情向量
    expPC.fromfile(Expbin, 3 * exp_dim[0] * n_vertex)  # 读取主成分矩阵
    Expbin.close()

    # 转换为 NumPy 数组并重塑形状
    expPC = np.array(expPC)
    expPC = np.reshape(expPC, [exp_dim[0], -1])   # (n_exp, 3*n_vertex)
    expPC = np.transpose(expPC)                     # 转置为 (3*n_vertex, n_exp)

    # 加载表情特征值（标准差）
    expEV = np.loadtxt(osp.join(bfm_folder, 'std_exp.txt'))

    return expPC, expEV


# 将原始 BFM09 转换为项目使用的正面人脸模型
def transferBFM09(bfm_folder='BFM'):
    """将原始 BFM09 模型转换为精简的正面人脸模型。

    转换步骤：
    1. 加载原始 BFM09 模型的形状、纹理和表情基函数
    2. 将基函数乘以对应的特征值进行缩放
    3. 截取前80个形状/纹理基和64个表情基
    4. 根据顶点索引提取正面区域
    5. 保存为新的模型文件

    参数:
        bfm_folder (str): BFM 模型文件夹路径
    """
    print('Transfer BFM09 to BFM_model_front......')

    # 加载原始 BFM09 模型数据
    original_BFM = loadmat(osp.join(bfm_folder, '01_MorphableModel.mat'))
    shapePC = original_BFM['shapePC']    # 形状 PCA 基函数
    shapeEV = original_BFM['shapeEV']    # 形状特征值
    shapeMU = original_BFM['shapeMU']    # 平均人脸形状
    texPC = original_BFM['texPC']        # 纹理 PCA 基函数
    texEV = original_BFM['texEV']        # 纹理特征值
    texMU = original_BFM['texMU']        # 平均人脸纹理

    # 加载表情基函数
    expPC, expEV = LoadExpBasis(bfm_folder)

    # ============ 处理形状基函数 ============
    # 乘以特征值进行缩放（归一化到分米尺度）
    idBase = shapePC * np.reshape(shapeEV, [-1, 199])
    idBase = idBase / 1e5               # 缩放到分米尺度
    idBase = idBase[:, :80]             # 只使用前80个基函数

    # ============ 处理表情基函数 ============
    exBase = expPC * np.reshape(expEV, [-1, 79])
    exBase = exBase / 1e5               # 缩放到分米尺度
    exBase = exBase[:, :64]             # 只使用前64个基函数

    # ============ 处理纹理基函数 ============
    texBase = texPC * np.reshape(texEV, [-1, 199])
    texBase = texBase[:, :80]           # 只使用前80个基函数

    # ============ 根据顶点索引提取正面区域 ============
    # 项目的人脸模型沿人脸关键点裁剪，仅包含 35709 个顶点
    # 原始 BFM09 包含 53490 个顶点，Guo 等人的表情基包含 53215 个顶点

    # 加载正面区域的顶点索引（用于表情基）
    index_exp = loadmat(osp.join(bfm_folder, 'BFM_front_idx.mat'))
    index_exp = index_exp['idx'].astype(np.int32) - 1  # 转换为从0开始的索引

    # 加载精简后的顶点索引（用于形状/纹理基）
    index_shape = loadmat(osp.join(bfm_folder, 'BFM_exp_idx.mat'))
    index_shape = index_shape['trimIndex'].astype(np.int32) - 1  # 从0开始
    index_shape = index_shape[index_exp]  # 提取对应的正面顶点

    # 按顶点索引裁剪形状基函数
    idBase = np.reshape(idBase, [-1, 3, 80])
    idBase = idBase[index_shape, :, :]
    idBase = np.reshape(idBase, [-1, 80])

    # 按顶点索引裁剪纹理基函数
    texBase = np.reshape(texBase, [-1, 3, 80])
    texBase = texBase[index_shape, :, :]
    texBase = np.reshape(texBase, [-1, 80])

    # 按顶点索引裁剪表情基函数
    exBase = np.reshape(exBase, [-1, 3, 64])
    exBase = exBase[index_exp, :, :]
    exBase = np.reshape(exBase, [-1, 64])

    # 裁剪平均形状
    meanshape = np.reshape(shapeMU, [-1, 3]) / 1e5
    meanshape = meanshape[index_shape, :]
    meanshape = np.reshape(meanshape, [1, -1])

    # 裁剪平均纹理
    meantex = np.reshape(texMU, [-1, 3])
    meantex = meantex[index_shape, :]
    meantex = np.reshape(meantex, [1, -1])

    # 加载其他信息：三角面片、光度损失区域、皮肤纹理正则化区域、68个关键点索引等
    other_info = loadmat(osp.join(bfm_folder, 'facemodel_info.mat'))
    frontmask2_idx = other_info['frontmask2_idx']    # 正面遮罩索引
    skinmask = other_info['skinmask']                # 皮肤区域遮罩
    keypoints = other_info['keypoints']              # 关键点索引
    point_buf = other_info['point_buf']              # 顶点缓冲
    tri = other_info['tri']                          # 三角面片
    tri_mask2 = other_info['tri_mask2']              # 遮罩区域的三角面片

    # 保存转换后的人脸模型
    savemat(osp.join(bfm_folder, 'BFM_model_front.mat'), {
        'meanshape': meanshape,     # 平均形状
        'meantex': meantex,         # 平均纹理
        'idBase': idBase,           # 形状基函数
        'exBase': exBase,           # 表情基函数
        'texBase': texBase,         # 纹理基函数
        'tri': tri,                 # 三角面片
        'point_buf': point_buf,     # 顶点缓冲
        'tri_mask2': tri_mask2,     # 遮罩三角面片
        'keypoints': keypoints,     # 关键点索引
        'frontmask2_idx': frontmask2_idx,  # 正面遮罩索引
        'skinmask': skinmask        # 皮肤区域遮罩
    })


# 加载用于图像预处理的标准人脸关键点
def load_lm3d(bfm_folder):
    """加载标准3D人脸的5个关键点，用于图像预处理对齐。

    从68个关键点中选取5个关键点（鼻子、左右眼中心、左右嘴角），
    用于将输入图像对齐到标准位置。

    参数:
        bfm_folder (str): BFM 模型文件夹路径

    返回:
        Lm3D (ndarray): 5个关键点的3D坐标 (5, 3)
    """
    # 加载68个3D关键点
    Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # 从68个关键点中选取5个关键点（索引从1开始）
    # 31: 鼻尖, 37/40: 左眼上下取平均, 43/46: 右眼上下取平均, 49: 左嘴角, 55: 右嘴角
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1  # 转为从0开始
    Lm3D = np.stack([
        Lm3D[lm_idx[0], :],                    # 鼻尖
        np.mean(Lm3D[lm_idx[[1, 2]], :], 0),   # 左眼中心（上下平均）
        np.mean(Lm3D[lm_idx[[3, 4]], :], 0),   # 右眼中心（上下平均）
        Lm3D[lm_idx[5], :],                     # 左嘴角
        Lm3D[lm_idx[6], :]                      # 右嘴角
    ], axis=0)
    # 重新排列顺序：左眼、右眼、鼻尖、左嘴角、右嘴角
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


if __name__ == '__main__':
    # 执行 BFM09 模型转换
    transferBFM09()
