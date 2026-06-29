# utils.py - 通用工具模块
# 本模块提供人脸对齐、仿射变换、视频处理等通用工具函数
# 主要用于FaceNet/ArcFace等模型的人脸预处理和后处理

import cv2
import numpy as np
from skimage import transform as trans  # 用于仿射变换


# ArcFace标准人脸关键点（5个关键点的坐标）
# 这是112x112像素的标准人脸对齐模板
arcface_src = np.array([[38.2946, 51.6963],  # 左眼
                        [73.5318, 51.5014],  # 右眼
                        [56.0252, 71.7366],  # 鼻子
                        [41.5493, 92.3655],  # 左嘴角
                        [70.7299, 92.2041]], dtype=np.float32)  # 右嘴角
arcface_src = np.expand_dims(arcface_src, axis=0)  # 扩展维度为(1, 5, 2)


def estimate_norm(lmk, face_size, dst_face_size, expand_size):
    """估计人脸关键点到标准模板的仿射变换矩阵
    
    使用相似变换（旋转+缩放+平移）将检测到的人脸关键点
    对齐到标准模板位置。
    
    Args:
        lmk: 检测到的人脸关键点，shape=(5, 2)
        face_size: 原始人脸尺寸（固定为112）
        dst_face_size: 目标人脸尺寸
        expand_size: 扩展后的目标尺寸
    
    Returns:
        (最优变换矩阵M, 最优索引)
    """
    assert lmk.shape == (5, 2)  # 确保输入是5个关键点
    tform = trans.SimilarityTransform()  # 创建相似变换对象（支持旋转、缩放、平移）
    # 在关键点坐标中插入一列1（齐次坐标，用于仿射变换）
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1) 
    min_M = []                                              # 存储最优变换矩阵
    min_index = []                                          # 存储最优索引
    min_error = float('inf')   # 初始化最小误差为无穷大

    assert face_size == 112  # 确保输入尺寸为112（ArcFace标准）
    # 将标准模板从112x112缩放并平移到目标尺寸
    src = (arcface_src / face_size * dst_face_size) + (expand_size - dst_face_size) / 2               
   
    for i in np.arange(src.shape[0]):  # 遍历所有模板（当前只有一个）
        # 估计从关键点到模板的变换矩阵
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]  # 取仿射变换矩阵的前两行（2x3矩阵）
        # 使用变换矩阵变换关键点
        results = np.dot(M, lmk_tran.T)
        results = results.T
        # 计算变换后的关键点与模板之间的误差
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))

        # 选择误差最小的变换矩阵
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def metrix_M(face_size, expand_size, keypoints=None):
    """构建完整的人脸对齐变换矩阵
    
    结合关键点检测和标准模板，计算最终的仿射变换矩阵。
    
    Args:
        face_size: 目标人脸尺寸
        expand_size: 扩展后的目标尺寸
        keypoints: 人脸关键点列表
    
    Returns:
        2x3仿射变换矩阵M
    """
    id_size = 112  # ArcFace标准输入尺寸
    detected_lmk = np.concatenate(keypoints).reshape(5, 2)  # 将关键点重塑为(5, 2)
    M, _ = estimate_norm(detected_lmk, id_size, face_size, expand_size)  # 估计变换矩阵
    # 构建3x3齐次变换矩阵
    Minv = np.identity(3, dtype=np.single)  # 创建3x3单位矩阵
    Minv[0:2, :] = M  # 填入仿射变换参数
    M = Minv[0:2, :]  # 取前两行作为最终变换矩阵
    return M   


def decompose_tfm(tfm):
    """将仿射变换矩阵分解为旋转平移矩阵和缩放矩阵
    
    将2x3的仿射变换矩阵分解为：
    - rt: 旋转和平移矩阵
    - s: 缩放矩阵
    
    数学原理：仿射变换 = 缩放 × (旋转 + 平移)
    
    Args:
        tfm: 2x3仿射变换矩阵
    
    Returns:
        (rt: 2x3旋转平移矩阵, s: 2x3缩放矩阵)
    """
    tfm = tfm.copy()
    # 计算x和y方向的缩放因子
    s_x = np.sqrt(tfm[0][0] ** 2 + tfm[0][1] ** 2)  # x方向缩放因子
    s_y = np.sqrt(tfm[1][0] ** 2 + tfm[1][1] ** 2)  # y方向缩放因子

    t_x = tfm[0][2]  # x方向平移量
    t_y = tfm[1][2]  # y方向平移量

    #平移旋转矩阵rt（去除缩放后的变换）
    rt = np.array([
        [tfm[0][0] / s_x, tfm[0][1] / s_x, t_x / s_x],
        [tfm[1][0] / s_y, tfm[1][1] / s_y, t_y / s_y],
    ])

    #缩放矩阵s（仅包含缩放因子）
    s = np.array([
        [s_x, 0, 0],
        [0, s_y, 0]
    ])

    # 验证分解的正确性（已注释）
    # _rt = np.vstack([rt, [[0, 0, 1]]])
    # _s = np.vstack([s, [[0, 0, 1]]])
    # print(np.dot(_s, _rt)[:2] - tfm)

    return rt, s


def img_warp(img, M, expand_size, adjust=0):
    """使用仿射变换矩阵对图片进行扭曲
    
    将人脸图片通过仿射变换对齐到标准位置。
    
    Args:
        img: 输入图片
        M: 2x3仿射变换矩阵
        expand_size: 输出图片尺寸
        adjust: 亮度调整值（负值增加亮度，正值降低亮度）
    
    Returns:
        扭曲后的图片
    """
    warped = cv2.warpAffine(img, M, (expand_size, expand_size))  # 执行仿射变换
    warped = warped - np.uint8(adjust)  # 亮度调整
    warped = np.clip(warped, 0, 255)  # 裁剪到有效范围
    return warped


def img_warp_back_inv_m(img, img_to, inv_m):
    """使用逆仿射变换矩阵将图片反向扭曲回去
    
    用于将增强后的人脸图片映射回原始图片坐标系。
    
    Args:
        img: 需要反向扭曲的图片
        img_to: 目标图片（作为底图）
        inv_m: 逆仿射变换矩阵（3x3）
    
    Returns:
        反向扭曲后的图片（与img_to叠加）
    """
    h_up, w_up, c = img_to.shape  # 获取目标图片尺寸

    # 创建掩码（用于确定有效像素区域）
    mask = np.ones_like(img).astype(np.float32)
    inv_mask = cv2.warpAffine(mask, inv_m, (w_up, h_up))  # 应用逆变换到掩码
    inv_img = cv2.warpAffine(img, inv_m, (w_up, h_up))  # 应用逆变换到图片

    # 将反向扭曲后的像素复制到目标图片的对应位置
    img_to[inv_mask == 1] = inv_img[inv_mask == 1]
    return img_to


def get_video_fps(vfile):
    """获取视频文件的帧率
    
    Args:
        vfile: 视频文件路径
    
    Returns:
        视频帧率（FPS）
    """
    cap = cv2.VideoCapture(vfile)  # 打开视频文件
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率属性
    cap.release()  # 释放视频资源
    return fps


class laplacianSmooth(object):
    """拉普拉斯平滑类
    
    用于对人脸关键点进行时序平滑处理，
    减少关键点在帧间的抖动，使动画更加自然。
    基于指数加权移动平均（EWMA）实现。
    """

    def __init__(self, smoothAlpha=0.3):
        """初始化平滑器
        
        Args:
            smoothAlpha: 平滑系数（0-1），越大越平滑，越小越跟随当前帧
        """
        self.smoothAlpha = smoothAlpha  # 平滑系数
        self.pts_last = None  # 上一帧的关键点坐标

    def smooth(self, pts_cur):
        """对当前帧的关键点进行平滑处理
        
        使用指数加权方式将当前帧的关键点与上一帧的关键点混合，
        基于关键点之间的距离自适应调整权重。
        
        Args:
            pts_cur: 当前帧的关键点坐标数组
        
        Returns:
            平滑后的关键点坐标数组
        """
        if self.pts_last is None:
            # 第一帧直接返回，无法平滑
            self.pts_last = pts_cur.copy()
            return pts_cur.copy()
        # 计算关键点区域的宽度（用于归一化距离）
        x1 = min(pts_cur[:, 0])
        x2 = max(pts_cur[:, 0])
        y1 = min(pts_cur[:, 1])
        y2 = max(pts_cur[:, 1])
        width = x2 - x1
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]  # 当前帧关键点
            x_old, y_old = self.pts_last[i]  # 上一帧关键点
            # 计算当前帧与上一帧关键点之间的欧氏距离
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            # 基于距离计算权重（距离越大，越倾向于使用当前帧的值）
            w = np.exp(-tmp / (width * self.smoothAlpha))
            # 加权混合：权重越大越接近上一帧，越小越接近当前帧
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])
        pts_update = np.array(pts_update)
        self.pts_last = pts_update.copy()  # 更新上一帧的关键点

        return pts_update
