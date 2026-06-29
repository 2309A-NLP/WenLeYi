"""皮肤区域注意力掩码生成模块

该脚本用于为 Deep3DFaceRecon_pytorch 生成皮肤注意力掩码。
使用基于 YCbCr 颜色空间的高斯混合模型 (GMM) 来区分皮肤和非皮肤区域。
"""
import math                         # 数学函数模块
import numpy as np                  # NumPy 数值计算库
import os                           # 操作系统接口模块
import cv2                          # OpenCV 图像处理库


class GMM:
    """高斯混合模型 (Gaussian Mixture Model)。

    用于对皮肤和非皮肤区域的 YCbCr 颜色分布进行建模。
    支持多分量的高斯分布，可计算数据点在模型下的似然度。
    """

    def __init__(self, dim, num, w, mu, cov, cov_det, cov_inv):
        """初始化高斯混合模型。

        参数:
            dim (int): 特征维度（YCbCr 为 3）
            num (int): 高斯分量数量
            w (list): 各高斯分量的权重（标量列表）
            mu (list): 各高斯分量的均值（1xdim 向量列表）
            cov (list): 协方差矩阵（dimxdim 矩阵列表）
            cov_det (list): 预计算的协方差矩阵行列式（标量列表）
            cov_inv (list): 预计算的协方差逆矩阵（dimxdim 矩阵列表）
        """
        self.dim = dim               # 特征维度
        self.num = num               # 高斯分量数量
        self.w = w                   # 权重列表
        self.mu = mu                 # 均值列表
        self.cov = cov               # 协方差矩阵列表
        self.cov_det = cov_det       # 协方差行列式列表
        self.cov_inv = cov_inv       # 协方差逆矩阵列表

        # 预计算归一化因子：(2π)^(d/2) * |Σ|^(1/2)
        self.factor = [0] * num
        for i in range(self.num):
            self.factor[i] = (2 * math.pi) ** (self.dim / 2) * self.cov_det[i] ** 0.5
        
    def likelihood(self, data):
        """计算数据点在高斯混合模型下的似然度（概率密度）。

        对于每个数据点，计算其在所有高斯分量下的加权概率密度之和：
        P(x) = Σ w_i * N(x | μ_i, Σ_i)

        参数:
            data (ndarray): 输入数据 (N, dim)，N 为样本数

        返回:
            lh (ndarray): 每个数据点的似然度 (N,)
        """
        assert(data.shape[1] == self.dim)
        N = data.shape[0]
        lh = np.zeros(N)  # 初始化似然度为零

        for i in range(self.num):
            # 计算数据点到均值的差值
            data_ = data - self.mu[i]

            # 计算马氏距离的平方：(x-μ)^T * Σ^(-1) * (x-μ)
            tmp = np.matmul(data_, self.cov_inv[i]) * data_
            tmp = np.sum(tmp, axis=1)
            power = -0.5 * tmp

            # 计算概率密度
            p = np.array([math.exp(power[j]) for j in range(N)])
            p = p / self.factor[i]  # 除以归一化因子
            lh += p * self.w[i]    # 加权求和
        
        return lh


def _rgb2ycbcr(rgb):
    """将 RGB 颜色空间转换为 YCbCr 颜色空间。

    YCbCr 颜色空间更适合皮肤检测，因为皮肤颜色主要集中在
    Cb（蓝色色度）和 Cr（红色色度）的特定范围内。

    参数:
        rgb (ndarray): RGB 图像数据

    返回:
        ycbcr (ndarray): YCbCr 颜色空间的图像数据
    """
    # RGB 到 YCbCr 的转换矩阵
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = rgb.shape
    rgb = rgb.reshape((shape[0] * shape[1], 3))  # 展平为 (N, 3)
    ycbcr = np.dot(rgb, m.transpose() / 255.)     # 矩阵乘法转换
    ycbcr[:, 0] += 16.     # Y 分量偏移
    ycbcr[:, 1:] += 128.   # Cb, Cr 分量偏移
    return ycbcr.reshape(shape)


def _bgr2ycbcr(bgr):
    """将 BGR 颜色空间（OpenCV 默认格式）转换为 YCbCr。

    参数:
        bgr (ndarray): BGR 图像数据

    返回:
        ycbcr (ndarray): YCbCr 颜色空间的图像数据
    """
    rgb = bgr[..., ::-1]   # BGR -> RGB（通道反转）
    return _rgb2ycbcr(rgb)


# ============ 皮肤 GMM 参数 ============
# 4个高斯分量的权重、均值、协方差行列式和协方差逆矩阵
gmm_skin_w = [0.24063933, 0.16365987, 0.26034665, 0.33535415]
gmm_skin_mu = [np.array([113.71862, 103.39613, 164.08226]),
                np.array([150.19858, 105.18467, 155.51428]),
                np.array([183.92976, 107.62468, 152.71820]),
                np.array([114.90524, 113.59782, 151.38217])]
gmm_skin_cov_det = [5692842.5, 5851930.5, 2329131., 1585971.]
gmm_skin_cov_inv = [np.array([[0.0019472069, 0.0020450759, -0.00060243998],[0.0020450759, 0.017700525, 0.0051420014],[-0.00060243998, 0.0051420014, 0.0081308950]]),
                    np.array([[0.0027110141, 0.0011036990, 0.0023122299],[0.0011036990, 0.010707724, 0.010742856],[0.0023122299, 0.010742856, 0.017481629]]),
                    np.array([[0.0048026871, 0.00022935172, 0.0077668377],[0.00022935172, 0.011729696, 0.0081661865],[0.0077668377, 0.0081661865, 0.025374353]]),
                    np.array([[0.0011989699, 0.0022453172, -0.0010748957],[0.0022453172, 0.047758564, 0.020332102],[-0.0010748957, 0.020332102, 0.024502251]])]

# 创建皮肤 GMM 模型实例
gmm_skin = GMM(3, 4, gmm_skin_w, gmm_skin_mu, [], gmm_skin_cov_det, gmm_skin_cov_inv)

# ============ 非皮肤 GMM 参数 ============
gmm_nonskin_w = [0.12791070, 0.31130761, 0.34245777, 0.21832393]
gmm_nonskin_mu = [np.array([99.200851, 112.07533, 140.20602]),
                    np.array([110.91392, 125.52969, 130.19237]),
                    np.array([129.75864, 129.96107, 126.96808]),
                    np.array([112.29587, 128.85121, 129.05431])]
gmm_nonskin_cov_det = [458703648., 6466488., 90611376., 133097.63]
gmm_nonskin_cov_inv = [np.array([[0.00085371657, 0.00071197288, 0.00023958916],[0.00071197288, 0.0025935620, 0.00076557708],[0.00023958916, 0.00076557708, 0.0015042332]]),
                    np.array([[0.00024650150, 0.00045542428, 0.00015019422],[0.00045542428, 0.026412144, 0.018419769],[0.00015019422, 0.018419769, 0.037497383]]),
                    np.array([[0.00037054974, 0.00038146760, 0.00040408765],[0.00038146760, 0.0085505722, 0.0079136286],[0.00040408765, 0.0079136286, 0.010982352]]),
                    np.array([[0.00013709733, 0.00051228428, 0.00012777430],[0.00051228428, 0.28237113, 0.10528370],[0.00012777430, 0.10528370, 0.23468947]])]

# 创建非皮肤 GMM 模型实例
gmm_nonskin = GMM(3, 4, gmm_nonskin_w, gmm_nonskin_mu, [], gmm_nonskin_cov_det, gmm_nonskin_cov_inv)

# 先验概率：皮肤出现的先验概率
prior_skin = 0.8
prior_nonskin = 1 - prior_skin  # 非皮肤先验概率 = 0.2


# 计算皮肤注意力掩码
def skinmask(imbgr):
    """使用贝叶斯分类器计算皮肤注意力掩码。

    使用两个高斯混合模型（皮肤和非皮肤）在 YCbCr 颜色空间中
    计算每个像素属于皮肤区域的后验概率。

    算法流程：
    1. 将 BGR 图像转换为 YCbCr 颜色空间
    2. 计算每个像素在皮肤和非皮肤 GMM 下的似然度
    3. 使用贝叶斯公式计算后验概率
    4. 将概率值映射到 [0, 255] 范围

    参数:
        imbgr (ndarray): BGR 格式的输入图像

    返回:
        post_skin (ndarray): 皮肤注意力掩码 (H, W, 3)，值范围 [0, 255]
    """
    im = _bgr2ycbcr(imbgr)  # 转换颜色空间

    data = im.reshape((-1, 3))  # 展平为 (N, 3)

    # 计算似然度
    lh_skin = gmm_skin.likelihood(data)       # 皮肤似然度
    lh_nonskin = gmm_nonskin.likelihood(data) # 非皮肤似然度

    # 贝叶斯后验概率计算
    tmp1 = prior_skin * lh_skin      # 皮肤先验 × 似然度
    tmp2 = prior_nonskin * lh_nonskin  # 非皮肤先验 × 似然度
    post_skin = tmp1 / (tmp1 + tmp2)   # 后验概率 P(皮肤|颜色)

    # 重塑为图像形状
    post_skin = post_skin.reshape((im.shape[0], im.shape[1]))

    # 映射到 [0, 255] 范围并转换为 uint8
    post_skin = np.round(post_skin * 255)
    post_skin = post_skin.astype(np.uint8)
    # 扩展为3通道以匹配图像格式
    post_skin = np.tile(np.expand_dims(post_skin, 2), [1, 1, 3])

    return post_skin


def get_skin_mask(img_path):
    """批量生成图像的皮肤注意力掩码。

    遍历指定目录中的所有图像文件，
    为每张图像生成皮肤掩码并保存到 mask/ 子目录。

    参数:
        img_path (str): 包含图像的目录路径
    """
    print('generating skin masks......')
    # 获取目录中所有图像文件
    names = [i for i in sorted(os.listdir(
        img_path)) if 'jpg' in i or 'png' in i or 'jpeg' in i or 'PNG' in i]
    # 创建掩码保存目录
    save_path = os.path.join(img_path, 'mask')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    for i in range(0, len(names)):
        name = names[i]
        print('%05d' % (i), ' ', name)
        full_image_name = os.path.join(img_path, name)
        # 读取图像并转换为 float32
        img = cv2.imread(full_image_name).astype(np.float32)
        # 生成皮肤掩码
        skin_img = skinmask(img)
        # 保存掩码图像
        cv2.imwrite(os.path.join(save_path, name), skin_img.astype(np.uint8))
