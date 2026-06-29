"""AWing 人脸关键点检测网络架构

该模块实现了基于 CoordConv 和 Hourglass 架构的面部关键点检测网络 (FAN)。
包含以下核心组件：
- calculate_points: 从热力图中提取关键点坐标
- AddCoordsTh/CoordConvTh: 坐标卷积层
- BasicBlock/ConvBlock: 基础卷积块
- HourGlass: 沙漏网络模块
- FAN: 面部对齐网络 (Face Alignment Network)
"""
import cv2                          # OpenCV 图像处理库
import numpy as np                  # NumPy 数值计算库
import torch                        # PyTorch 深度学习框架
import torch.nn as nn               # PyTorch 神经网络模块
import torch.nn.functional as F     # PyTorch 函数式接口


def calculate_points(heatmaps):
    """从热力图中提取关键点坐标。

    使用 argmax 找到每个关键点的最大激活位置，
    并通过相邻像素差值进行亚像素精度的偏移修正。

    参数:
        heatmaps (ndarray): 热力图 (B, N, H, W)
            B: 批次大小, N: 关键点数量, H/W: 热力图尺寸

    返回:
        preds (ndarray): 预测的关键点坐标 (B, N, 2)，格式为 (x, y)
    """
    B, N, H, W = heatmaps.shape
    HW = H * W                     # 热力图展平后的长度
    BN_range = np.arange(B * N)    # 批次和关键点的索引范围

    # 将热力图展平为 (B, N, H*W)
    heatline = heatmaps.reshape(B, N, HW)
    # 找到每个关键点的最大值索引
    indexes = np.argmax(heatline, axis=2)

    # 将一维索引转换为二维坐标 (x, y)
    preds = np.stack((indexes % W, indexes // W), axis=2)
    preds = preds.astype(np.float, copy=False)

    # 亚像素精度修正：利用最大值相邻像素的差值进行偏移
    inr = indexes.ravel()  # 展平索引

    heatline = heatline.reshape(B * N, HW)
    x_up = heatline[BN_range, inr + 1]      # x方向右侧值
    x_down = heatline[BN_range, inr - 1]    # x方向左侧值

    # y方向：需要处理边界情况
    if any((inr + W) >= 4096):
        y_up = heatline[BN_range, 4095]     # 防止越界
    else:
        y_up = heatline[BN_range, inr + W]  # y方向下方值
    if any((inr - W) <= 0):
        y_down = heatline[BN_range, 0]      # 防止越界
    else:
        y_down = heatline[BN_range, inr - W]  # y方向上方值

    # 计算偏移方向并乘以 0.25 的步长
    think_diff = np.sign(np.stack((x_up - x_down, y_up - y_down), axis=1))
    think_diff *= .25

    # 应用亚像素偏移并加 0.5 使坐标居中
    preds += think_diff.reshape(B, N, 2)
    preds += .5
    return preds


class AddCoordsTh(nn.Module):
    """坐标添加层（PyTorch 版本）。

    向输入张量中添加 x 和 y 坐标通道，
    使网络能够感知空间位置信息。
    坐标值归一化到 [-1, 1] 范围。
    """

    def __init__(self, x_dim=64, y_dim=64, with_r=False, with_boundary=False):
        """初始化坐标添加层。

        参数:
            x_dim (int): x方向维度
            y_dim (int): y方向维度
            with_r (bool): 是否添加半径通道（到原点的距离）
            with_boundary (bool): 是否添加边界通道
        """
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, heatmap=None):
        """前向传播：为输入张量添加坐标通道。

        参数:
            input_tensor (Tensor): 输入张量 (batch, c, x_dim, y_dim)
            heatmap (Tensor, optional): 热力图，用于边界通道计算

        返回:
            ret (Tensor): 添加坐标通道后的张量
        """
        batch_size_tensor = input_tensor.shape[0]

        # 生成 x 坐标通道：每一列的值等于列索引
        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32, device=input_tensor.device)
        xx_ones = xx_ones.unsqueeze(-1)
        xx_range = torch.arange(self.x_dim, dtype=torch.int32, device=input_tensor.device).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)
        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)

        # 生成 y 坐标通道：每一行的值等于行索引
        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32, device=input_tensor.device)
        yy_ones = yy_ones.unsqueeze(1)
        yy_range = torch.arange(self.y_dim, dtype=torch.int32, device=input_tensor.device).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)
        yy_channel = torch.matmul(yy_range.float(), yy_ones.float())
        yy_channel = yy_channel.unsqueeze(-1)

        # 调整维度顺序为 (batch, 1, x, y)
        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        # 归一化到 [-1, 1] 范围
        xx_channel = xx_channel / (self.x_dim - 1)
        yy_channel = yy_channel / (self.y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        # 扩展到批次维度
        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        # 如果启用边界通道，根据热力图生成边界坐标
        if self.with_boundary and heatmap is not None:
            # 提取边界热力图并截断到 [0, 1]
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = torch.zeros_like(xx_channel)
            # 在边界区域保留坐标值，非边界区域设为零
            xx_boundary_channel = torch.where(boundary_channel > 0.05, xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, yy_channel, zero_tensor)
        if self.with_boundary and heatmap is not None:
            xx_boundary_channel = xx_boundary_channel.to(input_tensor.device)
            yy_boundary_channel = yy_boundary_channel.to(input_tensor.device)

        # 将坐标通道与输入拼接
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        # 如果启用半径通道，计算到原点的距离
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            rr = rr / torch.max(rr)  # 归一化到 [0, 1]
            ret = torch.cat([ret, rr], dim=1)

        # 如果启用边界通道，拼接边界坐标
        if self.with_boundary and heatmap is not None:
            ret = torch.cat([ret, xx_boundary_channel, yy_boundary_channel], dim=1)
        return ret


class CoordConvTh(nn.Module):
    """坐标卷积层 (CoordConv)。

    论文: "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution"
    在标准卷积之前添加坐标通道，帮助网络学习空间关系。
    """

    def __init__(self, x_dim, y_dim, with_r, with_boundary, in_channels, first_one=False, *args, **kwargs):
        """初始化坐标卷积层。

        参数:
            x_dim, y_dim: 空间维度
            with_r: 是否使用半径特征
            with_boundary: 是否使用边界特征
            in_channels: 输入通道数
            first_one: 是否为第一层（影响边界通道数）
        """
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(x_dim=x_dim, y_dim=y_dim, with_r=with_r, with_boundary=with_boundary)
        in_channels += 2  # 添加 x, y 两个坐标通道
        if with_r:
            in_channels += 1  # 添加半径通道
        if with_boundary and not first_one:
            in_channels += 2  # 添加边界坐标通道
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        """前向传播。

        参数:
            input_tensor: 输入特征图
            heatmap: 热力图（可选）

        返回:
            ret: 卷积输出
            last_channel: 最后两个坐标通道（用于可视化）
        """
        ret = self.addcoords(input_tensor, heatmap)  # 添加坐标
        last_channel = ret[:, -2:, :, :]  # 保存坐标通道用于可视化
        ret = self.conv(ret)              # 标准卷积
        return ret, last_channel


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False, dilation=1):
    """3x3 卷积层，带有填充。

    参数:
        in_planes: 输入通道数
        out_planes: 输出通道数
        strd: 步长
        padding: 填充大小
        bias: 是否使用偏置
        dilation: 膨胀卷积的膨胀率

    返回:
        nn.Conv2d: 3x3 卷积层
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias, dilation=dilation)


class BasicBlock(nn.Module):
    """基础残差块。

    包含两个 3x3 卷积层和一个跳跃连接。
    """
    expansion = 1  # 输出通道数扩展倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """初始化基础残差块。

        参数:
            inplanes: 输入通道数
            planes: 输出通道数
            stride: 步长
            downsample: 下采样层（当输入输出维度不匹配时使用）
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)  # 第一个卷积
        # self.bn1 = nn.BatchNorm2d(planes)  # 批归一化（当前未使用）
        self.relu = nn.ReLU(inplace=True)   # ReLU 激活函数
        self.conv2 = conv3x3(planes, planes)  # 第二个卷积
        # self.bn2 = nn.BatchNorm2d(planes)  # 批归一化（当前未使用）
        self.downsample = downsample        # 下采样层
        self.stride = stride

    def forward(self, x):
        """前向传播。

        参数:
            x: 输入张量

        返回:
            out: 残差块输出
        """
        residual = x                       # 保存输入作为跳跃连接

        out = self.conv1(x)                # 第一个卷积
        out = self.relu(out)               # 激活

        out = self.conv2(out)              # 第二个卷积

        if self.downsample is not None:
            residual = self.downsample(x)  # 下采样跳跃连接

        out += residual                    # 残差连接
        out = self.relu(out)               # 最终激活

        return out


class ConvBlock(nn.Module):
    """卷积块：包含三层卷积和通道拼接（DenseNet 风格）。

    该模块使用多尺度特征提取和密集连接，
    三个卷积层的输出在通道维度上拼接后与残差相加。
    """

    def __init__(self, in_planes, out_planes):
        """初始化卷积块。

        参数:
            in_planes: 输入通道数
            out_planes: 输出通道数
        """
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)                    # 批归一化
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))   # 第一个卷积：输出通道/2
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4), padding=1, dilation=1)  # 第二个卷积：输出通道/4
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4), padding=1, dilation=1)  # 第三个卷积：输出通道/4

        # 如果输入输出通道不匹配，使用下采样层调整
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),  # 1x1 卷积调整通道
            )
        else:
            self.downsample = None

    def forward(self, x):
        """前向传播。

        参数:
            x: 输入张量

        返回:
            out3: 三层卷积输出拼接后与残差相加的结果
        """
        residual = x

        # 三层卷积，逐层降维
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        # 将三层卷积的输出在通道维度上拼接
        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual  # 残差连接

        return out3


class HourGlass(nn.Module):
    """沙漏网络模块 (Hourglass Module)。

    沙漏网络通过递归的下采样和上采样来捕获多尺度特征。
    结构：上分支（保持分辨率）+ 下分支（下采样后递归处理再上采样）
    """

    def __init__(self, num_modules, depth, num_features, first_one=False):
        """初始化沙漏模块。

        参数:
            num_modules (int): 模块数量
            depth (int): 沙漏的递归深度
            num_features (int): 特征图数量
            first_one (bool): 是否为第一个沙漏模块（影响 CoordConv 的边界通道）
        """
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        # 坐标卷积层：添加空间坐标信息
        self.coordconv = CoordConvTh(
            x_dim=64,
            y_dim=64,
            with_r=True,          # 使用半径特征
            with_boundary=True,   # 使用边界特征
            in_channels=256,
            first_one=first_one,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0)
        # 递归生成网络层
        self._generate_network(self.depth)

    def _generate_network(self, level):
        """递归生成沙漏网络的卷积层。

        参数:
            level (int): 当前递归深度
        """
        self.add_module('b1_' + str(level), ConvBlock(256, 256))    # 上分支卷积

        self.add_module('b2_' + str(level), ConvBlock(256, 256))    # 下分支卷积

        if level > 1:
            self._generate_network(level - 1)  # 递归生成更深的层
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))  # 最深层额外卷积

        self.add_module('b3_' + str(level), ConvBlock(256, 256))    # 上采样后卷积

    def _forward(self, level, inp):
        """递归前向传播。

        参数:
            level (int): 当前递归深度
            inp: 输入特征图

        返回:
            特征图（上分支 + 上采样后的下分支）
        """
        # 上分支：保持分辨率
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # 下分支：先下采样再处理
        low1 = F.avg_pool2d(inp, 2, stride=2)  # 2x2 平均池化下采样
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)  # 递归处理
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # 上采样回到原始分辨率
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2  # 跳跃连接相加

    def forward(self, x, heatmap):
        """前向传播。

        参数:
            x: 输入特征图
            heatmap: 当前热力图（用于坐标卷积的边界通道）

        返回:
            沙漏模块输出
            last_channel: 坐标卷积的坐标通道（用于可视化）
        """
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


class FAN(nn.Module):
    """面部对齐网络 (Face Alignment Network)。

    基于沙漏网络的多阶段关键点检测架构。
    包含一个共享的编码器和多个堆叠的沙漏模块，
    每个沙漏模块预测一组热力图并传递给下一个模块。

    参考论文: "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    """

    def __init__(self, num_modules=1, end_relu=False, gray_scale=False, num_landmarks=68, device='cuda'):
        """初始化 FAN 网络。

        参数:
            num_modules (int): 沙漏模块的堆叠数量
            end_relu (bool): 是否在输出后使用 ReLU
            gray_scale (bool): 是否使用灰度输入
            num_landmarks (int): 关键点数量，默认68
            device (str): 计算设备
        """
        super(FAN, self).__init__()
        self.device = device
        self.num_modules = num_modules
        self.gray_scale = gray_scale
        self.end_relu = end_relu
        self.num_landmarks = num_landmarks

        # ============ 编码器部分（特征提取） ============
        # 第一层：坐标卷积 + 批归一化
        if self.gray_scale:
            self.conv1 = CoordConvTh(
                x_dim=256, y_dim=256, with_r=True, with_boundary=False,
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        else:
            self.conv1 = CoordConvTh(
                x_dim=256, y_dim=256, with_r=True, with_boundary=False,
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)            # 批归一化
        self.conv2 = ConvBlock(64, 128)          # 卷积块：64 -> 128
        self.conv3 = ConvBlock(128, 128)         # 卷积块：128 -> 128
        self.conv4 = ConvBlock(128, 256)         # 卷积块：128 -> 256

        # ============ 堆叠沙漏部分（关键点预测） ============
        for hg_module in range(self.num_modules):
            first_one = (hg_module == 0)  # 第一个模块使用边界通道
            # 添加沙漏模块
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256, first_one))
            # 沙漏输出后的卷积处理
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            # 最终的 1x1 卷积
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            # 预测热力图（num_landmarks + 1，多出的1个用于边界通道）
            self.add_module('l' + str(hg_module), nn.Conv2d(256, num_landmarks + 1, kernel_size=1, stride=1, padding=0))

            # 除了最后一个模块，都需要将信息传递给下一个模块
            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module),
                                nn.Conv2d(num_landmarks + 1, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        """前向传播。

        参数:
            x: 输入图像 (batch, 3, 256, 256)

        返回:
            outputs: 各阶段的热力图输出列表
            boundary_channels: 各阶段的边界通道列表
        """
        # 编码器特征提取
        x, _ = self.conv1(x)                    # 坐标卷积 (3 -> 64)
        x = F.relu(self.bn1(x), True)           # BN + ReLU
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)  # 下采样 (64 -> 128)
        x = self.conv3(x)                       # 卷积 (128 -> 128)
        x = self.conv4(x)                       # 卷积 (128 -> 256)

        previous = x                            # 保存编码器输出

        outputs = []                            # 各阶段的热力图输出
        boundary_channels = []                  # 各阶段的边界通道
        tmp_out = None                          # 上一阶段的热力图输出

        for i in range(self.num_modules):
            # 通过沙漏模块
            hg, boundary_channel = self._modules['m' + str(i)](previous, tmp_out)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)  # 卷积处理

            # BN + ReLU
            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)

            # 预测热力图
            tmp_out = self._modules['l' + str(i)](ll)
            if self.end_relu:
                tmp_out = F.relu(tmp_out)  # 可选的 ReLU 激活

            outputs.append(tmp_out)              # 保存输出
            boundary_channels.append(boundary_channel)  # 保存边界通道

            # 将当前模块的信息传递给下一个模块
            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)        # 特征变换
                tmp_out_ = self._modules['al' + str(i)](tmp_out)  # 热力图变换
                previous = previous + ll + tmp_out_  # 残差连接

        return outputs, boundary_channels

    def get_landmarks(self, img):
        """从图像中检测68个人脸关键点。

        参数:
            img (ndarray): 输入图像 (H, W, 3)，BGR 格式

        返回:
            pred (ndarray): 预测的关键点坐标 (68, 2)
        """
        H, W, _ = img.shape
        # 计算缩放偏移量：从热力图坐标到原始图像坐标的映射
        offset = W / 64, H / 64, 0, 0

        # 预处理：缩放到 256x256，转换通道顺序 (BGR -> RGB)
        img = cv2.resize(img, (256, 256))
        inp = img[..., ::-1]
        inp = torch.from_numpy(np.ascontiguousarray(inp.transpose((2, 0, 1)))).float()
        inp = inp.to(self.device)
        inp.div_(255.0).unsqueeze_(0)  # 归一化到 [0, 1] 并添加批次维度

        # 前向传播获取热力图
        outputs, _ = self.forward(inp)
        out = outputs[-1][:, :-1, :, :]  # 取最后一个阶段的输出，去掉边界通道
        heatmaps = out.detach().cpu().numpy()

        # 从热力图中提取关键点坐标
        pred = calculate_points(heatmaps).reshape(-1, 2)

        # 将坐标从热力图空间映射回原始图像空间
        pred *= offset[:2]
        pred += offset[-2:]

        return pred
