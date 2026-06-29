# ------------------------------------------------------------------------------
# HRNet - 高分辨率网络（High-Resolution Network）
# 参考文献: https://github.com/HRNet/HRNet-Image-Classification
# 
# 核心思想：在整个人体网络中始终保持高分辨率的特征表示，
# 通过并行的多分辨率子网络和反复的多尺度融合来学习丰富的特征。
# 与传统的逐步降采样再上采样的方法不同，HRNet通过并行分支
# 同时处理不同分辨率的特征，最后将所有分辨率的特征拼接输出。
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# 定义模块公开接口：可选的HRNet变体
__all__ = [ 'hrnet18s', 'hrnet18', 'hrnet32' ]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3卷积层的便捷函数
    
    参数:
        in_planes: 输入通道数
        out_planes: 输出通道数
        stride: 卷积步长，默认为1
    返回:
        3x3卷积层，padding=1 保持空间尺寸不变（当stride=1时）
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """基础残差块（Basic Residual Block）
    
    由两个3x3卷积层组成的残差块，用于较浅的网络层。
    结构：Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + 残差连接 -> ReLU
    
    expansion = 1 表示输出通道数等于输入通道数（不进行通道扩展）
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """初始化基础残差块
        
        参数:
            inplanes: 输入通道数
            planes: 中间层通道数
            stride: 第一个卷积层的步长
            downsample: 下采样层（当输入输出维度不匹配时使用）
        """
        super(BasicBlock, self).__init__()
        # 第一个3x3卷积层
        self.conv1 = conv3x3(inplanes, planes, stride)
        # 第一个批量归一化层
        self.bn1 = nn.BatchNorm2d(planes, )
        # ReLU激活函数（inplace=True 节省内存）
        self.relu = nn.ReLU(inplace=True)
        # 第二个3x3卷积层
        self.conv2 = conv3x3(planes, planes)
        # 第二个批量归一化层
        self.bn2 = nn.BatchNorm2d(planes, )
        # 下采样层（用于匹配残差连接的维度）
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """前向传播
        
        残差连接的核心：输出 = F(x) + x
        当维度不匹配时，通过 downsample 对输入进行变换
        """
        # 保存输入作为残差
        residual = x

        # 第一个卷积-BN-ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积-BN（此处不加ReLU，加在残差连接之后）
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，对残差进行维度匹配
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接：主路径输出 + 残差
        out += residual
        # 最终ReLU激活
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """瓶颈残差块（Bottleneck Residual Block）
    
    由1x1、3x3、1x1三个卷积层组成，效率更高。
    结构：Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN + 残差 -> ReLU
    
    expansion = 4 表示输出通道数是中间层通道数的4倍
    1x1卷积用于降维和升维，3x3卷积负责特征提取，减少计算量
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """初始化瓶颈残差块
        
        参数:
            inplanes: 输入通道数
            planes: 中间层通道数（最终输出为 planes * 4）
            stride: 3x3卷积层的步长
            downsample: 下采样层
        """
        super(Bottleneck, self).__init__()
        # 第一个1x1卷积：降维，将通道数从 inplanes 降到 planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3卷积：空间特征提取，可选步长进行下采样
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 第二个1x1卷积：升维，将通道数从 planes 扩展到 planes * expansion
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """前向传播 - 瓶颈残差块"""
        residual = x

        # 1x1降维 + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3空间卷积 + BN + ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1升维 + BN（不加ReLU）
        out = self.conv3(out)
        out = self.bn3(out)

        # 残差连接
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    """高分辨率模块 - HRNet的核心组件
    
    该模块包含多个并行分支（每个分支处理不同分辨率的特征），
    以及多尺度融合层，将不同分支的特征进行融合。
    
    这是HRNet区别于其他网络的关键设计：
    - 并行分支保持多分辨率特征
    - 融合层实现跨分辨率的特征交互
    """
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        """初始化高分辨率模块
        
        参数:
            num_branches: 并行分支数量
            blocks: 残差块类型列表（每个分支可以不同）
            num_blocks: 每个分支的残差块数量列表
            num_inchannels: 每个分支的输入通道数列表
            num_channels: 每个分支的输出通道数列表
            fuse_method: 特征融合方法（如 'SUM' 表示逐元素相加）
            multi_scale_output: 是否输出多尺度特征（最后一个模块可关闭）
        """
        super(HighResolutionModule, self).__init__()
        # 验证分支参数的一致性
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels  # 各分支的输入通道数
        self.fuse_method = fuse_method  # 融合方法
        self.num_branches = num_branches  # 分支数量

        self.multi_scale_output = multi_scale_output  # 是否多尺度输出

        # 创建各分支的残差块序列
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        # 创建多尺度融合层
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        """验证分支参数的一致性
        
        确保分支数量与块数量、通道数量等参数列表长度一致
        """
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        """构建单个分支的残差块序列
        
        参数:
            branch_index: 分支索引
            block: 残差块类型
            num_blocks: 该分支的残差块数量
            num_channels: 该分支的通道数
            stride: 第一个块的步长
        返回:
            包含多个残差块的Sequential模块
        """
        downsample = None
        # 当步长不为1或输入输出通道数不匹配时，需要下采样层
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion),
            )

        layers = []
        # 第一个残差块可能带有下采样
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        # 更新输入通道数（expansion倍）
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        # 后续残差块不再下采样
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """构建所有并行分支
        
        每个分支独立处理不同分辨率的特征图
        """
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """构建多尺度特征融合层
        
        融合层的作用是将不同分支（不同分辨率）的特征对齐后相加。
        融合策略：
        - 如果 j > i（高分辨率融合低分辨率）：使用1x1卷积调整通道 + 上采样
        - 如果 j == i：同分辨率直接相加（不需要变换）
        - 如果 j < i（低分辨率融合高分辨率）：使用3x3步长卷积进行下采样
        """
        # 单分支时不需要融合
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        # 遍历每个目标分辨率
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            # 遍历每个源分辨率
            for j in range(num_branches):
                if j > i:
                    # 高分辨率分支需要下采样到低分辨率：使用1x1卷积 + 上采样
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    # 同一分支，不需要变换，直接相加
                    fuse_layer.append(None)
                else:
                    # 低分辨率分支需要上采样到高分辨率：使用步长为2的3x3卷积逐级下采样
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            # 最后一个3x3卷积，输出通道匹配目标分辨率
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            # 中间的3x3卷积，保持通道数不变
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        """获取各分支的当前输入通道数"""
        return self.num_inchannels

    def forward(self, x):
        """前向传播 - 多分支并行处理 + 特征融合
        
        参数:
            x: 输入特征图列表，每个元素对应一个分支的输入
        返回:
            融合后的多尺度特征图列表
        """
        # 单分支特殊情况
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # 各分支独立处理
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        # 多尺度融合：将所有分支的特征按分辨率对齐后相加
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            # 从第一个分支开始
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            # 逐个分支融合
            for j in range(1, self.num_branches):
                if i == j:
                    # 同分辨率直接相加
                    y = y + x[j]
                else:
                    # 不同分辨率通过融合层对齐后相加
                    y = y + self.fuse_layers[i][j](x[j])
            # ReLU激活
            x_fuse.append(self.relu(y))

        return x_fuse

class HighResolutionNet(nn.Module):
    """HRNet - 高分辨率网络主模型
    
    整体架构：
    1. 初始层：两个3x3卷积进行4倍下采样
    2. Layer1：标准残差层（单分辨率）
    3. Stage2-4：逐渐增加并行分支数量（1->2->3->4个分支）
    4. 每个Stage之间通过过渡层连接
    5. 最终将所有分辨率的特征图上采样到相同尺寸后拼接
    
    这种设计使得网络能同时处理不同尺度的特征，
    对于人脸关键点检测这种需要精细定位的任务非常有效。
    """

    def __init__(self, num_modules, num_branches, block, 
            num_blocks, num_channels, fuse_method, **kwargs):
        """初始化HRNet
        
        参数:
            num_modules: 每个Stage的模块数量列表
            num_branches: 每个Stage的分支数量列表
            block: 每个Stage使用的残差块类型列表
            num_blocks: 每个Stage每个分支的块数量列表
            num_channels: 每个Stage每个分支的通道数列表
            fuse_method: 每个Stage的融合方法列表
        """
        super(HighResolutionNet, self).__init__()
        self.num_modules = num_modules
        self.num_branches = num_branches
        self.block = block
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.fuse_method = fuse_method

        # 初始卷积层：3通道输入 -> 64通道，2倍下采样
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 第二个卷积层：继续2倍下采样，总共4倍下采样
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # === Stage 1: 单分辨率残差层 ===
        num_channels, num_blocks = self.num_channels[0][0], self.num_blocks[0][0]
        self.layer1 = self._make_layer(self.block[0], 64, num_channels, num_blocks)
        stage1_out_channel = self.block[0].expansion*num_channels

        # === Stage 2: 过渡层 + 2分支高分辨率模块 ===
        num_channels, num_blocks = self.num_channels[1], self.num_blocks[1]
        num_channels = [
            num_channels[i] * self.block[1].expansion for i in range(len(num_channels))]
        # 过渡层：将Stage1的单分辨率特征转换为多分辨率
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(1, num_channels)

        # === Stage 3: 过渡层 + 3分支高分辨率模块 ===
        num_channels, num_blocks = self.num_channels[2], self.num_blocks[2]
        num_channels = [
            num_channels[i] * self.block[2].expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(2, num_channels)

        # === Stage 4: 过渡层 + 4分支高分辨率模块 ===
        num_channels, num_blocks = self.num_channels[3], self.num_blocks[3]
        num_channels = [
            num_channels[i] * self.block[3].expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(3, num_channels, multi_scale_output=True)
        # 最终输出通道数 = 所有分支通道数之和
        self._out_channels = sum(pre_stage_channels)
        
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """构建过渡层 - 连接相邻Stage
        
        过渡层负责在不同Stage之间进行特征维度的适配：
        - 如果前后通道数相同：不进行变换（None）
        - 如果通道数不同：使用3x3卷积调整
        - 如果需要新增分支：使用步长为2的3x3卷积创建新的低分辨率分支
        
        参数:
            num_channels_pre_layer: 前一Stage各分支的通道数列表
            num_channels_cur_layer: 当前Stage各分支的通道数列表
        返回:
            过渡层列表
        """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                # 已有分支：检查通道数是否需要调整
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], ),
                        nn.ReLU(inplace=True)))
                else:
                    # 通道数相同，不需要变换
                    transition_layers.append(None)
            else:
                # 新增分支：通过步长为2的3x3卷积创建新的低分辨率分支
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, ),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """构建残差层 - 用于Stage1
        
        创建由多个残差块组成的Sequential层，
        可选进行下采样。
        
        参数:
            block: 残差块类型
            inplanes: 输入通道数
            planes: 中间层通道数
            blocks: 残差块数量
            stride: 下采样步长
        """
        downsample = None
        # 当需要下采样或维度不匹配时，创建下采样层
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, ),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, stage_index, in_channels,
                    multi_scale_output=True):
        """构建一个Stage - 包含多个高分辨率模块
        
        每个Stage由多个HighResolutionModule串联组成，
        模块间通过特征传递实现信息流动。
        
        参数:
            stage_index: Stage索引（0-3）
            in_channels: 输入通道数列表
            multi_scale_output: 是否多尺度输出
        返回:
            模块序列和输出通道数列表
        """
        num_modules = self.num_modules[stage_index]
        num_branches = self.num_branches[stage_index]
        num_blocks = self.num_blocks[stage_index]
        num_channels = self.num_channels[stage_index]
        block = self.block[stage_index]
        fuse_method = self.fuse_method[stage_index]
        modules = []
        for i in range(num_modules):
            # 只有最后一个模块可以关闭多尺度输出
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      in_channels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            # 更新下个模块的输入通道数
            in_channels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), in_channels

    def forward(self, x):
        """前向传播 - HRNet完整推理流程
        
        1. 初始卷积层进行4倍下采样
        2. Layer1进行特征提取
        3. 通过过渡层和各Stage逐步增加分支
        4. 最终将所有分辨率特征上采样到同一尺寸后拼接
        
        参数:
            x: 输入图像 [N, 3, H, W]
        返回:
            多尺度融合特征图 [N, C_total, H/4, W/4]
        """
        # 初始卷积：3 -> 64 通道，4倍下采样
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # Stage 1：单分辨率特征提取
        x = self.layer1(x)

        # Stage 2：通过过渡层扩展为2个分支
        x_list = []
        for i in range(self.num_branches[1]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # Stage 3：通过过渡层扩展为3个分支
        x_list = []
        for i in range(self.num_branches[2]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # Stage 4：通过过渡层扩展为4个分支
        x_list = []
        for i in range(self.num_branches[3]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        # 将所有分辨率的特征图上采样到最大分辨率后沿通道维度拼接
        kwargs = {
            'size': tuple(y_list[0].shape[-2:]),  # 目标尺寸 = 最高分辨率分支的尺寸
            'mode': 'bilinear', 'align_corners': False,  # 双线性插值
        }
        return torch.cat([F.interpolate(y,**kwargs) for y in y_list], 1)

# ==================== HRNet 模型工厂函数 ====================

def hrnet18s(pretrained=True, **kwargs):
    """构建 HRNet-W18-S（轻量版）模型
    
    最轻量的HRNet变体，适合计算资源有限的场景。
    分支通道数较小：(18, 36, 72, 144)
    """
    model = HighResolutionNet(
        num_modules = [1, 1, 3, 2],       # 各Stage的模块数量
        num_branches = [1, 2, 3, 4],       # 各Stage的分支数量
        block = [Bottleneck, BasicBlock, BasicBlock, BasicBlock],  # 各Stage的残差块类型
        num_blocks = [(2,), (2,2), (2,2,2), (2,2,2,2)],           # 各分支的块数量
        num_channels = [(64,), (18,36), (18,36,72), (18,36,72,144)],  # 各分支的通道数
        fuse_method = ['SUM', 'SUM', 'SUM', 'SUM'],  # 所有Stage使用求和融合
        **kwargs
    )
    if pretrained:
        # 加载ImageNet预训练权重（严格=False 允许部分加载）
        model.load_state_dict(model_zoo.load_url(model_urls['hrnet_w18s']), strict=False)
    return model

def hrnet18(pretrained=False, **kwargs):
    """构建 HRNet-W18 标准模型
    
    平衡精度和速度的中等规模HRNet。
    分支通道数：(18, 36, 72, 144)
    比hrnet18s有更深的网络（更多残差块和模块）
    """
    model = HighResolutionNet(
        num_modules = [1, 1, 4, 3],
        num_branches = [1, 2, 3, 4],
        block = [Bottleneck, BasicBlock, BasicBlock, BasicBlock],
        num_blocks = [(4,), (4,4), (4,4,4), (4,4,4,4)],
        num_channels = [(64,), (18,36), (18,36,72), (18,36,72,144)],
        fuse_method = ['SUM', 'SUM', 'SUM', 'SUM'],
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['hrnet18']), strict=False)
    return model

def hrnet32(pretrained=False, **kwargs):
    """构建 HRNet-W32 模型
    
    最大的HRNet变体，精度最高但计算量也最大。
    分支通道数更大：(32, 64, 128, 256)
    适合对精度要求高且计算资源充足的场景
    """
    model = HighResolutionNet(
        num_modules = [1, 1, 4, 3],
        num_branches = [1, 2, 3, 4],
        block = [Bottleneck, BasicBlock, BasicBlock, BasicBlock],
        num_blocks = [(4,), (4,4), (4,4,4), (4,4,4,4)],
        num_channels = [(64,), (32,64), (32,64,128), (32,64,128,256)],
        fuse_method = ['SUM', 'SUM', 'SUM', 'SUM'],
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['hrnet32']), strict=False)
    return model

