"""
人脸验证评估模块

该模块实现了人脸识别模型的验证评估功能，主要用于在 LFW（Labeled Faces in the Wild）
等标准人脸验证数据集上测试模型性能。

主要功能包括：
1. 加载验证数据集（pickle 格式的二进制文件）
2. 使用模型提取人脸嵌入特征
3. 计算 ROC 曲线和验证精度（Verification Accuracy）
4. 支持 k-fold 交叉验证和 PCA 降维

原始代码基于 David Sandberg 的 FaceNet 实现，使用 MIT 许可证。
"""

"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
import os
import pickle

import mxnet as mx
import numpy as np
import sklearn
import torch
from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


class LFold:
    """
    LFold 类：L-Fold 交叉验证分割器
    
    用于将验证数据集分割成 k 折进行交叉验证。
    当 n_splits=1 时，不进行交叉验证，直接返回原始数据。
    
    参数:
        n_splits (int): 交叉验证的折数，默认为 2
        shuffle (bool): 是否在分割前打乱数据，默认为 False
    """
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        """
        将索引集合分割为训练集和测试集
        
        参数:
            indices: 待分割的索引数组
            
        返回:
            当 n_splits > 1 时返回 KFold 的分割结果，
            否则返回 [(indices, indices)] 即不分割
        """
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    """
    计算 ROC（Receiver Operating Characteristic）曲线
    
    通过 k-fold 交叉验证计算不同阈值下的 TPR（True Positive Rate）
    和 FPR（False Positive Rate），用于评估人脸识别模型的验证性能。
    
    参数:
        thresholds: 阈值数组，用于判断两个人脸是否为同一人
        embeddings1: 第一组人脸嵌入特征（N × D）
        embeddings2: 第二组人脸嵌入特征（N × D）
        actual_issame: 真实标签数组（True 表示同一人，False 表示不同人）
        nrof_folds: 交叉验证的折数，默认 10
        pca: PCA 降维维度（0 表示不使用 PCA），默认 0
        
    返回:
        tpr: 各阈值下的平均真正例率
        fpr: 各阈值下的平均假正例率
        accuracy: 各折的验证精度
    """
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    # 初始化存储数组
    tprs = np.zeros((nrof_folds, nrof_thresholds))  # 真正例率矩阵
    fprs = np.zeros((nrof_folds, nrof_thresholds))  # 假正例率矩阵
    accuracy = np.zeros((nrof_folds))                 # 各折精度
    indices = np.arange(nrof_pairs)

    # 不使用 PCA 时，直接计算嵌入向量间的平方欧氏距离
    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    # 对每一折进行交叉验证
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # 如果指定 PCA 降维，则在训练集上拟合 PCA 模型并转换
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            # PCA 后进行 L2 归一化
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # 在训练集上寻找最优阈值
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        # 使用最优阈值在测试集上评估
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    # 计算所有折的平均 TPR 和 FPR
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    """
    计算给定阈值下的验证精度、TPR 和 FPR
    
    根据阈值将嵌入距离转换为预测结果（距离 < 阈值则认为是同一人），
    然后与真实标签对比计算各项指标。
    
    参数:
        threshold: 判断阈值（距离小于此值认为是同一人）
        dist: 嵌入向量间的距离数组
        actual_issame: 真实标签数组
        
    返回:
        tpr: 真正例率 = TP / (TP + FN)
        fpr: 假正例率 = FP / (FP + TN)
        acc: 总体精度 = (TP + TN) / 总样本数
    """
    # 根据阈值生成预测结果
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))      # 真正例
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))  # 假正例
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))               # 真负例
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))  # 假负例

    # 计算 TPR、FPR 和精度
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    """
    计算指定 FAR（False Accept Rate）目标下的验证指标
    
    通过 k-fold 交叉验证，在给定的 FAR 目标（如 1e-3）下，
    计算对应的验证接受率（VAL），用于评估模型在特定安全等级下的性能。
    
    参数:
        thresholds: 阈值数组
        embeddings1: 第一组人脸嵌入特征
        embeddings2: 第二组人脸嵌入特征
        actual_issame: 真实标签
        far_target: 目标假接受率（如 0.001 = 0.1%）
        nrof_folds: 交叉验证折数
        
    返回:
        val_mean: 平均验证接受率
        val_std: 验证接受率标准差
        far_mean: 平均假接受率
    """
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    # 计算嵌入向量间的平方欧氏距离
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # 在训练集上寻找使 FAR 等于目标值的阈值
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            # 使用线性插值找到精确的阈值
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        # 在测试集上使用该阈值评估
        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    """
    计算给定阈值下的验证接受率（VAL）和假接受率（FAR）
    
    参数:
        threshold: 判断阈值
        dist: 嵌入向量间的距离数组
        actual_issame: 真实标签
        
    返回:
        val: 验证接受率 = 正确接受的对数 / 所有同类对数
        far: 假接受率 = 错误接受的对数 / 所有异类对数
    """
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)          # 同一人对的数量
    n_diff = np.sum(np.logical_not(actual_issame))  # 不同人对的数量
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    """
    综合评估函数：计算 ROC 曲线和验证指标
    
    这是评估的主入口函数，内部调用 calculate_roc 和 calculate_val
    计算完整的评估指标。
    
    参数:
        embeddings: 人脸嵌入特征（交错排列：[img1a, img1b, img2a, img2b, ...]）
        actual_issame: 真实标签列表
        nrof_folds: 交叉验证折数
        pca: PCA 降维维度（0=不使用）
        
    返回:
        tpr: 真正例率数组
        fpr: 假正例率数组
        accuracy: 各折精度数组
        val: 验证接受率
        val_std: 验证接受率标准差
        far: 假接受率
    """
    # 将交错的嵌入向量分为两组（原始图和翻转图）
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]   # 偶数索引：原始图像的嵌入
    embeddings2 = embeddings[1::2]   # 奇数索引：水平翻转图像的嵌入
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    # 使用更细粒度的阈值计算 FAR=1e-3 下的验证指标
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

@torch.no_grad()
def load_bin(path, image_size):
    """
    从二进制文件加载人脸验证数据集
    
    数据集格式为 pickle 序列化的元组 (bins, issame_list)：
    - bins: 图像二进制数据列表（JPEG 编码）
    - issame_list: 真实标签列表（True/False）
    
    加载时会同时生成原始图像和水平翻转图像两个版本。
    
    参数:
        path: 二进制文件路径
        image_size: 目标图像尺寸 (H, W)
        
    返回:
        data_list: [原始图像张量, 翻转图像张量]
        issame_list: 真实标签列表
    """
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    # 为原始图像和翻转图像分别分配内存
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    # 逐张加载和处理图像
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        # 如果图像尺寸不匹配，进行缩放
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        # 转换为 CHW 格式（通道在前）
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                # 水平翻转图像（用于测试翻转一致性）
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list

@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    """
    使用模型对验证数据集进行推理测试
    
    将数据集中的图像分批送入骨干网络提取嵌入特征，
    然后合并原始和翻转图像的特征进行归一化，
    最后调用 evaluate 函数计算验证精度。
    
    参数:
        data_set: load_bin 返回的数据集元组
        backbone: 人脸识别骨干网络（PyTorch 模型）
        batch_size: 推理时的批量大小
        nfolds: 交叉验证折数
        
    返回:
        acc1: 未使用的精度指标（始终为 0）
        std1: 未使用的标准差（始终为 0）
        acc2: 翻转测试的平均验证精度
        std2: 翻转测试精度的标准差
        _xnorm: 嵌入向量的平均 L2 范数
        embeddings_list: 嵌入特征列表 [原始, 翻转]
    """
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    # 对原始图像和翻转图像分别提取嵌入特征
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            # 图像预处理：归一化到 [-1, 1] 范围
            img = ((_data / 255) - 0.5) / 0.5
            # 前向推理提取嵌入特征
            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    # 计算所有嵌入向量的平均 L2 范数（用于监控训练质量）
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    acc1 = 0.0
    std1 = 0.0
    # 合并原始和翻转图像的嵌入特征
    embeddings = embeddings_list[0] + embeddings_list[1]
    # L2 归一化
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    # 调用评估函数计算验证指标
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


def dumpR(data_set,
          backbone,
          batch_size,
          name='',
          data_extra=None,
          label_shape=None):
    """
    导出验证数据集的嵌入特征到文件
    
    与 test() 函数类似，但将提取的嵌入特征保存到 pickle 文件中，
    而不是计算验证指标。用于后续分析或特征可视化。
    
    参数:
        data_set: 数据集元组
        backbone: 骨干网络模型
        batch_size: 推理批量大小
        name: 名称标识
        data_extra: 额外输入数据（用于双输入模型）
        label_shape: 标签形状（未使用）
    """
    print('dump verification embedding..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba

            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra),
                                     label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    actual_issame = np.asarray(issame_list)
    # 保存嵌入特征和标签到二进制文件
    outname = os.path.join('temp.bin')
    with open(outname, 'wb') as f:
        pickle.dump((embeddings, issame_list),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL)


# 以下为被注释掉的命令行入口代码
# 该代码用于从命令行加载模型并在验证集上进行评估
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='do verification')
#     # general
#     parser.add_argument('--data-dir', default='', help='')
#     parser.add_argument('--model',
#                         default='../model/softmax,50',
#                         help='path to load model.')
#     parser.add_argument('--target',
#                         default='lfw,cfp_ff,cfp_fp,agedb_30',
#                         help='test targets.')
#     parser.add_argument('--gpu', default=0, type=int, help='gpu id')
#     parser.add_argument('--batch-size', default=32, type=int, help='')
#     parser.add_argument('--max', default='', type=str, help='')
#     parser.add_argument('--mode', default=0, type=int, help='')
#     parser.add_argument('--nfolds', default=10, type=int, help='')
#     args = parser.parse_args()
#     image_size = [112, 112]
#     print('image_size', image_size)
#     ctx = mx.gpu(args.gpu)
#     nets = []
#     vec = args.model.split(',')
#     prefix = args.model.split(',')[0]
#     epochs = []
#     if len(vec) == 1:
#         pdir = os.path.dirname(prefix)
#         for fname in os.listdir(pdir):
#             if not fname.endswith('.params'):
#                 continue
#             _file = os.path.join(pdir, fname)
#             if _file.startswith(prefix):
#                 epoch = int(fname.split('.')[0].split('-')[1])
#                 epochs.append(epoch)
#         epochs = sorted(epochs, reverse=True)
#         if len(args.max) > 0:
#             _max = [int(x) for x in args.max.split(',')]
#             assert len(_max) == 2
#             if len(epochs) > _max[1]:
#                 epochs = epochs[_max[0]:_max[1]]
#
#     else:
#         epochs = [int(x) for x in vec[1].split('|')]
#     print('model number', len(epochs))
#     time0 = datetime.datetime.now()
#     for epoch in epochs:
#         print('loading', prefix, epoch)
#         sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
#         all_layers = sym.get_internals()
#         sym = all_layers['fc1_output']
#         model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
#         model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0],
#                                           image_size[1]))])
#         model.set_params(arg_params, aux_params)
#         nets.append(model)
#     time_now = datetime.datetime.now()
#     diff = time_now - time0
#     print('model loading time', diff.total_seconds())
#
#     ver_list = []
#     ver_name_list = []
#     for name in args.target.split(','):
#         path = os.path.join(args.data_dir, name + ".bin")
#         if os.path.exists(path):
#             print('loading.. ', name)
#             data_set = load_bin(path, image_size)
#             ver_list.append(data_set)
#             ver_name_list.append(name)
#
#     if args.mode == 0:
#         for i in range(len(ver_list)):
#             results = []
#             for model in nets:
#                 acc1, std1, acc2, std2, xnorm, embeddings_list = test(
#                     ver_list[i], model, args.batch_size, args.nfolds)
#                 print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
#                 print('[%s]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], acc1, std1))
#                 print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], acc2, std2))
#                 results.append(acc2)
#             print('Max of [%s] is %1.5f' % (ver_name_list[i], np.max(results)))
#     elif args.mode == 1:
#         raise ValueError
#     else:
#         model = nets[0]
#         dumpR(ver_list[0], model, args.batch_size, args.target)
