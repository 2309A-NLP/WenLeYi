"""
IJB（IARPA Janus Benchmark）ONNX 模型评测脚本

该脚本用于在 IJB-C 或 IJB-B 数据集上评测 ArcFace ONNX 模型的人脸验证性能。
IJB 是人脸识别领域最权威的基准测试之一，包含大量约束和非约束场景的人脸图像。

主要功能：
1. 加载 IJB 元数据（模板、媒体、图像-模板对应关系）
2. 使用 ONNX 模型批量提取人脸特征
3. 执行模板级别的特征聚合
4. 计算验证对的相似度分数
5. 输出 TPR@FPR 指标表格

评测模式：
- N1: 使用归一化分数
- D1: 使用检测器分数加权
- F1: 使用翻转测试（原始 + 水平翻转）
"""

import argparse
import os
import pickle
import timeit

import cv2
import mxnet as mx
import numpy as np
import pandas as pd
import prettytable
import skimage.transform
from sklearn.metrics import roc_curve
from sklearn.preprocessing import normalize

from onnx_helper import ArcFaceORT

# 标准 5 点人脸关键点目标位置（112×112 人脸对齐用）
# 这是 ArcFace 训练时使用的标准关键点位置
SRC = np.array(
    [
        [30.2946, 51.6963],   # 左眼
        [65.5318, 51.5014],   # 右眼
        [48.0252, 71.7366],   # 鼻尖
        [33.5493, 92.3655],   # 左嘴角
        [62.7299, 92.2041]]   # 右嘴角
    , dtype=np.float32)
SRC[:, 0] += 8.0  # 水平偏移 8 像素


class AlignedDataSet(mx.gluon.data.Dataset):
    """
    人脸对齐数据集类
    
    从 IJB 数据集加载人脸图像，使用 5 点关键点进行仿射变换对齐到 112×112。
    同时生成原始图像和水平翻转图像两个版本。
    
    参数:
        root (str): 图像文件根目录
        lines (list): 图像信息行列表（格式：路径 关键点1_x 关键点1_y ... 关键点5_y 分数）
        align (bool): 是否进行人脸对齐
    """
    def __init__(self, root, lines, align=True):
        self.lines = lines
        self.root = root
        self.align = align

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        """
        获取第 idx 张图像及其翻转版本
        
        返回:
            output: 包含原始图像和翻转图像的张量 [2, 3, 112, 112]
        """
        each_line = self.lines[idx]
        # 解析图像路径和关键点坐标
        name_lmk_score = each_line.strip().split(' ')
        name = os.path.join(self.root, name_lmk_score[0])
        img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        landmark5 = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32).reshape((5, 2))
        
        # 使用相似变换将关键点对齐到标准位置
        st = skimage.transform.SimilarityTransform()
        st.estimate(landmark5, SRC)
        img = cv2.warpAffine(img, st.params[0:2, :], (112, 112), borderValue=0.0)
        
        # 生成原始图像和水平翻转图像
        img_1 = np.expand_dims(img, 0)
        img_2 = np.expand_dims(np.fliplr(img), 0)  # 水平翻转
        output = np.concatenate((img_1, img_2), axis=0).astype(np.float32)
        # 转换维度顺序：HWC -> CHW
        output = np.transpose(output, (0, 3, 1, 2))
        output = mx.nd.array(output)
        return output


def extract(model_root, dataset):
    """
    使用 ONNX 模型从数据集中批量提取人脸特征
    
    对每张图像同时提取原始和翻转两个特征，拼接为一个 2×feat_dim 的向量。
    
    参数:
        model_root (str): ONNX 模型路径
        dataset: AlignedDataSet 数据集实例
        
    返回:
        feat_mat: 特征矩阵 [N, 2*feat_dim]
    """
    model = ArcFaceORT(model_path=model_root)
    model.check()
    # 特征矩阵：原始特征 + 翻转特征
    feat_mat = np.zeros(shape=(len(dataset), 2 * model.feat_dim))

    def batchify_fn(data):
        return mx.nd.concat(*data, dim=0)

    # 使用 DataLoader 进行批量加载
    data_loader = mx.gluon.data.DataLoader(
        dataset, 128, last_batch='keep', num_workers=4,
        thread_pool=True, prefetch=16, batchify_fn=batchify_fn)
    num_iter = 0
    for batch in data_loader:
        batch = batch.asnumpy()
        # 图像预处理：(pixel - mean) / std
        batch = (batch - model.input_mean) / model.input_std
        # ONNX 推理提取特征
        feat = model.session.run(model.output_names, {model.input_name: batch})[0]
        # 将原始和翻转特征拼接
        feat = np.reshape(feat, (-1, model.feat_dim * 2))
        feat_mat[128 * num_iter: 128 * num_iter + feat.shape[0], :] = feat
        num_iter += 1
        if num_iter % 50 == 0:
            print(num_iter)
    return feat_mat


def read_template_media_list(path):
    """
    读取模板-媒体对应关系列表
    
    IJB 数据集中，每个身份对应一个模板（template），
    每个模板包含多个媒体（media），每个媒体对应多张图像。
    
    参数:
        path: 元数据文件路径
        
    返回:
        templates: 模板 ID 数组
        medias: 媒体 ID 数组
    """
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    """
    读取验证对列表
    
    IJB 数据集的验证任务由成对的模板组成，每对标注为
    同一人（1）或不同人（0）。
    
    参数:
        path: 验证对文件路径
        
    返回:
        t1: 第一个模板 ID 数组
        t2: 第二个模板 ID 数组
        label: 标签数组（1=同一人, 0=不同人）
    """
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


def read_image_feature(path):
    """
    从 pickle 文件读取预提取的图像特征
    
    参数:
        path: 特征文件路径
        
    返回:
        img_feats: 图像特征矩阵
    """
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


def image2template_feature(img_feats=None,
                           templates=None,
                           medias=None):
    """
    将图像级特征聚合为模板级特征
    
    IJB 评测的关键步骤：将同一模板下的多张图像特征聚合成一个模板特征。
    同一媒体（视频）的多帧特征先取平均，然后将不同媒体的特征求和。
    最后进行 L2 归一化。
    
    参数:
        img_feats: 图像特征矩阵 [N, D]
        templates: 模板 ID 数组
        medias: 媒体 ID 数组
        
    返回:
        template_norm_feats: 归一化后的模板特征矩阵
        unique_templates: 唯一模板 ID 数组
    """
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                # 单帧媒体：直接使用该帧特征
                media_norm_feats += [face_norm_feats[ind_m]]
            else:
                # 多帧媒体（视频）：将同一视频的帧特征取平均
                # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], axis=0, keepdims=True), ]
        media_norm_feats = np.array(media_norm_feats)
        # 将同一模板下不同媒体的特征求和
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    # L2 归一化
    template_norm_feats = normalize(template_feats)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    """
    计算验证对的相似度分数
    
    使用模板 ID 映射表将验证对转换为特征索引，
    然后计算余弦相似度（因为特征已归一化，点积即为余弦相似度）。
    
    参数:
        template_norm_feats: 归一化后的模板特征矩阵
        unique_templates: 唯一模板 ID 数组
        p1: 第一个模板 ID 数组
        p2: 第二个模板 ID 数组
        
    返回:
        score: 验证对的相似度分数数组
    """
    # 建立模板 ID 到特征索引的映射表
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))
    total_pairs = np.array(range(len(p1)))
    # 分批计算以避免内存溢出
    batchsize = 100000
    sublists = [total_pairs[i: i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        # 余弦相似度 = 特征点积（特征已归一化）
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    """
    计算验证对的相似度分数（与 verification 功能相同）
    
    注意：此函数与 verification() 完全相同，保留两个版本可能是为了兼容性。
    
    参数:
        template_norm_feats: 归一化后的模板特征矩阵
        unique_templates: 唯一模板 ID 数组
        p1: 第一个模板 ID 数组
        p2: 第二个模板 ID 数组
        
    返回:
        score: 验证对的相似度分数数组
    """
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def main(args):
    """
    IJB 评测主函数
    
    完整的评测流程：
    1. 读取 IJB 元数据（模板、媒体、验证对）
    2. 加载图像列表并创建数据集
    3. 使用 ONNX 模型提取特征
    4. 特征聚合和验证计算
    5. 输出 TPR@FPR 指标表格
    
    参数:
        args: 命令行参数（包含模型路径、图像路径、结果目录等）
    """
    use_norm_score = True  # if Ture, TestMode(N1) —— 使用归一化分数
    use_detector_score = True  # if Ture, TestMode(D1) —— 使用检测器分数加权
    use_flip_test = True  # if Ture, TestMode(F1) —— 使用翻转测试
    assert args.target == 'IJBC' or args.target == 'IJBB'

    # 读取模板-媒体对应关系
    start = timeit.default_timer()
    templates, medias = read_template_media_list(
        os.path.join('%s/meta' % args.image_path, '%s_face_tid_mid.txt' % args.target.lower()))
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    # 读取验证对列表和标签
    start = timeit.default_timer()
    p1, p2, label = read_template_pair_list(
        os.path.join('%s/meta' % args.image_path,
                     '%s_template_pair_label.txt' % args.target.lower()))
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    # 加载图像列表并提取特征
    start = timeit.default_timer()
    img_path = '%s/loose_crop' % args.image_path
    img_list_path = '%s/meta/%s_name_5pts_score.txt' % (args.image_path, args.target.lower())
    img_list = open(img_list_path)
    files = img_list.readlines()
    dataset = AlignedDataSet(root=img_path, lines=files, align=True)
    img_feats = extract(args.model_root, dataset)

    # 读取检测器分数（人脸质量分数）
    faceness_scores = []
    for each_line in files:
        name_lmk_score = each_line.split()
        faceness_scores.append(name_lmk_score[-1])
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))
    start = timeit.default_timer()

    # 翻转测试：合并原始和翻转特征
    if use_flip_test:
        img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2] + img_feats[:, img_feats.shape[1] // 2:]
    else:
        img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

    # 归一化处理
    if use_norm_score:
        img_input_feats = img_input_feats
    else:
        img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))

    # 使用检测器分数加权（质量越高，权重越大）
    if use_detector_score:
        print(img_input_feats.shape, faceness_scores.shape)
        img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
    else:
        img_input_feats = img_input_feats

    # 图像特征 -> 模板特征聚合
    template_norm_feats, unique_templates = image2template_feature(
        img_input_feats, templates, medias)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    # 计算验证对的相似度分数
    start = timeit.default_timer()
    score = verification(template_norm_feats, unique_templates, p1, p2)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    
    # 保存分数结果
    save_path = os.path.join(args.result_dir, "{}_result".format(args.target))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    score_save_file = os.path.join(save_path, "{}.npy".format(args.model_root))
    np.save(score_save_file, score)
    
    # 计算并输出 TPR@FPR 指标表格
    files = [score_save_file]
    methods = []
    scores = []
    for file in files:
        methods.append(os.path.basename(file))
        scores.append(np.load(file))
    methods = np.array(methods)
    scores = dict(zip(methods, scores))
    # FPR 目标值：1e-6 到 1e-1
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    tpr_fpr_table = prettytable.PrettyTable(['Methods'] + [str(x) for x in x_labels])
    for method in methods:
        fpr, tpr, _ = roc_curve(label, scores[method])
        fpr = np.flipud(fpr)  # 翻转以从高 FPR 到低 FPR
        tpr = np.flipud(tpr)  # 选择相同 FPR 下最大的 TPR
        tpr_fpr_row = []
        tpr_fpr_row.append("%s-%s" % (method, args.target))
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(
                list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
        tpr_fpr_table.add_row(tpr_fpr_row)
    print(tpr_fpr_table)


if __name__ == '__main__':
    # 命令行入口：执行 IJB 数据集上的人脸验证评测
    parser = argparse.ArgumentParser(description='do ijb test')
    # general
    parser.add_argument('--model-root', default='', help='path to load model.')
    parser.add_argument('--image-path', default='', type=str, help='')
    parser.add_argument('--result-dir', default='.', type=str, help='')
    parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
    main(parser.parse_args())
