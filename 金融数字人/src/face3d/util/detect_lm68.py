"""68点人脸关键点检测模块

该脚本用于检测图像中的68个人脸关键点（landmarks）。
使用预训练的 TensorFlow 模型进行关键点检测，
并支持批量处理、可视化和结果保存。
"""
import os                           # 操作系统接口模块
import cv2                          # OpenCV 图像处理库
import numpy as np                  # NumPy 数值计算库
from scipy.io import loadmat        # MATLAB 文件读取
import tensorflow as tf             # TensorFlow 深度学习框架
from util.preprocess import align_for_lm  # 关键点检测的图像预对齐函数
from shutil import move             # 文件移动操作

# 加载标准平均人脸关键点坐标（68个点）
# 用于将检测结果从对齐坐标系转换回原始坐标系
mean_face = np.loadtxt('util/test_mean_face.txt')
mean_face = mean_face.reshape([68, 2])  # 重塑为 (68, 2) 的形状


def save_label(labels, save_path):
    """将关键点标签保存为文本文件。

    参数:
        labels (ndarray): 关键点坐标数组
        save_path (str): 保存路径
    """
    np.savetxt(save_path, labels)


def draw_landmarks(img, landmark, save_name):
    """在图像上绘制关键点并保存可视化结果。

    在每个关键点位置画一个 2x2 的红色方块，
    用于可视化检测到的关键点位置是否正确。

    参数:
        img (ndarray): 原始图像
        landmark (ndarray): 关键点坐标 (68, 2)
        save_name (str): 保存路径
    """
    landmark = landmark
    # 创建与原始图像同样大小的可视化图像
    lm_img = np.zeros([img.shape[0], img.shape[1], 3])
    lm_img[:] = img.astype(np.float32)
    landmark = np.round(landmark).astype(np.int32)  # 坐标取整

    # 在每个关键点位置画红色方块
    for i in range(len(landmark)):
        for j in range(-1, 1):
            for k in range(-1, 1):
                # 边界检查：确保坐标在图像范围内
                if img.shape[0] - 1 - landmark[i, 1]+j > 0 and \
                        img.shape[0] - 1 - landmark[i, 1]+j < img.shape[0] and \
                        landmark[i, 0]+k > 0 and \
                        landmark[i, 0]+k < img.shape[1]:
                    # 绘制红色像素 (BGR格式: [0, 0, 255])
                    lm_img[img.shape[0] - 1 - landmark[i, 1]+j, landmark[i, 0]+k,
                           :] = np.array([0, 0, 255])
    lm_img = lm_img.astype(np.uint8)

    cv2.imwrite(save_name, lm_img)  # 保存可视化图像


def load_data(img_name, txt_name):
    """加载图像和对应的关键点文本文件。

    参数:
        img_name (str): 图像文件路径
        txt_name (str): 关键点文本文件路径

    返回:
        img (ndarray): 图像数据
        landmarks (ndarray): 关键点坐标
    """
    return cv2.imread(img_name), np.loadtxt(txt_name)


# 创建关键点检测器的 TensorFlow 计算图
def load_lm_graph(graph_filename):
    """加载预训练的关键点检测 TensorFlow 模型。

    参数:
        graph_filename (str): TensorFlow 冻结图文件路径 (.pb)

    返回:
        lm_sess: TensorFlow 会话
        img_224: 输入图像张量（224x224x3）
        output_lm: 输出关键点张量
    """
    # 读取冻结的计算图定义
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 创建新的计算图并导入模型
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='net')
        # 获取输入和输出张量的引用
        img_224 = graph.get_tensor_by_name('net/input_imgs:0')     # 输入：224x224 的图像
        output_lm = graph.get_tensor_by_name('net/lm:0')           # 输出：关键点坐标
        lm_sess = tf.Session(graph=graph)  # 创建 TensorFlow 会话

    return lm_sess, img_224, output_lm


# 68点关键点检测主函数
def detect_68p(img_path, sess, input_op, output_op):
    """批量检测图像中的68个人脸关键点。

    处理流程：
    1. 遍历目录中的所有图像
    2. 加载图像和5点关键点
    3. 对齐图像以适配68点检测器
    4. 使用 TensorFlow 模型检测68点关键点
    5. 将结果从对齐坐标系转换回原始坐标系

    参数:
        img_path (str): 包含图像的目录路径
        sess: TensorFlow 会话
        input_op: 输入张量
        output_op: 输出张量
    """
    print('detecting landmarks......')

    # 获取目录中所有图像文件名（支持 jpg/png/jpeg 格式）
    names = [i for i in sorted(os.listdir(
        img_path)) if 'jpg' in i or 'png' in i or 'jpeg' in i or 'PNG' in i]

    # 创建输出子目录
    vis_path = os.path.join(img_path, 'vis')           # 可视化结果目录
    remove_path = os.path.join(img_path, 'remove')      # 无效图像移除目录
    save_path = os.path.join(img_path, 'landmarks')     # 关键点保存目录
    if not os.path.isdir(vis_path):
        os.makedirs(vis_path)
    if not os.path.isdir(remove_path):
        os.makedirs(remove_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i in range(0, len(names)):
        name = names[i]
        print('%05d' % (i), ' ', name)
        full_image_name = os.path.join(img_path, name)
        # 构建5点关键点文件路径
        txt_name = '.'.join(name.split('.')[:-1]) + '.txt'
        full_txt_name = os.path.join(img_path, 'detections', txt_name)

        # 如果图像没有对应的5点关键点检测结果，将其移除
        if not os.path.isfile(full_txt_name):
            move(full_image_name, os.path.join(remove_path, name))
            continue 

        # 加载图像和5点关键点
        img, five_points = load_data(full_image_name, full_txt_name)
        # 对齐图像以便进行68点关键点检测
        input_img, scale, bbox = align_for_lm(img, five_points)

        # 如果对齐失败（scale=0），移除相关文件
        if scale == 0:
            move(full_txt_name, os.path.join(
                remove_path, txt_name))
            move(full_image_name, os.path.join(remove_path, name))
            continue

        # 调整图像形状以适配模型输入 (1, 224, 224, 3)
        input_img = np.reshape(
            input_img, [1, 224, 224, 3]).astype(np.float32)
        # 使用 TensorFlow 模型检测关键点
        landmark = sess.run(
            output_op, feed_dict={input_op: input_img})

        # 将检测结果从对齐坐标系转换回原始图像坐标系
        landmark = landmark.reshape([68, 2]) + mean_face  # 加上平均人脸偏移
        landmark[:, 1] = 223 - landmark[:, 1]              # 翻转y轴
        landmark = landmark / scale                         # 反缩放
        landmark[:, 0] = landmark[:, 0] + bbox[0]          # 加上边界框偏移
        landmark[:, 1] = landmark[:, 1] + bbox[1]
        landmark[:, 1] = img.shape[0] - 1 - landmark[:, 1]  # 再次翻转y轴

        # 每100张图像保存一次可视化结果
        if i % 100 == 0:
            draw_landmarks(img, landmark, os.path.join(vis_path, name))
        # 保存关键点坐标到文本文件
        save_label(landmark, os.path.join(save_path, txt_name))
