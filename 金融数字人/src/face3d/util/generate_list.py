"""训练列表文件生成模块

该脚本用于为 Deep3DFaceRecon_pytorch 生成训练/验证数据的列表文件。
列表文件包含图像、掩码和关键点的路径，供数据集类加载使用。
"""
import os                           # 操作系统接口模块


# 将文件列表写入指定路径
def write_list(lms_list, imgs_list, msks_list, mode='train', save_folder='datalist', save_name=''):
    """将关键点、图像和掩码的文件路径列表写入文本文件。

    生成三个文本文件：
    - landmarks.txt: 关键点文件路径列表
    - images.txt: 图像文件路径列表
    - masks.txt: 掩码文件路径列表

    参数:
        lms_list (list): 关键点文件路径列表
        imgs_list (list): 图像文件路径列表
        msks_list (list): 掩码文件路径列表
        mode (str): 数据模式（'train' 或 'val'）
        save_folder (str): 保存文件夹名称
        save_name (str): 文件名前缀
    """
    # 构建保存路径，例如 'datalist/train/'
    save_path = os.path.join(save_folder, mode)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)  # 如果目录不存在则创建

    # 写入关键点文件列表
    with open(os.path.join(save_path, save_name + 'landmarks.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in lms_list])

    # 写入图像文件列表
    with open(os.path.join(save_path, save_name + 'images.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in imgs_list])

    # 写入掩码文件列表
    with open(os.path.join(save_path, save_name + 'masks.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in msks_list])   


# 检查文件路径列表的有效性
def check_list(rlms_list, rimgs_list, rmsks_list):
    """检查三组文件路径列表中对应文件是否都存在。

    遍历所有路径，验证关键点文件、图像文件和掩码文件
    是否同时存在，只保留三者都存在的有效条目。

    参数:
        rlms_list (list): 关键点文件路径列表（原始）
        rimgs_list (list): 图像文件路径列表（原始）
        rmsks_list (list): 掩码文件路径列表（原始）

    返回:
        lms_list (list): 有效的关键点文件路径列表
        imgs_list (list): 有效的图像文件路径列表
        msks_list (list): 有效的掩码文件路径列表
    """
    lms_list, imgs_list, msks_list = [], [], []
    for i in range(len(rlms_list)):
        flag = 'false'  # 标记文件是否有效
        lm_path = rlms_list[i]
        im_path = rimgs_list[i]
        msk_path = rmsks_list[i]
        # 检查三个文件是否都存在
        if os.path.isfile(lm_path) and os.path.isfile(im_path) and os.path.isfile(msk_path):
            flag = 'true'
            lms_list.append(rlms_list[i])
            imgs_list.append(rimgs_list[i])
            msks_list.append(rmsks_list[i])
        print(i, rlms_list[i], flag)  # 打印检查结果
    return lms_list, imgs_list, msks_list
