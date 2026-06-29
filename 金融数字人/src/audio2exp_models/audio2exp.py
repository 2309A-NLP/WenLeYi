# audio2exp.py - 将声音转换成面部表情参数的模块
# 作用：从音频中提取特征，预测出64个数值，用来控制3D人脸模型做出相应的表情
# 这是SadTalker系统中“音频驱动表情”部分的核心推理代码

from tqdm import tqdm
import torch
from torch import nn


class Audio2Exp(nn.Module):
    """
    音频到表情的神经网络模型。
    输入一段声音的梅尔频谱图（声音的一种特征图），
    再结合一个参考帧的表情信息（比如初始表情），
    预测每一帧应该有的64个表情系数（即控制3D人脸表情的参数）。

    构造参数：
        netG: 核心生成器网络（通常是SimpleWrapperV2），真正做预测的模型
        cfg: 配置文件，存着各种超参数
        device: 跑在CPU还是GPU上（如'cuda'或'cpu'）
        prepare_training_loss: 是否准备训练损失计算（目前没用到，忽略即可）
    """

    def __init__(self, netG, cfg, device, prepare_training_loss=False):
        super(Audio2Exp, self).__init__()
        self.cfg = cfg  # 保存配置信息
        self.device = device  # 保存计算设备
        self.netG = netG.to(device)  # 将生成器网络移动到指定设备

    def test(self, batch):
        """
        推理函数：给一批数据，预测所有帧的表情系数。

        输入batch（一个字典）包含：
            - 'indiv_mels': 每一帧的梅尔频谱图，形状为 [bs, T, 1, 80, 16]
                bs: 批次大小（一次处理几个视频/音频）
                T: 总帧数（时间长度）
                80x16: 梅尔频谱图的高和宽（固定尺寸）
            - 'ref': 参考帧的表情系数，只取前64维（即只取表情部分，不含姿态等）
            - 'ratio_gt': 每一帧的表情强度比例因子（控制表情幅度）

        输出：
            - 'exp_coeff_pred': 预测出的表情系数，形状为 [bs, T, 64]
        """

        # 获取梅尔频谱输入，形状为 [批次大小, 总帧数, 1, 80, 16]
        mel_input = batch['indiv_mels']  # bs T 1 80 16
        bs = mel_input.shape[0]  # 批次大小
        T = mel_input.shape[1]  # 总帧数

        # 用来存放每10帧一组预测出的表情系数
        exp_coeff_pred = []

        # 每10帧一组分批预测，这么做的原因是为了节省显存，防止GPU爆显存，这在视频生成任务中是非常常见的‘分块推理’策略。
        # 循环每次取连续的10帧
        for i in tqdm(range(0, T, 10), 'audio2exp:'):  # every 10 frames

            # 取出当前的10帧梅尔频谱
            current_mel_input = mel_input[:, i:i + 10]

            # 取出对应的参考表情系数（也是取这10帧的）
            # ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch['ref'][:, :, :64][:, i:i + 10]

            # 取出对应的比例因子
            ratio = batch['ratio_gt'][:, i:i + 10]  # bs T

            # 将10帧的梅尔图展平，合并批次和帧数，变成 [批次*10, 1, 80, 16] 以便网络处理
            audiox = current_mel_input.view(-1, 1, 80, 16)  # bs*T 1 80 16

            # 用生成器网络预测这10帧的表情系数，输出形状为 [批次, 10, 64]
            curr_exp_coeff_pred = self.netG(audiox, ref, ratio)  # bs T 64

            # 把这一组的结果存起来
            exp_coeff_pred += [curr_exp_coeff_pred]

        # 把各组的预测结果按帧顺序拼接起来，得到完整的 [批次, 总帧数, 64]
        results_dict = {
            'exp_coeff_pred': torch.cat(exp_coeff_pred, axis=1)
        }
        return results_dict