import torch
import torch.nn as nn
from torch import linalg as LA


# similarity calculate
class SimiValue(nn.Module):
    def __init__(self):
        super(SimiValue, self).__init__()

    def forward(self, feature_1, feature_2):
        """
        :param feature_1:  [b, N, 1, 1]
        :param feature_2:  [b, N, 1, 1]
        :return:
        """
        # cosine
        norm_1 = LA.norm(feature_1, dim=1)
        norm_2 = LA.norm(feature_2, dim=1)

        dot_12 = torch.sum(feature_1 * feature_2, dim=1).cuda()
        similarity = dot_12 / (norm_1 * norm_2)
        # print(torch.min(similarity).item(), torch.max(similarity).item())
        return similarity


# loss2 DiffLoss
class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, similarity, cons_flag):
        """
        :param similarity: 余弦相似度
        :param cons_flag: [batch_size] tensor 1/-1 正/负样本的标记
        :return:
        """
        diff_vec = -1 * similarity * cons_flag
        diff_loss = torch.mean(diff_vec)

        return diff_loss


# loss3 SDPloss
class SDPloss(nn.Module):
    def __init__(self):
        super(SDPloss, self).__init__()

    def forward(self, similarity_all, cons_flag, rate=0.2):
        """
        :param similarity_all: tuple ([batch_size], [batch_size], ...) 不同层特征的相似度的集合 按由从浅到深排列
        :param cons_flag: [batch_size] 1/-1 正/负样本的标记
        :param rate: 参考值比例
        :return:
        """
        # 每个batch计算loss
        pair_num = len(similarity_all)
        layer_weight = torch.tensor(range(pair_num))/pair_num

        prog_vec = 0
        for i in range(pair_num-1):
            prog_vec += (similarity_all[i]-similarity_all[i+1]-cons_flag*rate*layer_weight[i+1]) *\
                        (similarity_all[i]-similarity_all[i+1]-cons_flag*rate*layer_weight[i+1])

        # loss
        prog_loss = torch.mean(prog_vec)

        return prog_loss


