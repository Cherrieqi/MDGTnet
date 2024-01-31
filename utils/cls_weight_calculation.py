import torch
import numpy as np


def weight_calc_HSI(label, cls_id: list):
    cls_num = len(cls_id)
    cls_list = [0] * cls_num
    cls_list = [np.uint64(x)+np.sum(label == cls_id[y]) for y, x in enumerate(cls_list)]

    weight: list = [sum(cls_list)/np.uint64(x) for x in cls_list]
    weight = [round(x/sum(weight), 6) for x in weight]

    return torch.tensor(weight).float()






