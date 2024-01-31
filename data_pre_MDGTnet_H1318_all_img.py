import gc
import numpy as np
import torch

from utils.data_split import data_split
from utils.normHSI import normHSI_all
from utils.readHSI import readHSI, label_trans, one_hot_slice
from utils.set_slc_division import set_division

rate_test = 0.999999999
slice_size = 3
class_num = 4
norm_rate = 1

# readHSI --> normHSI_all --> label_trans --> data_split --> set_division

# PaviaU
path = './data/raw/PaviaU/'
image_name = 'paviaU'
label_name = 'paviaU_gt'
image_PU, label_PU = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])

image_PU_norm = normHSI_all(image_PU, norm_rate)
del image_PU
gc.collect()

image_PU_new = torch.full((144, image_PU_norm.shape[1], image_PU_norm.shape[2]), 0.)
image_PU_new[5:5+103] = image_PU_norm

PU_label = label_trans(label_PU, [2, 6, 4, 1], [1, 2, 3, 4])
del label_PU
gc.collect()

PU_image_slice, PU_label_slice, PU_row_col = data_split(slice_size, image_PU_new, PU_label)
del image_PU_norm, PU_label
gc.collect()


PU_image, PU_gt, PU_point_idx = set_division(5, [0, 1, 2, 3, 4], PU_image_slice, PU_label_slice, 'train', rate_test, PU_row_col)
del PU_image_slice, PU_label_slice
gc.collect()

PU_gt_OH = one_hot_slice(PU_gt, class_num=class_num+1)
del PU_gt
gc.collect()

np.save("./data/MDGTnet_H1318/gen_PU_all_img/img_norm_all.npy", PU_image)
del PU_image
gc.collect()

np.save("./data/MDGTnet_H1318/gen_PU_all_img/gt_norm_all.npy", torch.cat((PU_gt_OH, PU_point_idx), dim=1))
del PU_gt_OH, PU_point_idx



