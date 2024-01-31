import gc
import numpy as np
import torch

from utils.data_split import data_split
from utils.normHSI import normHSI_all
from utils.readHSI import readHSI, label_trans, one_hot_slice
from utils.set_slc_division import set_division, set_division_pro

rate_train_1 = [5000, 0.99999999, 3000, 0.999999999]
rate_train_2 = [0.99999999, 0.99999999, 6000, 4000]
rate_test = 0.999999999
slice_size = 3
class_num = 4
norm_rate = 1

# readHSI --> normHSI_all --> label_trans --> data_split --> set_division

# Houston + PU
# PaviaU
path = './data/raw/PaviaU/'
image_name = 'paviaU'
label_name = 'paviaU_gt'
image_PU, label_PU = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])

image_PU_norm = normHSI_all(image_PU, norm_rate)
del image_PU
gc.collect()

image_PU_new = image_PU_norm[1:]
del image_PU_norm
gc.collect()

PU_label = label_trans(label_PU, [2, 6, 4, 1], [1, 2, 3, 4])
del label_PU
gc.collect()

PU_image_slice, PU_label_slice, __ = data_split(slice_size, image_PU_new, PU_label)
del image_PU_new, PU_label
gc.collect()

PU_image, PU_gt = set_division_pro(4, [1, 2, 3, 4], PU_image_slice, PU_label_slice, 'train', rate_train_1)
del PU_image_slice, PU_label_slice
gc.collect()

PU_gt_OH = one_hot_slice(PU_gt, class_num=class_num)
del PU_gt
gc.collect()

np.save("data/MDGTnet_PUPC/gen_PU/img_norm_all.npy", PU_image)
del PU_image
gc.collect()
np.save("data/MDGTnet_PUPC/gen_PU/gt_norm_all.npy", PU_gt_OH)
del PU_gt_OH
gc.collect()

# PaviaC
path = './data/raw/PaviaC/'
image_name = 'pavia'
label_name = 'pavia_gt'
image_PC, label_PC = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])

image_PC_norm = normHSI_all(image_PC, norm_rate)
del image_PC
gc.collect()

PC_label = label_trans(label_PC, [3, 9, 2, 6], [1, 2, 3, 4])
del label_PC
gc.collect()

PC_image_slice, PC_label_slice, __ = data_split(slice_size, image_PC_norm, PC_label)
del image_PC_norm, PC_label
gc.collect()

PC_image, PC_gt = set_division_pro(4, [1, 2, 3, 4], PC_image_slice, PC_label_slice, 'train', rate_train_2)
del PC_image_slice, PC_label_slice
gc.collect()

PC_gt_OH = one_hot_slice(PC_gt, class_num=class_num)
del PC_gt
gc.collect()

np.save("data/MDGTnet_PUPC/gen_PC/img_norm_all.npy", PC_image)
del PC_image
gc.collect()
np.save("data/MDGTnet_PUPC/gen_PC/gt_norm_all.npy", PC_gt_OH)
del PC_gt_OH
gc.collect()


# Houston2013
path = './data/raw/Houston2013/'
image_name = 'Houston'
label_name = 'Houston_gt'
image_H13, label_H13 = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])

image_H13_norm = normHSI_all(image_H13, norm_rate)
del image_H13
gc.collect()

image_H13_new = image_H13_norm[6:6+102]
del image_H13_norm
gc.collect()

H13_label = label_trans(label_H13, [1, 5, 4, 9], [1, 2, 3, 4])
del label_H13
gc.collect()

H13_image_slice, H13_label_slice, H13_row_col = data_split(slice_size, image_H13_new, H13_label)
del image_H13_new, H13_label
gc.collect()

H13_image, H13_gt, H13_point_id = set_division(4, [1, 2, 3, 4], H13_image_slice, H13_label_slice, 'train', rate_test, H13_row_col)
del H13_image_slice, H13_label_slice
gc.collect()

H13_gt_OH = one_hot_slice(H13_gt, class_num=class_num)
del H13_gt
gc.collect()

np.save("data/MDGTnet_PUPC/gen_H13/img_norm_all.npy", H13_image)
del H13_image
gc.collect()
np.save("data/MDGTnet_PUPC/gen_H13/gt_norm_all.npy", torch.cat((H13_gt_OH, H13_point_id), dim=1))
del H13_gt_OH
gc.collect()


# Houston2018
path = './data/raw/Houston2018/'
image_name = 'HoustonU'
label_name = 'HoustonU_gt'
image_H18_raw, label_H18 = readHSI(path, image_name, label_name, mode=1, img_order=[0, 1, 2])
image_H18 = torch.full([144, image_H18_raw.shape[1], image_H18_raw.shape[2]], 0.0)
image_H18[0:144:3] = image_H18_raw[:48]
image_H18[1:144:3] = image_H18_raw[:48]
image_H18[2:144:3] = image_H18_raw[:48]

image_H18_norm = normHSI_all(image_H18, norm_rate)
del image_H18
gc.collect()

image_H18_new = image_H18_norm[4:4+102]
del image_H18_norm
gc.collect()

H18_label = label_trans(label_H18, [1, 6, 4, 10], [1, 2, 3, 4])
del label_H18
gc.collect()

H18_image_slice, H18_label_slice, H18_row_col = data_split(slice_size, image_H18_new, H18_label)
del image_H18_new, H18_label
gc.collect()

H18_image, H18_gt, H18_point_id = set_division(4, [1, 2, 3, 4], H18_image_slice, H18_label_slice, 'train', rate_test, H18_row_col)
del H18_image_slice, H18_label_slice
gc.collect()

H18_gt_OH = one_hot_slice(H18_gt, class_num=class_num)
del H18_gt
gc.collect()


np.save("data/MDGTnet_PUPC/gen_H18/img_norm_all.npy", H18_image)
del H18_image
gc.collect()
np.save("data/MDGTnet_PUPC/gen_H18/gt_norm_all.npy", torch.cat((H18_gt_OH, H18_point_id), dim=1))
del H18_gt_OH
gc.collect()




