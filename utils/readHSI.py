import torch
import scipy.io as sio
import h5py
import numpy as np


def readHSI(path, image_name, label_name, mode, img_order):
    """
    :param path:
    :param image_name:
    :param label_name:
    :param mode: ’0‘ scipy   ’1‘ h5py
    :param img_order: list ,TO [c, h, w]
    :return: image: tensor，[c,h,w]
             label: tensor，[h,w]
    """
    label = 0
    if mode == 0:
        image = sio.loadmat(path + image_name + '.mat')  # 导入样本数据
        if label_name is not None:
            label = sio.loadmat(path + label_name + '.mat')  # 导入真值
    elif mode == 1:
        image = h5py.File(path + image_name + '.mat')  # 导入样本数据
        if label_name is not None:
            label = h5py.File(path + label_name + '.mat')  # 导入真值
    else:
        print('not exist')

    image = image[tuple(image.keys())[-1]]
    if label_name is not None:
        label = label[tuple(label.keys())[-1]]

    image = np.array(image)
    if label_name is not None:
        label = np.array(label)
    image = image.astype(np.float)
    if label_name is not None:
        label = label.astype(np.float)

    image = torch.from_numpy(image)
    if label_name is not None:
        label = torch.from_numpy(label)
        # label = torch.tensor(label, dtype=torch.long)

    # ori --> [c, h, w]
    image = image.permute(img_order[0], img_order[1], img_order[2])

    return image, label


def label_trans(label, label_list, label_list_new):
    """
    :param label: [h, w]
    :param label_list:
    :param label_list_new:
    :return:
    """
    h = label.shape[0]
    w = label.shape[1]

    cls_num = len(set(label_list))   # 要转换的类别数量
    label_new = torch.full([h, w], 0, dtype=torch.long)
    trans_id_row = []
    trans_id_col = []
    for i in range(cls_num):
        id_row, id_col = np.where(label == label_list[i])  # 得到所需标签值的对应位置
        trans_id_row.append(id_row)
        trans_id_col.append(id_col)

    for i in range(cls_num):
        label_new[trans_id_row[i], trans_id_col[i]] = label_list_new[i]

    return label_new


def one_hot_slice(label_slice, class_num):
    """
    :param label_slice: [N] torch.long
    :param class_num:
    :return:
    """
    label_slice_OH = torch.full([len(label_slice), class_num], 0.)
    for cls in range(class_num):
        idx = np.where(label_slice == cls + 1)
        label_slice_OH[idx, cls] = 1.

    return label_slice_OH

