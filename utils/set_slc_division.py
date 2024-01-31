import torch


def set_division(class_num, class_list, image_slice, label_slice, mode, rate, row_col):
    """
    :param class_num: Class number to be retained
    :param class_list: Class to be retained, list
    :param image_slice:  [N, c, slice_size, slice_size]
    :param label_slice: [N]
    :param mode: 'train'/'val'/'test' default 'test'
    :param rate:  rate<1：Proportional sampling  rate≥1：Sampling by quantity
    :param row_col: (row, col) in gt, [N, 2]

    :return: image,label:
    """
    N = image_slice.shape[0]
    c = image_slice.shape[1]
    slice_size = image_slice.shape[2]

    image_idx = torch.full([class_num, N], 0)
    class_len = torch.full([class_num], 0)

    for p, cls in enumerate(class_list):
        a = 0
        for i in range(N):
            if label_slice[i] == cls:
                a = a + 1
                image_idx[p, a - 1] = i
                class_len[p] += 1

    # train/val/test
    image_ex = torch.full([N, c, slice_size, slice_size], 0.).double()
    label_ex = torch.full([N], 0)
    row_col_ex = torch.full([N, 2], 0)
    a = 0
    if mode == 'train':  # train
        for i in range(class_num):
            if rate < 1:
                for j in range(0, int(rate * class_len[i])):
                    a = a + 1
                    image_ex[a - 1] = image_slice[image_idx[i, j]]
                    label_ex[a - 1] = class_list[i]
                    row_col_ex[a - 1] = row_col[image_idx[i, j]]
            else:
                for j in range(0, rate):
                    a = a + 1
                    image_ex[a - 1] = image_slice[image_idx[i, j]]
                    label_ex[a - 1] = class_list[i]
                    row_col_ex[a - 1] = row_col[image_idx[i, j]]

    elif mode == 'val':  # val
        for i in range(class_num):
            if rate < 1:
                for j in range(int(rate * class_len[i]), int(rate*2 * class_len[i])):
                    a = a + 1
                    image_ex[a - 1] = image_slice[image_idx[i, j]]
                    label_ex[a - 1] = class_list[i]
                    row_col_ex[a - 1] = row_col[image_idx[i, j]]
            else:
                if rate * 2 >= class_len[i]:  # There aren't enough samples left to take
                    for j in range(rate, class_len[i]):
                        a = a + 1
                        image_ex[a - 1] = image_slice[image_idx[i, j]]
                        label_ex[a - 1] = class_list[i]
                        row_col_ex[a - 1] = row_col[image_idx[i, j]]
                    for j in range(class_len[i], int(rate*2)):
                        a = a + 1
                        image_ex[a - 1] = image_slice[image_idx[i, j-class_len[i]]]
                        label_ex[a - 1] = class_list[i]
                        row_col_ex[a - 1] = row_col[image_idx[i, j]]

                else:
                    for j in range(rate, rate * 2):
                        a = a + 1
                        image_ex[a - 1] = image_slice[image_idx[i, j]]
                        label_ex[a - 1] = class_list[i]
                        row_col_ex[a - 1] = row_col[image_idx[i, j]]

    else:  # test
        for i in range(class_num):
            if rate < 1:
                for j in range(int(rate * class_len[i]), class_len[i]):
                    a = a + 1
                    image_ex[a - 1] = image_slice[image_idx[i, j]]
                    label_ex[a - 1] = class_list[i]
                    row_col_ex[a - 1] = row_col[image_idx[i, j]]
            else:
                for j in range(rate, class_len[i]):
                    a = a + 1
                    image_ex[a - 1] = image_slice[image_idx[i, j]]
                    label_ex[a - 1] = class_list[i]
                    row_col_ex[a - 1] = row_col[image_idx[i, j]]

    num = a  # train/val/test sum

    # [num, c, slice_size, slice_size]
    image = image_ex[:num]
    label = label_ex[:num]
    point_idx = row_col_ex[:num]

    return image, label, point_idx


def set_division_pro(class_num, class_list, image_slice, label_slice, mode, rate: list):
    """
    :param class_num: Class number to be retained
    :param class_list: Class to be retained, list
    :param image_slice:  [N, c, slice_size, slice_size]
    :param label_slice: [N]
    :param mode: 'train'/'val'/'test' default 'test'
    :param rate:  rate<1：Proportional sampling  rate≥1：Sampling by quantity

    :return: image,label:
    """
    N = image_slice.shape[0]
    c = image_slice.shape[1]
    slice_size = image_slice.shape[2]

    image_idx = torch.full([class_num, N], 0)
    class_len = torch.full([class_num], 0)

    for p, cls in enumerate(class_list):
        a = 0
        for i in range(N):
            if label_slice[i] == cls:
                a = a + 1
                image_idx[p, a - 1] = i
                class_len[p] += 1

    # train/val/test
    image_ex = torch.full([N, c, slice_size, slice_size], 0.).double()
    label_ex = torch.full([N], 0)
    a = 0
    if mode == 'train':  # train
        for i in range(class_num):
            if rate[i] < 1:
                for j in range(0, int(rate[i] * class_len[i])):
                    a = a + 1
                    image_ex[a - 1] = image_slice[image_idx[i, j]]
                    label_ex[a - 1] = class_list[i]
            else:
                for j in range(0, rate[i]):
                    a = a + 1
                    image_ex[a - 1] = image_slice[image_idx[i, j]]
                    label_ex[a - 1] = class_list[i]

    elif mode == 'val':  # val
        for i in range(class_num):
            if rate[i] < 1:
                for j in range(int(rate[i] * class_len[i]), int(rate*2 * class_len[i])):
                    a = a + 1
                    image_ex[a - 1] = image_slice[image_idx[i, j]]
                    label_ex[a - 1] = class_list[i]
            else:
                if rate[i] * 2 >= class_len[i]:  # There aren't enough samples left to take
                    for j in range(rate[i], class_len[i]):
                        a = a + 1
                        image_ex[a - 1] = image_slice[image_idx[i, j]]
                        label_ex[a - 1] = class_list[i]
                    for j in range(class_len[i], int(rate[i]*2)):
                        a = a + 1
                        image_ex[a - 1] = image_slice[image_idx[i, j-class_len[i]]]
                        label_ex[a - 1] = class_list[i]

                else:
                    for j in range(rate[i], rate[i] * 2):
                        a = a + 1
                        image_ex[a - 1] = image_slice[image_idx[i, j]]
                        label_ex[a - 1] = class_list[i]

    else:  # test
        for i in range(class_num):
            if rate[i] < 1:
                for j in range(int(rate[i] * class_len[i]), class_len[i]):
                    a = a + 1
                    image_ex[a - 1] = image_slice[image_idx[i, j]]
                    label_ex[a - 1] = class_list[i]
            else:
                for j in range(rate[i], class_len[i]):
                    a = a + 1
                    image_ex[a - 1] = image_slice[image_idx[i, j]]
                    label_ex[a - 1] = class_list[i]

    num = a  # train/val/test sum

    # [num, c, slice_size, slice_size]
    image = image_ex[:num]
    label = label_ex[:num]

    return image, label

