import torch


def data_split(slice_size, image, label):
    c = image.shape[0]
    h = image.shape[1]
    w = image.shape[2]

    point_id = []

    image_ex = torch.full([c, h + slice_size - 1, w + slice_size - 1], 0.)  # 扩展后的高光谱图像
    image_ex[:, (slice_size - 1) // 2:(slice_size - 1) // 2 + h,
             (slice_size - 1) // 2:(slice_size - 1) // 2 + w] = image

    image_slice = torch.full([h * w, c, slice_size, slice_size], 0.)  # 高光谱图像数据的切片
    label_slice = torch.full([h * w], 0)
    a = 0
    for i in range(h):
        for j in range(w):
            a = a + 1
            image_slice[a - 1] = image_ex[:, i: i + slice_size, j: j + slice_size]
            if label is not None:
                label_slice[a - 1] = label[i, j]
                point_id.append([i, j])
    point_id = torch.tensor(point_id)
    return image_slice, label_slice, point_id
