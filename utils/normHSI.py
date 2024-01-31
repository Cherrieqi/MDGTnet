import torch


def normHSI_all(image, rate=1):
    """
    :param image: tensor,[c, h, w]
    :param rate:
    :return: image_norm: tensor,[c, h, w]
    """
    max_value = torch.max(image)
    min_value = torch.min(image)
    image_norm = rate * (image - min_value) / (max_value - min_value)

    return image_norm


def normHSI_smp_s(image, rate=1, eps=0.00000000001):
    """
    :param image: tensor,[N, c, slice_size, slice_size]
    :param rate:
    :param eps:
    :return: image_norm: tensor,[N, c, slice_size, slice_size]
    """
    image_norm = torch.full(image.shape, 0.)
    for i in range(image.shape[0]):
        max_value = torch.max(image[i], dim=0).values
        min_value = torch.min(image[i], dim=0).values
        image_norm[i] = rate * (image[i] - min_value) / (max_value - min_value+eps)

    return image_norm

