import cv2
import numpy as np


def label_vision_1d(label, row_cols, row, col, save_path):
    """
    :param label:  [N]
    :param row_cols:[N,2]
    :param save_path: str
    :return:
    """
    color_list = [(0, 66, 198), (255, 100, 187), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                  (0, 255, 255), (3, 144, 188)]
    label_rgb = np.zeros((row, col, 3), dtype=np.uint8)
    for j, i in enumerate(label):
        x, y = row_cols[j]

        label_rgb[x][y][0] = color_list[i][0]
        label_rgb[x][y][1] = color_list[i][1]
        label_rgb[x][y][2] = color_list[i][2]

    cv2.imwrite(save_path, label_rgb)

















