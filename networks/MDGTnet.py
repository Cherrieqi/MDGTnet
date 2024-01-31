import torch.nn as nn
from networks.baseblock import conv_block1, conv_block4, conv_block2
from networks.intra_DUEB import AEM


class IFEM(nn.Module):
    def __init__(self, in_ch, out_ch: list, padding=0):
        super(IFEM, self).__init__()
        self.block1 = conv_block4(in_ch, out_ch[0], padding=padding)
        self.block2 = conv_block4(out_ch[0], out_ch[1], padding=0)
        self.block3 = conv_block4(out_ch[1], out_ch[2], padding=0)
        self.block4 = conv_block4(out_ch[2], out_ch[3], padding=0)

    def forward(self, x, x_side: list):
        y_main_1, y_side_1 = self.block1(x, x_side[0])
        y_main_2, y_side_2 = self.block2(y_main_1, x_side[1])
        y_main_3, y_side_3 = self.block3(y_main_2, x_side[2])
        y_main_4, y_side_4 = self.block4(y_main_3, x_side[3])

        return y_main_1, y_main_2, y_main_3, y_main_4, y_side_1, y_side_2, y_side_3, y_side_4


class MDGTnet(nn.Module):
    def __init__(self, in_ch, out_ch: list, padding, slice_size: int, spec_range: list, class_num: int):
        super(MDGTnet, self).__init__()

        self.AEM = AEM(in_ch, padding, slice_size, spec_range)
        self.intra_1 = conv_block2(in_ch, out_ch[0], padding=padding)
        self.intra_2 = conv_block2(out_ch[0], out_ch[1])
        self.intra_3 = conv_block2(out_ch[1], out_ch[2])
        self.intra_4 = conv_block2(out_ch[2], out_ch[3])

        self.IFEH = nn.Sequential(conv_block1(in_ch, out_ch[4]),
                                  conv_block1(out_ch[4], out_ch[5]))
        self.IFEM = IFEM(out_ch[5], [out_ch[0], out_ch[1], out_ch[2], out_ch[3]], padding=padding)

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(slice_size*slice_size*out_ch[3], out_ch[6]),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(out_ch[6], out_ch[7]),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(out_ch[7], class_num))

    def forward(self, x_intra, x_inter):
        y_intra = self.AEM(x_intra)
        y_intra_1 = self.intra_1(y_intra)
        y_intra_2 = self.intra_2(y_intra_1)
        y_intra_3 = self.intra_3(y_intra_2)
        y_intra_4 = self.intra_4(y_intra_3)
        y_inter = self.IFEH(x_inter)
        x_side = [y_intra_1, y_intra_2, y_intra_3, y_intra_4]
        __, __, __, y_inter_4, y_side_1, y_side_2, y_side_3, y_side_4 = self.IFEM(y_inter, x_side)
        y = self.classifier(y_inter_4).squeeze()

        return y, y_side_1, y_side_2, y_side_3, y_side_4

