import torch
import torch.nn as nn
from networks.baseblock import conv_block3, conv11_block, conv_block2


def trans_fwd(data):
    """
    :param data: [b, c, slice_size, slice_size]
    :return: data_trans: [b, slice_size×slice_size, c, 1]
    """
    b, c, slice_size, __ = data.shape
    data_trans = data.permute(0, 2, 3, 1)
    data_trans = data_trans.reshape(b, slice_size**2, c)
    data_trans = data_trans.unsqueeze(dim=-1)

    return data_trans


def trans_bwd(data, slice_size):
    """
    :param data: [b, slice_size×slice_size, c, 1]
    :param slice_size:
    :return: data_trans: [b, c, slice_size, slice_size]
    """
    b, __, c, __ = data.shape
    data_trans = data.reshape(b, slice_size, slice_size, c)
    data_trans = data_trans.permute(0, 3, 1, 2)

    return data_trans


class AEM(nn.Module):
    def __init__(self, in_ch, padding, slice_size: int, spec_range: list):
        super(AEM, self).__init__()
        self.slice_size = slice_size
        self.spec_range = spec_range
        self.wt_calc = nn.Sequential(conv_block3(in_ch, in_ch, padding=padding),
                                     nn.Softmax(dim=1))

        self.conv1 = conv11_block(in_ch=slice_size ** 2)
        self.conv21 = conv11_block(in_ch=spec_range[0])
        self.conv22 = conv11_block(in_ch=spec_range[1]-spec_range[0])
        if len(self.spec_range) > 2:
            self.conv23 = conv11_block(in_ch=spec_range[2]-spec_range[1])
        else:
            pass

    def forward(self, x):
        tup = ()
        wt = self.wt_calc(x)

        spec1 = x[:, :self.spec_range[0]]
        spec1 = trans_fwd(spec1)
        spec1 = self.conv1(spec1)
        spec1 = trans_bwd(spec1, self.slice_size)
        spec1 = self.conv21(spec1)
        tup = tup + tuple(spec1.unsqueeze(dim=0))

        spec2 = x[:, self.spec_range[0]:self.spec_range[1]]
        spec2 = trans_fwd(spec2)
        spec2 = self.conv1(spec2)
        spec2 = trans_bwd(spec2, self.slice_size)
        spec2 = self.conv22(spec2)
        tup = tup + tuple(spec2.unsqueeze(dim=0))

        if len(self.spec_range) > 2:
            spec3 = x[:, self.spec_range[1]:self.spec_range[2]]
            spec3 = trans_fwd(spec3)
            spec3 = self.conv1(spec3)
            spec3 = trans_bwd(spec3, self.slice_size)
            spec3 = self.conv23(spec3)
            tup = tup + tuple(spec3.unsqueeze(dim=0))
        else:
            pass

        spec = torch.cat(tup, dim=1)
        y = wt*spec

        return y

