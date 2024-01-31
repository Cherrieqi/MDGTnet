import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class conv_block1(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super(conv_block1, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=padding),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.net(x)

        return y


class conv_block2(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super(conv_block2, self).__init__()
        self.net_block1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=padding),
                                        nn.BatchNorm2d(out_ch),
                                        nn.ReLU(inplace=True))
        self.ca1 = ChannelAttention(out_ch)
        self.net_block2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(out_ch),
                                        nn.ReLU(inplace=True))
        self.net_block3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(out_ch),
                                        nn.ReLU(inplace=True))
        self.ca2 = ChannelAttention(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=padding)

    def forward(self, x):
        y = self.net_block1(x)
        w_1 = self.ca1(y)
        y = y.mul(w_1)
        y = self.net_block2(y)
        w_2 = self.ca2(y)
        y = y.mul(w_2)
        y += self.conv1(x)

        return y


class conv_block3(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super(conv_block3, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=padding),
                                 nn.BatchNorm2d(out_ch),
                                 nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=padding),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.net(x)

        return y


class conv11_block(nn.Module):
    def __init__(self, in_ch):
        super(conv11_block, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, 2*in_ch, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(2*in_ch),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(2*in_ch, in_ch, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(in_ch),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.net(x)

        return y


class conv_block4(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super(conv_block4, self).__init__()
        self.net_main = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=padding),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_ch, 2 * out_ch, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(2*out_ch),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(2 * out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(out_ch))
        self.net_side = nn.Sequential(nn.Conv2d(out_ch, int(out_ch/15), kernel_size=3, stride=1, padding=0),
                                      nn.BatchNorm2d(int(out_ch/15)))
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=padding),
                                  nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_main, x_side):
        y_main = self.net_main(x_main)
        y = y_main + self.conv(x_main) - x_side
        y_side = self.net_side(y_main - x_side)
        y_side = torch.softmax(y_side, dim=1)

        return y, y_side




