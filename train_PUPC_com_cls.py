import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
import gc
from tqdm import tqdm
import random

from ImageDataset_MDGTnet_PUPC_com_cls import ImgDataset_train
from networks.MDGTnet import MDGTnet
from loss import SimiValue, DiffLoss, SDPloss

from utils.draw_loss_curve import draw_loss_curve
from utils.cls_weight_calculation import weight_calc_HSI
from sklearn.metrics import accuracy_score
from utils.lr_adjust import lr_adj

seed = 6

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

f = open("./logs/train_logs.txt", 'w')
f.write("Training loss logs:")
f.write("\n")
f.close()

# set model paras
in_ch = 102
out_ch = [500, 750, 500, 500, 500, 500, 300, 150]
spec_range = [65, 102]
padding = 0
class_num = 4
slice_size = 3
batch_size = 1024
num_epoch = 10
learning_rate = 0.06
device = "cuda:0"
# device = "cpu"
w = [1, 3, 10]

train_loss_list = []

# load train data PU or PC
img_1 = np.load("data/MDGTnet_PUPC/gen_PU/img_norm_all.npy")
img_1 = torch.from_numpy(img_1).float()
label_1 = np.load("data/MDGTnet_PUPC/gen_PU/gt_norm_all.npy")

img_2 = np.load("data/MDGTnet_PUPC/gen_PC/img_norm_all.npy")
img_2 = torch.from_numpy(img_2).float()
label_2 = np.load("data/MDGTnet_PUPC/gen_PC/gt_norm_all.npy")

label_1 = torch.LongTensor(label_1)
label_2 = torch.LongTensor(label_2)

data_and_label = (img_1, label_1, img_2, label_2)
del img_1, img_2, label_1, label_2

train_set = ImgDataset_train(*data_and_label)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

del data_and_label
gc.collect()

# calculate the number of samples
label_all = []
for i, data in enumerate(train_loader):
    for j in range(data[2].shape[0]):
        label_all.append(data[2][j].tolist())
        label_all.append(data[5][j].tolist())

label_all = torch.tensor(label_all).argmax(dim=1) + 1
label_all = label_all.numpy()
weight_cls = weight_calc_HSI(label_all, cls_id=list(range(1, class_num + 1)))
print(weight_cls)


# define and load model
model = MDGTnet(in_ch=in_ch, out_ch=out_ch, padding=padding, slice_size=slice_size, spec_range=spec_range,
                class_num=class_num).to(device)

# define loss functions
loss_classify = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=weight_cls.to(device))
simi_cal = SimiValue()
loss_diff = DiffLoss()
loss_prog = SDPloss()

# train
time_start = time.time()

for epoch in range(num_epoch):
    f = open("./logs/train_logs.txt", 'a')
    epoch_start_time = time.time()
    train_loss = 0.0

    model.train()

    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    loop_len = len(loop)
    for i, data in loop:
        num_iters = epoch * loop_len + i + 1
        optimizer, learning_rate = lr_adj(num_iters, learning_rate, model)
        optimizer.zero_grad()

        # img_all_1, label_all_1, img_all_2, label_all_2
        y_out_1, side_1_1, side_1_2, side_1_3, side_1_4 = model(data[1].to(device), data[0].to(device))
        y_out_2, side_2_1, side_2_2, side_2_3, side_2_4 = model(data[4].to(device), data[3].to(device))

        cons_flag = torch.full([data[0].shape[0]], -1).to(device)
        for batch in range(data[0].shape[0]):
            cons = data[2][batch].equal(data[5][batch])
            if cons:
                cons_flag[batch] = 1

        # loss1
        bce_weight_1 = torch.full(data[2].size(), 1, device=device)
        bce_weight_1[data[2] > 0.5] = 2
        bce_weight_2 = torch.full(data[5].size(), 1, device=device)
        bce_weight_2[data[5] > 0.5] = 2

        loss1_1 = loss_classify(y_out_1, data[2].float().to(device))
        loss1_2 = loss_classify(y_out_2, data[5].float().to(device))

        loss1 = 1 * torch.mean(bce_weight_1 * loss1_1) + 1 * torch.mean(bce_weight_2 * loss1_2)

        # calculate the similarity between the two domains
        simi_1 = simi_cal(side_1_1, side_2_1)
        simi_2 = simi_cal(side_1_2, side_2_2)
        simi_3 = simi_cal(side_1_3, side_2_3)
        simi_4 = simi_cal(side_1_4, side_2_4)

        similarity_all = [simi_1, simi_2, simi_3, simi_4]

        # loss2
        loss2_1 = loss_diff(simi_1, cons_flag)
        loss2_2 = loss_diff(simi_2, cons_flag)
        loss2_3 = loss_diff(simi_3, cons_flag)
        loss2_4 = loss_diff(simi_4, cons_flag)

        loss2 = 1 * loss2_1 + 1 * loss2_2 + 1 * loss2_3 + 1 * loss2_4

        # loss3
        loss3 = loss_prog(similarity_all, cons_flag)

        batch_loss = w[0] * loss1 + w[1] * loss2 + w[2] * loss3

        batch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += batch_loss.item()
            gt_1 = data[2].argmax(dim=1).flatten().cpu().numpy()
            pred_1 = y_out_1.argmax(dim=1).flatten().cpu().numpy()
            gt_2 = data[5].argmax(dim=1).flatten().cpu().numpy()
            pred_2 = y_out_2.argmax(dim=1).flatten().cpu().numpy()
            oa_1 = accuracy_score(gt_1, pred_1)
            oa_2 = accuracy_score(gt_2, pred_2)

        loop.set_description(f'Epoch [{epoch + 1}/{num_epoch}]')
        loop.set_postfix(classify_loss=loss1.item(), diff_loss=loss2.item(), prog_loss=loss3.item(),
                         batch_loss=batch_loss.item(), lr=optimizer.state_dict()['param_groups'][0]['lr'],
                         oa_1=oa_1, oa_2=oa_2)
        optimizer.zero_grad()

    torch.save(model.state_dict(), './models/MDGTnet_PUPC/model{}.pth'.format(epoch))

    print('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f' %
          (epoch + 1, num_epoch, time.time() - epoch_start_time, train_loss))

    f.write('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f \n' %
            (epoch + 1, num_epoch, time.time() - epoch_start_time, train_loss))
    f.close()

    train_loss_list.append(train_loss)


time_end = time.time()

# plot loss curve
epoch_list = [(i + 1) for i in range(num_epoch)]
draw_loss_curve(epoch_list, train_loss=train_loss_list)
print("training time:", time_end - time_start, 's')
f.close()
