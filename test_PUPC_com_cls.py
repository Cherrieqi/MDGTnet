import torch
import gc
import numpy as np
from torch.utils.data import DataLoader

from ImageDataset_MDGTnet_H1318_com_cls import ImgDataset_test_bce
from networks.MDGTnet import MDGTnet
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import random
from utils.label_vision import label_vision_1d

seed = 6

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

# set model paras
in_ch = 102
out_ch = [500, 750, 500, 500, 500, 500, 300, 150]
spec_range = [65, 102]
padding = 0
class_num = 4
slice_size = 3
batch_size = 1024
device = "cuda:0"
# device = "cpu"

model_path = r"./models/MDGTnet_PUPC/model9.pth"

# load test data H13 or H18
img = np.load("./data/MDGTnet_PUPC/gen_H13/img_norm_all.npy")
label = np.load("./data/MDGTnet_PUPC/gen_H13/gt_norm_all.npy")

# img = np.load("data/MDGTnet_PUPC/gen_H18/img_norm_all.npy")
# label = np.load("data/MDGTnet_PUPC/gen_H18/gt_norm_all.npy")

img = torch.from_numpy(img).float()
label = torch.LongTensor(label)

test_set = ImgDataset_test_bce(img, label)
del img, label
gc.collect()

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# define and load model
model = MDGTnet(in_ch=in_ch, out_ch=out_ch, padding=padding, slice_size=slice_size, spec_range=spec_range,
                class_num=class_num).to(device)
model.load_state_dict(torch.load(model_path))

# test
correct_num = 0
gt_total = []
pred_total = []
row_col_total = []
with torch.no_grad():
    loop = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, data in loop:
        y_out_te, __, __, __, __ = model(data[1].to(device), data[0].to(device))
        gt_te = data[2][:, :4].argmax(dim=1).flatten().cpu().numpy()
        row_col = data[2][:, 4:]
        pred_prob = torch.sigmoid(y_out_te)
        pred = pred_prob.argmax(dim=1).flatten().cpu().numpy()

        gt_total.extend(gt_te)
        pred_total.extend(pred)
        row_col_total.extend(row_col)
        oa_batch = np.sum(gt_te - pred == 0) / data[0].shape[0]

        loop.set_description(f'[{i}/{len(test_loader)}]')
        loop.set_postfix(oa_batch=oa_batch)


# evaluation
print(confusion_matrix(gt_total, pred_total))
print(accuracy_score(gt_total, pred_total))
print(cohen_kappa_score(gt_total, pred_total))

# plot classification results
label_vision_1d(pred_total, row_col_total, 349, 1905, "./logs/pred_PUPC_H13.png")
label_vision_1d(gt_total, row_col_total, 349, 1905, "./logs/gt_PUPC_H13.png")
# label_vision_1d(pred_total, row_col_total, 2384, 601, "./logs/pred_PUPC_H18.png")
# label_vision_1d(gt_total, row_col_total, 2384, 601, "./logs/gt_PUPC_H18.png")


