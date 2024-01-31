from torch.utils.data import Dataset
import torch
import random
from utils.normHSI import normHSI_smp_s


class ImgDataset_train(Dataset):
    def __init__(self, *data_and_label):
        # [num, c, slice_size, slice_size]
        self.A_intra_img = torch.cat((data_and_label[0], data_and_label[0][1:],
                                      data_and_label[0][2:], data_and_label[0][3:]), dim=0)
        self.A_img = normHSI_smp_s(self.A_intra_img)
        self.A_gt = torch.cat((data_and_label[1], data_and_label[1][1:],
                               data_and_label[1][2:], data_and_label[1][3:]), dim=0)
        self.B_intra_img = data_and_label[2]
        self.B_img = normHSI_smp_s(self.B_intra_img)
        self.B_gt = data_and_label[3]

    def __len__(self):
        return 150000

    def __getitem__(self, index):
        num_A = 4989
        num_B = 34315

        idx_A = index % (num_A*4-6)

        if index % 4 == 0:

            sample_img_A = self.A_img[idx_A]
            sample_img_A_intra = self.A_intra_img[idx_A]
            sample_gt_A = self.A_gt[idx_A]

            idx_B = random.randint(0, num_B-1)
            while torch.argmax(sample_gt_A) != torch.argmax(self.B_gt[idx_B]):
                idx_B = random.randint(0, num_B-1)

            sample_img_B = self.B_img[idx_B]
            sample_img_B_intra = self.B_intra_img[idx_B]
            sample_gt_B = self.B_gt[idx_B]

            return [sample_img_A, sample_img_A_intra, sample_gt_A,
                    sample_img_B, sample_img_B_intra, sample_gt_B]

        else:
            sample_img_A = self.A_img[idx_A]
            sample_img_A_intra = self.A_intra_img[idx_A]
            sample_gt_A = self.A_gt[idx_A]

            idx_B = random.randint(0, num_B-1)
            while torch.argmax(sample_gt_A) == torch.argmax(self.B_gt[idx_B]):
                idx_B = random.randint(0, num_B-1)

            sample_img_B = self.B_img[idx_B]
            sample_img_B_intra = self.B_intra_img[idx_B]
            sample_gt_B = self.B_gt[idx_B]

            return [sample_img_A, sample_img_A_intra, sample_gt_A,
                    sample_img_B, sample_img_B_intra, sample_gt_B]


class ImgDataset_test_bce(Dataset):
    def __init__(self, image, label):
        # [num, c, slice_size, slice_size]
        self.image = normHSI_smp_s(image)
        self.image_intra = image
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        try:
            img = self.image[index]
            img_intra = self.image_intra[index]
            gt = self.label[index]

        except:
            print(index)

        return img, img_intra, gt





