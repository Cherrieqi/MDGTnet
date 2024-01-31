# MDGTnet

<img src="C:\Users\cherry\Desktop\FIG 1.png" >

[Multi-Source Domain Generalization Two-Branch Network for Hyperspectral Image Cross-Domain Classification](https://ieeexplore.ieee.org/document/10410893)

[Yunxiao Qi](https://ieeexplore.ieee.org/author/37090046528); [Junping Zhang](https://ieeexplore.ieee.org/author/37293675400); [Dongyang Liu](https://ieeexplore.ieee.org/author/37089208447); [Ye Zhang](https://ieeexplore.ieee.org/author/37279965600)

## Requirements

This code is based on **Python 3.10** and **Pytorch 1.12**.

*Installation list:*

**· pytorch** 

**· matplotlib**

**· opencv-python**

**· scipy**

**· h5py**

**· tqdm**

**· scikit-learn**

## Models

**· SD--H13+H18 :** 

[model9.pth]: https://pan.baidu.com/s/1DOecuJQPklCug4V0RgDh9Q?pwd=1111

**· SD--PU+PC :** 

[model9.pth]: https://pan.baidu.com/s/1Q9HCafos_9zU7pV13siw5A?pwd=1111

## Datasets

[raw]: https://pan.baidu.com/s/1IvzKEQOw7xwVPwUrpOrhxQ?pwd=1111

**:** Houston2013 / Houston2018 / PaviaU / PaviaC

**

[H13+H18 -- PU/PC]: https://pan.baidu.com/s/1DFd5cb3xalw5lcSigS57BQ?pwd=1111

 : **gen_H13 / gen_H18 / gen_PU / gen_PC

**

[PU+PC]: https://pan.baidu.com/s/1Q4pFYugFKw1YmAmZ8sDsrg?pwd=1111

 :** gen_H13 / gen_H18 / gen_PU / gen_PC

## Getting start:

##### · Dataset structure

data/MDGTnet_H1318
├── gen_H13
│   ├── img_norm_all.npy
│   └── gt_norm_all.npy
├── gen_H18
│   ├── img_norm_all.npy
│   └── gt_norm_all.npy

├── gen_PC
│   ├── img_norm_all.npy
│   └── gt_norm_all.npy

└── gen_PU
     ├── img_norm_all.npy
     └── gt_norm_all.npy



data/MDGTnet_PUPC
├── gen_H13
│   ├── img_norm_all.npy
│   └── gt_norm_all.npy
├── gen_H18
│   ├── img_norm_all.npy
│   └── gt_norm_all.npy

├── gen_PC
│   ├── img_norm_all.npy
│   └── gt_norm_all.npy

└── gen_PU
     ├── img_norm_all.npy
     └── gt_norm_all.npy



data/raw
├── Houston2013
│   ├── Houston.mat
│   └── Houston_gt.mat
├── Houston2018
│   ├── HoustonU.mat
│   └── HoustonU_gt.mat

├── PaviaC
│   ├── pavia.mat
│   └── pavia_gt.mat

└── PaviaU
     ├── paviaU.mat
     └── paviaU_gt.mat

**NOTE:**

​       Training and test data can be generated via *data_pre_MDGTnet_xxxxx.py* respectively. Where *_H1318* indicates that the source domains are H13 and H18 and *_PUPC* indicates that the source domains are PU and PC.

##### · Train

​       Run *train_xxxx_com_cls.py*. 

##### · Test

​       Run *test_xxxx_com_cls.py*. 























