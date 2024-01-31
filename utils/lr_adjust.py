import torch


def lr_adj(iter_num, lr_init, model):
    lr = lr_init

    if iter_num % 40 == 0:
        lr = lr*0.2

    else:
        lr = lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)
    return optimizer, lr

