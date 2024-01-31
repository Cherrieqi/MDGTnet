from matplotlib import pyplot as plt


def draw_loss_curve(epoch, train_loss):
    plt.title("Loss change")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(epoch, train_loss, label='train_loss')
    plt.legend()
    plt.savefig('./logs/loss.png')
