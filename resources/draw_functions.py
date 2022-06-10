import matplotlib.pyplot as plt


def draw_loss_curve(train_loss_list, test_loss_list):
    """

    Args:
        train_loss_list: a list containing train loss of each epoch, like [1, 2, 5, 8, 4]
        test_loss_list: a list containing test loss of each epoch, like [2, 6, 7, 5, 3]

    Returns: the loss curves of train and test process
    """

    epoch_list = range(0, len(train_loss_list))

    plt.plot(epoch_list, train_loss_list, color='red', label='Train Loss')
    plt.plot(epoch_list, test_loss_list, color='blue', label='Test Loss')
    plt.xticks(epoch_list, ['{}'.format(i+1) for i in epoch_list])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.4)

    plt.show()
