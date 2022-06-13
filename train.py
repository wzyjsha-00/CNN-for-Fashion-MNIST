import copy
import time
import logging
import warnings
from torch.optim import lr_scheduler

from config.hyper_param import *
from config.path import *
from config.torch_package import *
from resources import utils
from model.LeNet import LeNet5
from model.AlexNet import AlexNetLight
from model.VGGNet import VGGNet16
from model.data_loader import fetch_dataloader
from resources.draw_functions import draw_loss_curve


warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, model_dir):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_list, test_loss_list = [], []

    for epoch in range(num_epochs):

        logging.info('Epoch {}/{} ------:'.format(epoch, num_epochs-1))

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
                scheduler.step()
                dataloader = dataloaders['train']
                dataset_size = dataloaders['train_size']
            else:
                model.eval()
                dataloader = dataloaders['test']
                dataset_size = dataloaders['test_size']

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()    # 将优化器中梯度置于0

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # 返回每一行中最大值的元素及列索引
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()  # 对优化器的参数进行更新
                # statistics
                running_loss += loss.item() * inputs.size(0)  # batch_loss * batch_size
                running_corrects += torch.sum(preds == labels.data)  # 预测正确的个数

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            logging.info('{} Loss: {:.4f} Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'train':
                train_loss_list.append(epoch_loss)
            elif phase == 'test':
                test_loss_list.append(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc

                    best_model_wts = copy.deepcopy(model.state_dict())
                    utils.save_checkpoint(
                        {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optim_dict': optimizer.state_dict()
                        }, is_best=True, checkpoint=model_dir
                    )
                else:
                    utils.save_checkpoint(
                        {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optim_dict': optimizer.state_dict()
                        }, is_best=False, checkpoint=model_dir
                    )

    time_elapsed = time.time() - since

    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed % 60))
    logging.info('Best test Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_loss_list, test_loss_list


if __name__ == '__main__':
    model_dir = os.path.join(os.getcwd(), 'model')

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        cuda = True
        torch.cuda.manual_seed(230)
    else:
        device = torch.device("cpu")
        cuda = False
        torch.manual_seed(230)

    # set the logger
    utils.set_logger(train_log_path)

    # Fetch Dataloaders
    logging.info("Loading the Datasets --:")
    dataloaders = fetch_dataloader(['train', 'test'])
    logging.info("-- Done.")

    net = LeNet5().to(device)
    # net = AlexNetLight().to(device)
    net = VGGNet16().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    # Train and Evaluate
    net, train_loss_list, test_loss_list = train_model(
        net, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
        num_epochs=num_epochs,  model_dir=model_folder_path
    )

    draw_loss_curve(train_loss_list, test_loss_list)
