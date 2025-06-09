from utils.dataLoader import load_data
from utils.model import ResNet18, BasicBlock
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
import torch
from time import time


def accuracy(y_hat, y):
    pre = torch.argmax(y_hat, 1)
    acc = (pre == y).sum().float()
    return acc, len(y)


def train(net, epoches, train_iter, test_iter, device, loss, optim, model_path):
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    net.to(device)

    for eopch in range(epoches):
        net.train()
        train_loss = 0
        train_acc = 0
        train_len = 0
        with tqdm(range(len(train_iter)), ncols=100, colour='red',
                  desc='train epoch {}/{}'.format(eopch + 1, epoches)) as pbar:
            for X, y in train_iter:
                optim.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optim.step()
                train_loss += l.detach()
                train_acc += accuracy(y_hat, y)[0]
                train_len += accuracy(y_hat, y)[1]
                pbar.set_postfix(
                    {'loss': '{:.6f}'.format(train_loss / train_len), 'acc': '{:.6f}'.format(train_acc / train_len)})
                pbar.update(1)
            train_acc_list.append(train_acc / train_len)
            train_loss_list.append(train_loss / train_len)

        net.eval()
        test_acc = 0
        test_loss = 0
        test_len = 0
        with tqdm(range(len(test_iter)), ncols=100, colour='blue',
                  desc='test epoch {}/{}'.format(eopch + 1, epoches)) as pbar:
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                test_acc += accuracy(y_hat, y)[0]
                test_len += accuracy(y_hat, y)[1]
                with torch.no_grad():
                    l = loss(y_hat, y)
                    test_loss += l.detach()
                    pbar.set_postfix(
                        {'loss': '{:.6f}'.format(test_loss / test_len), 'acc': '{:.6f}'.format(test_acc / test_len)})
                    pbar.update(1)
            test_acc_list.append(test_acc / test_len)
            test_loss_list.append(test_loss / test_len)

        # torch.save(net.state_dict(), model_path)

    plt.plot([i + 1 for i in range(len(train_acc_list))], train_acc_list, 'ro--', label='train_acc')
    plt.plot([i + 1 for i in range(len(test_acc_list))], test_acc_list, 'bo--', label='test_acc')
    plt.ylim([0.7, 1])
    plt.title('train_acc vs test_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('logs/acc.png')

    plt.plot([i + 1 for i in range(len(train_loss_list))], train_loss_list, 'ro--', label='train_loss')
    plt.plot([i + 1 for i in range(len(test_loss_list))], test_loss_list, 'bo--', label='test_loss')
    plt.ylim([0, 0.1])
    plt.title('train_loss vs test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('logs/loss.png')


if __name__ == '__main__':
    epoch = 10
    lr = 0.05
    batch_size = 128
    weight_decay = 1e-4

    net = ResNet18(BasicBlock, 10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), lr, weight_decay=weight_decay)
    model_path = 'utils/ResNet18.pth'

    train_iter, test_iter = load_data(batch_size=batch_size)

    print('训练开始：')
    start_time = time()
    train(net, epoch, train_iter, test_iter, device, loss, opt, model_path)
    torch.save(net.state_dict(), model_path)
    end_time = time()
    time = end_time - start_time
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    print('训练结束')
    print('本次训练时长为：%02d:%02d:%02d' % (h, m, s))
