from utils.dataLoader import load_data
from utils.model import ResNet18, BasicBlock
import torch


net = ResNet18(BasicBlock, 10)
net.load_state_dict(torch.load('utils/ResNet18.pth'))
train_iter, test_iter = load_data(10)

print('train:')
for i, (X, y) in enumerate(train_iter):
    pre = torch.argmax(net(X), 1)
    print(y, pre)
    break

print('test:')
for i, (X, y) in enumerate(test_iter):
    pre_test = torch.argmax(net(X), 1)
    print(y, pre_test)
    break

