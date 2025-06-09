import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os


class MnistDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.trans = transform

        self.sample = []
        self.label = []

        label = os.listdir(self.root)
        for item in label:
            i = item + '/'
            sample = os.listdir(os.path.join(self.root, item))
            for s in sample:
                self.sample.append(os.path.join(self.root, i, s))
                self.label.append(int(item))

    def __getitem__(self, index):
        img_name = self.sample[index]
        img = Image.open(img_name)
        img = self.trans(img)

        label = self.label[index]

        return img, label

    def __len__(self):
        return len(self.sample)


def load_data(batch_size):
    train_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 转化三通道
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    train_iter = torch.utils.data.DataLoader(MnistDataset(root='./data/mnist_train/', transform=train_trans),
                                             batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(MnistDataset(root='./data/mnist_test/', transform=test_trans),
                                            batch_size=batch_size, shuffle=True)

    return train_iter, test_iter


if __name__ == '__main__':
    train_iter, test_iter = load_data(64)
    print(enumerate(train_iter))
    for x, y in train_iter:
        print(x.shape, type(y[0]))
        break
