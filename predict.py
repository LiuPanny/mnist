import random
import cv2
from utils.model import ResNet18, BasicBlock
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch


trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

path = 'data/mnist_test/3/'
pre = os.listdir(path)
pre = random.sample(pre, 10)
images = []
img_tensor = []
for item in pre:
    img = cv2.imread(os.path.join(path, item))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
    img = Image.fromarray(img)
    img = trans(img)
    img_tensor.append(img)
image = torch.stack((img_tensor), 0)

net = ResNet18(BasicBlock, 10)
net.load_state_dict(torch.load('utils/ResNet18.pth'))
pred = torch.argmax(net(image), 1)
# print(net(image))
print(pred)

for i in range(10):
    plt.subplot(2, 5, i+1)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.imshow(images[i])
plt.show()

