# patch example of an attack in neural networks

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
model.load_state_dict(torch.load('mnist_cnn.pt'))
model.eval()

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=False, download=True,
      transform=torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ])),
    batch_size=1, shuffle=True)


def patch(img, x, y, patch_size=28):
    img = img.clone()
    img[:, :, x:x+patch_size, y:y+patch_size] = 0
    return img


def predict(img):
    output = model(img)
    pred = output.data.max(1, keepdim=True)[1]
    return pred.item()


def test_patch(img, x, y, patch_size=28):
    img = patch(img, x, y, patch_size)
    return predict(img)

img, label = next(iter(test_loader))
img = img.cuda()
label = label.cuda()
img = img[0].unsqueeze(0)
label = label[0].unsqueeze(0)
print('Original label:', label.item())
print('Predicted label:', predict(img))


