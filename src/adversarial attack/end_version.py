#%%
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
import matplotlib.pyplot as plt

from art.estimators.classification import PyTorchClassifier

#%%
train_loader = torch.utils.data.DataLoader(
    datasets.LFWPeople('data/lfw_people', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])),
    train=True, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.LFWPeople('data/lfw_people', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])),
    train=False, batch_size=64, shuffle=True)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
model_resnet18 = models.resnet18(pretrained=True)
model = PyTorchClassifier(
    model=model_resnet18,
    loss=criterion,
    input_shape=(3, 224, 224),
    nb_classes=train_loader.dataset.nb_classes,
    device_type='gpu'
)

#%%


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


