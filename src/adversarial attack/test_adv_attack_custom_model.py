#%%
# !pip install adversarial-robustness-toolbox
#%%
import time
import numpy as np
import torch.nn as nn
from datetime import datetime

from PIL import Image
from torchvision import transforms
import torchvision
import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt

from art.estimators.classification import PyTorchClassifier

import warnings
import torch

from sklearn.datasets import fetch_lfw_people
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
warnings.filterwarnings('ignore')
#%%
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
#%%
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
#%%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
n_components = 150

print(
    "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
)
t0 = time.time()
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
print("done in %0.3fs" % (time.time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time.time()
X_train_pca = torch.Tensor(pca.transform(X_train))
X_test_pca = torch.Tensor(pca.transform(X_test))
print("done in %0.3fs" % (time.time() - t0))
#%%
#%%
# create dataset and dataloader
batch_size = 64

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.from_dlpack(X_train_pca).float(),
        torch.from_dlpack(y_train).long()
    ),
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.from_dlpack(X_test_pca).float(),
        torch.from_dlpack(y_test).long()
    ),
    batch_size=batch_size,
    shuffle=False
)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Linear(n_components, 2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, n_classes),
    nn.LogSoftmax(dim=1)
)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.NLLLoss()
#%%

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        print(batch_idx, (data.shape, target.shape))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#%%
epochs = 100
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
#%%
# Adversarial part
# !pip install torchattacks==3.1.0
#%%
import torchattacks
#%%
class Attack:
    def __init__(self, model, device, attack_type):
        self.model = model
        self.device = device
        self.attack_type = attack_type
        self.attack = None

    def __call__(self, x, y):
        if self.attack_type == 'FGSM':
            self.attack = torchattacks.FGSM(self.model, eps=8/255)
        elif self.attack_type == 'PGD':
            self.attack = torchattacks.PGD(self.model, eps=8/255, alpha=2/255, steps=7)
        elif self.attack_type == 'CW':
            self.attack = torchattacks.CW(self.model, c=1, kappa=0, steps=1000, lr=0.01)
        elif self.attack_type == 'DeepFool':
            self.attack = torchattacks.DeepFool(self.model)
        elif self.attack_type == 'BIM':
            self.attack = torchattacks.BIM(self.model, eps=8/255, alpha=2/255, steps=7)
        elif self.attack_type == 'RFGSM':
            self.attack = torchattacks.RFGSM(self.model, eps=8/255, alpha=2/255, steps=7)

        x_adv = self.attack(x, y)
        return x_adv
#%%
# attack
attack = Attack(model, device, 'FGSM')
#%%
# train adversarial examples
x_adv = []
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    x_adv.append(attack(data, target))
x_adv = torch.cat(x_adv, dim=0)
#%%
# adversarial loader
train_loader_adv = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.from_dlpack(x_adv).float(),
        torch.from_dlpack(y_train).long()
    ),
    batch_size=batch_size,
    shuffle=True
)

test_loader_adv = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.from_dlpack(x_adv).float(),
        torch.from_dlpack(y_test).long()
    ),
    batch_size=batch_size,
    shuffle=False
)
#%%
# test adversarial examples
test(model, device, test_loader_adv)

#%%
# plot adversarial examples
import matplotlib.pyplot as plt


def show_images_adversarial(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
#%%
# show adversarial examples
show_images_adversarial(x_adv.detach().cpu().numpy(), y_train.detach().cpu().numpy())

#%%
