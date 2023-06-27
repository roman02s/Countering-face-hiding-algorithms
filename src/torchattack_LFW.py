import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchattacks

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


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



# data
transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset from LFW
trainset = torchvision.datasets.LFWPeople(root='./data', split='train',
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

# testset from LFW
testset = torchvision.datasets.LFWPeople(root='./data', split='test',
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

classes = ('0', '1')

# train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# pretrained model

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# test
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 2000 test images: %d %%' % (100 * correct / total))

# test with attack
attack = Attack(model, device, 'FGSM')
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        images = attack(images, labels)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 2000 test images: %d %%' % (100 * correct / total))

# test with attack
attack = Attack(model, device, 'PGD')
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        images = attack(images, labels)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 2000 test images: %d %%' % (100 * correct / total))

# test with attack example2
attack = Attack(model, device, 'FGSM')

correct = 0
total = 0

x_adv = []
y_adv = []

for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    images_adv = attack(images, labels)

x_adv.append(images_adv.cpu().numpy())
y_adv.append(labels.cpu().numpy())

outputs = model(images_adv)
_, predicted = torch.max(outputs.data, 1)
total += labels.size(0)
correct += (predicted == labels).sum().item()
print('Accuracy of the network on the adversarial test images: %d %%' % (100 * correct / total))

x_adv = np.concatenate(x_adv, axis=0)
y_adv = np.concatenate(y_adv, axis=0)

np.save('x_adv.npy', x_adv)
np.save('y_adv.npy', y_adv)
