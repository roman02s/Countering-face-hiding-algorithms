import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# data
transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset from LFW
trainset = torchvision.datasets.LFWPeople(root='./data', split='train',
                                        download=True, transform=transform, min_faces_per_person=10)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

# testset from LFW
testset = torchvision.datasets.LFWPeople(root='./data', split='test',
                                        download=True, transform=transform, min_faces_per_person=10)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

# adversarial patch
adversarial_patch = np.zeros((224, 224, 3))
adversarial_patch[0:224, 0:224, 0] = 1

adversarial_patch = torch.from_numpy(adversarial_patch).float()
adversarial_patch = adversarial_patch.permute(2, 0, 1)
adversarial_patch = adversarial_patch.unsqueeze(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

adversarial_patch = adversarial_patch.to(device)

# model
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)
model = model.to(device)

# train
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i in range(10):
        # get the inputs
        inputs = adversarial_patch
        labels = torch.tensor([1]).to(device)

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
dataiter = iter(testloader)
images, labels = next(dataiter)
