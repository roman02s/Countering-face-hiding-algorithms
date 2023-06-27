from ResNet import ResNet, ResidualBlock
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# lfw_people dataset with resnet

train_loader = torch.utils.data.DataLoader(
    datasets.LFWPeople('data/train', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]), download=True),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.LFWPeople('data/train', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]), download=True),
    batch_size=64, shuffle=True)


# model
model = ResNet(ResidualBlock, [2, 2, 2, 2]).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train
losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(10):
    train_loss = 0
    train_acc = 0
    model.train()
    for im, label in train_loader:
        im = im.cuda()
        label = label.cuda()
        # forward
        out = model.forward(im)
        loss = criterion(out, label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss and accuracy
        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    # eval
    eval_loss = 0
    eval_acc = 0
    model.eval()
    for im, label in test_loader:
        im = im.cuda()
        label = label.cuda()
        out = model(im)
        out = model(im)
        loss = criterion(out, label)
        # loss and accuracy
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Eval Loss: {:.4f}, Eval Acc: {:.4f}'.format(
        e, train_loss / len(train_loader), train_acc / len(train_loader), eval_loss / len(test_loader),
        eval_acc / len(test_loader)))

# for e in range(10):
#     train_loss = 0
#     train_acc = 0
#     model.train()
#     for im, label in train_loader:
#         im = im.cuda()
#         label = label.cuda()
#         # forward
#         out = model(im)
#         loss = criterion(out, label)
#         # backward
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # loss and accuracy
#         train_loss += loss.item()
#         _, pred = out.max(1)
#         num_correct = (pred == label).sum().item()
#         acc = num_correct / im.shape[0]
#         train_acc += acc
#     losses.append(train_loss / len(train_loader))
#     acces.append(train_acc / len(train_loader))
#     # eval
#     eval_loss = 0
#     eval_acc = 0
#     model.eval()
#     for im, label in test_loader:
#         im = im.cuda()
#         label = label.cuda()
#         out = model(im)
#         loss = criterion(out, label)
#         # loss and accuracy
#         eval_loss += loss.item()
#         _, pred = out.max(1)
#         num_correct = (pred == label).sum().item()
#         acc = num_correct / im.shape[0]
#         eval_acc += acc
#     eval_losses.append(eval_loss / len(test_loader))
#     eval_acces.append(eval_acc / len(test_loader))
#     print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Eval Loss: {:.4f}, Eval Acc: {:.4f}'.format(
#         e, train_loss / len(train_loader), train_acc / len(train_loader),
#         eval_loss / len(test_loader), eval_acc / len(test_loader)))

# plot
plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)
plt.show()

plt.title('train accuracy')
plt.plot(np.arange(len(acces)), acces)
plt.show()

plt.title('eval loss')
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.show()

plt.title('eval accuracy')
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.show()

# save model
torch.save(model.state_dict(), 'model_resnet.pth')

# load model
model.load_state_dict(torch.load('model_resnet.pth'))

# test
model.eval()
for im, label in test_loader:
    im = im.cuda()
    label = label.cuda()
    out = model(im)
    _, pred = out.max(1)
    num_correct = (pred == label).sum().item()
    acc = num_correct / im.shape[0]
    print('Test Acc: {:.4f}'.format(acc))

# Test Acc: 0.9844

