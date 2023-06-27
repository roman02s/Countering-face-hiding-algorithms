import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def FGSM_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def PGD_attack(image, epsilon, data_grad, alpha):
    # Project the perturbed image to the L-inf ball
    perturbed_image = torch.clamp(image + alpha*data_grad.sign(), min=0, max=1)
    # Calculate the perturbation
    perturbation = perturbed_image - image
    # Project the perturbation to the L-inf ball
    perturbation = torch.clamp(perturbation, min=-epsilon, max=epsilon)
    # Add the perturbation to the original image to obtain the adversarial example
    perturbed_image = image + perturbation
    # Return the perturbed image
    return perturbed_image


def Carlini_and_wagner_attack(image, epsilon, data_grad, alpha):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + alpha*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def fast_flipping_attack(image, epsilon, data_grad, alpha):
    # fast flipping attack



    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + alpha*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


train_loader = torch.utils.data.DataLoader(
    datasets.LFWPeople('data/train', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])),
    batch_size=1, shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AlexNet().to(device)

model.eval()

epsilons = [0, .05, .1, .15, .2, .25, .3]
epsilon = epsilons[0]


def show_examples(examples):
    cnt = 0
    fig = plt.figure()
    for (init_pred, final_pred, ex) in examples:
        cnt += 1
        plt.subplot(1, len(examples), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if final_pred == 0:
            lab = 'Adversarial'
        else:
            lab = 'Original'
        plt.xlabel("{} -> {}".format(init_pred, final_pred, lab))
        plt.imshow(ex, cmap="gray")
    plt.show()

examples = []
for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    data.requires_grad = True
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if init_pred.item() != target.item():
        continue
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = FGSM_attack(data, epsilon, data_grad)
    output = model(perturbed_data)
    final_pred = output.max(1, keepdim=True)[1]
    if final_pred.item() == target.item():
        continue
    else:
        if len(examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            examples.append((init_pred.item(), final_pred.item(), adv_ex))
show_examples(examples)


def test(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        loss = nn.CrossEntropyLoss()(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        # perturbed_data = FGSM_attack(data, epsilon, data_grad)
        # perturbed_data = PGD_attack(model, data, epsilon, data_grad)
        perturbed_data = Carlini_and_wagner_attack(model, data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc, adv_examples

accuracies = []
examples = []


def main():
    pass
    # for epsilon in epsilons:
    #     acc, ex = test(model, device, train_loader, epsilon)
    #     accuracies.append(acc)
    #     examples.append(ex)
    #     show_examples(ex)
    # plt.figure(figsize=(5, 5))
    # plt.plot(epsilons, accuracies, "*-")
    # plt.yticks(np.arange(0, 1.1, step=0.1))
    # plt.xticks(np.arange(0, .35, step=0.05))
    # plt.title("Accuracy vs Epsilon")
    # plt.xlabel("Epsilon")
    # plt.ylabel("Accuracy")
    # plt.savefig('acc_vs_eps.png')
    # plt.show()


if __name__ == '__main__':
    main()
