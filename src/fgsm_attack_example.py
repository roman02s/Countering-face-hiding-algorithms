# FGSM attack example

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import numpy as np
import matplotlib.pyplot as plt

# from advertorch_examples.utils import get_mnist_train_loader
# from advertorch_examples.utils import TRAINED_MODEL_PATH


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


def main():
    # MNIST Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/MNIST', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=1, shuffle=True)

    # MNIST data image of shape 28 * 28 = 784
    num_features = 784
    # 0-9 digits recognition = 10 classes
    num_classes = 10

    # Fully connected neural network with one hidden layer
    class NeuralNet(nn.Module):
        def __init__(self, num_features, num_classes):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(num_features, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            out = F.relu(self.fc1(x))
            out = self.fc2(out)
            return out

    model = NeuralNet(num_features, num_classes)

    # Load pre-trained model
    # model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    model.eval()

    # FGSM attack code
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    correct = 0
    adv_examples = []

    # Run test for each epsilon
    for eps in epsilons:
        # Loop over all examples in test set
        for data, target in test_loader:
            # Send the data and label to the device
            data, target = data.to(device), target.to(device)

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = model(data.reshape(-1, 784))
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != target.item():
                continue

            # Calculate the loss
            loss = F.nll_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, eps, data_grad)

            # Re-classify the perturbed image
            output = model(perturbed_data.resize(1, 784))

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if final_pred.item() == target.item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (eps == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        # show some adv examples
        plt.figure(figsize=(8, 10))
        for i, (init_pred, final_pred, adv_ex) in enumerate(adv_examples):
            plt.subplot(1, len(adv_examples), i + 1)
            plt.xticks([], [])
            plt.yticks([], [])
            if i == 0:
                plt.ylabel("Eps: {}".format(eps), fontsize=14)
            orig, adv, final = init_pred, adv_ex, final_pred
            plt.title("{} -> {}".format(orig, final))
            plt.imshow(adv, cmap="gray")
        plt.show()
        # Calculate final accuracy for this epsilon
        final_acc = correct/float(len(test_loader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(eps, correct, len(test_loader), final_acc))

        # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


if __name__ == '__main__':
    main()
