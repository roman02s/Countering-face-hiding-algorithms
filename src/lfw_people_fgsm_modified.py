#%%
import torch
import torchvision
import torchvision.transforms as transforms
import torchattacks

#%%
# Load the LFW People dataset
transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
#%%
trainset = torchvision.datasets.LFWPeople(root='./data/lfw_people', split='train',
                                          download=True, transform=transform)
testset = torchvision.datasets.LFWPeople(root='./data/lfw_people', split='test',
                                         download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
#%%
# download the model resnet18
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
model.fc = torch.nn.Linear(in_features=512, out_features=1024, bias=True)
#%%
# Define the attack
attack = torchattacks.FGSM(model, eps=0.1)
#%%
# Generate adversarial examples and replace the original images with the perturbed ones
for i, (images, labels) in enumerate(trainloader):
    images, labels = images.cuda(), labels.cuda()
    adv_images = attack(images, labels)
    trainset.data[i*4:(i+1)*4] = (adv_images * 255).permute(0, 2, 3, 1).cpu().numpy().astype('uint8')
#%%
for i, (images, labels) in enumerate(testloader):
    images, labels = images.cuda(), labels.cuda()
    adv_images = attack(images, labels)
    testset.data[i*4:(i+1)*4] = (adv_images * 255).permute(0, 2, 3, 1).cpu().numpy().astype('uint8')
#%%
# Save the modified datasets
torch.save(trainset, './data/trainset_adv.pt')
torch.save(testset, './data/testset_adv.pt')