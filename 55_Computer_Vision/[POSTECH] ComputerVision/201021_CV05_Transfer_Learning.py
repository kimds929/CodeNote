import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

random_seed = 4332
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

torch.backends.cudnn.deterministic = True   # GPU 사용시 연산결과가 달라질 수 있음.
torch.backends.cudnn.benchmark = False

np.random.seed(random_seed)
random.seed(random_seed)

use_cuda = False
# if use_cuda and torch.cuda.is_available():
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)    

mean = torch.tensor((0.4914, 0.4822, 0.4465))
std = torch.tensor((0.2023, 0.1994, 0.2010))

# Dataset  ============================================================================
transform_train = torchvision.transforms.Compose(
    [
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
    ]
)
transform_test = torchvision.transforms.Compose(
    [
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
    ]
)

batch_size = 64
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                     shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                     shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# next(iter(trainloader))[0].shape
# a = next(iter(trainloader))[0]
# a.mean(axis=[0,2,3])
# a.std(axis=[0,2,3])
# a.max(axis=0)[0].max(axis=1)[0].max(axis=1)[0]
# a.min(axis=0)[0].min(axis=1)[0].min(axis=1)[0]



def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(''.join('%s, ' % classes[labels[j]] for j in range(4)))






# Transfer Learning =======================================================================
from torchvision import models 
# model library안에 저장이 되어있음: vgg11, vgg16, vgg19, resnet34, resnet50, resnet101, resnet152

# 32 × 32 → feature → feature size 작아짐 → 1×1 보다 작아질 수 있음
# 32 × 32 → 224 × 224 Resize
# Global Pooling이 들어가면, fc layer에서 feature size로 인한 error가 발생하지는 않ㅇ므
# 하지만 image size가 너무 작을 경우 network 안에서 feature pooling되면서 1×1보다 작아지면서 error가 발생 할 수 있음
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.resnet.fc = nn.Linear(512, 10)  # 마지막 fc Layer 재정의 (overriding)

        for param in self.resnet.parameters():
            # param.requires_grad_(False)   # in-place, return 없이 class의 variable을 set하는 method
            param.requires_grad = False

        #  for name, param in self.resnet.named_parameters()   # name이 같이 return되어 컨트롤하기 쉬움
        #     if name.startswith('layer'):
        #         param.requires_grad = False


    
    def forward(self, x):
        x = self.resnet(x)

        # x = self.resnet.conv1(x)
        # ...
        # x = self.resnet.layer3(x)
        # x = self.conv1(x)
        # x = self.resnet.layer4(x)
        # ...
        # x = self.resnet.avgpool(x)
        # x = x.view(x.shape[0], -1)
        # x = self.model.fc(x)
        
        return x

net = ResNet18().to(device)
print(net(images.to(device)).shape)
print(str(net))

a = models.resnet18(pretrained=True)
for parameter in a.parameters():
    print(parameter)
for parameter in a.named_parameters():
    print(parameter)

type(parameter)


# Function Definition ============================================================================
# ## Training
def training(model, loss_obj, optimizer, trainloader):
    # ## Train the network on the training data
    print('Start Training ')
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_obj(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 250 == 249:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    print('Finished Training')


# ### Test the network on the test data
# #### Accuracy
# Let us look at how the network performs on the whole dataset
def evaluate(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return correct / total



# #### Accuracy of each class
def evaluate_class(model, testloader):
    class_correct = [0.0] * 10
    class_total = [0.0] * 10
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


# Training - Test ============================================================================

loss_obj = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
# optimizer = torch.optim.SGD(net.fc.parameters(), lr=0.005, momentum=0.9)      # fc parameter만 학습

training(model=net, loss_obj=loss_obj, optimizer=optimizer, trainloader=trainloader)
evaluate(model=net, testloader=testloader)
evaluate_class(model=net, testloader=testloader)







