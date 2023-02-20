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



# Dataset  ============================================================================
transform = torchvision.transforms.Compose(
    [
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # torchvision.transforms.RandomAffine()
    ]
    )   # 이미지의 범위를 바꿔준다 [0, 1] → [-1, 1]

batch_size = 64
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                     shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
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



# ## Define a Convolutional Neural Network  ============================================================================
# 
# Network Model
#     1. Convolution - input channel: 3, output channel: 8, kernel_size: 3
#     2. Maxpoling   - size: 2, stride: 2
#     3. Convolution - input channel: 8, output channel: 16, kernel_size: 3
#     4. Maxpoling   - size: 2, stride: 2
#     5. Fully connected layer - in_features: 400, out_features: 120
#     6. Fully connected layer - in_features: 120, out_features: 84
#     7. Fully connected layer - in_features: 84, out_features: 10
# 
# * Apply ReLU activation function for hidden layers.
# * Do not apply softmax activation function for the output layer.
#     * softmax activation function are included in the loss function.

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)
# net( next(iter(trainloader))[0][:1].to(device) )

# Test case
# - check the default output of model before training
# - make sure that the definition of layers in your model is correct.
print(net(images.to(device)).shape)
print(str(net))





# ## Training ============================================================================
# ### Define a Loss function and optimizer
# * Use Classification Cross-Entropy loss
# * Use SGD with learning rate 0.005 and momentum 0.9
loss_obj = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)



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

# # ## Train the network on the training data
# print('Start Training ')
# for epoch in range(5):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs
#         inputs, labels = data[0].to(device), data[1].to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = loss_obj(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 250 == 249:
#             print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 500))
#             running_loss = 0.0

# print('Finished Training')




# ## Testing ============================================================================
# ### Show network prediction


# display ground truth
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GrondTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))



# display predicted
outputs = net(images.to(device))

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# - check the predicted output of model after training
# - make sure that the optimizer and its options are correctly defined.


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

# # Let us look at how the network performs on the whole dataset
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))


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



# class_correct = [0.0] * 10
# class_total = [0.0] * 10
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(labels.shape[0]):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1


# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))


















# VGG Net =============================================================================
# ## Define VGG-11 network
# * Network Model
#     * Convolution - in_channels=3, out_channels=64, kernel_size=3, padding=1
#     * Maxpoling2d - kernel_size=3, stride=2
#     * Convolution - in_channels=64, out_channels=128, kernel_size=3, padding=1
#     * Maxpoling2d - kernel_size=3, stride=2
#     * Convolution - in_channels=128, out_channels=256, kernel_size=3, padding=1
#     * Maxpoling2d - kernel_size=3, stride=2
#     * Convolution - in_channels=256, out_channels=512, kernel_size=3, padding=1
#     * Maxpoling2d - kernel_size=3, stride=2
#     * Fully connected layer - in_features: 512, out_features: 10
# 
# 
# * You can use `VGG.make_conv_relu` function.
# * Apply ReLU activation function for hidden layers.
# * Do not apply softmax activation function for the output layer.
#     * softmax activation function are included in the loss function.
# 


# kerenl 3×3 2개 → 5×5 1개 효과
# kerenl 3×3 3개 → 7×7 1개 효과
class VGG(nn.Module):
    
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1_1 = self.make_conv_relu(3, 64)
        self.conv2_1 = self.make_conv_relu(64, 128)
        self.conv3_1 = self.make_conv_relu(128, 256)
        self.conv3_2 = self.make_conv_relu(256, 512)
        self.conv4_1 = self.make_conv_relu(512, 512)
        self.conv4_2 = self.make_conv_relu(512, 512)
        self.conv5_1 = self.make_conv_relu(512, 512)
        self.conv5_2 = self.make_conv_relu(512, 512)

        self.pool = nn.MaxPool2d(2, 2)
        # self.relu = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(512, 10)

    def make_conv_relu(self, in_channels, out_channel):
        layers = []
        layers += [nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True)]   # inplace=True: 메모리를 적게쓰는 Option
        return nn.Sequential(*layers)
        # Sequential(
        # (0): Conv2d(in_channels, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (1): ReLU(inplace=True)
        # )

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.pool(x)

        x = self.conv2_1(x)
        x = self.pool(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.pool(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.pool(x)

        x = x.view(x.shape[0], -1)

        x = self.classifier(x)
        return x


net = VGG().to(device)

print(net(images.to(device)).shape)
print(str(net))

# net( next(iter(trainloader))[0][:10].to(device) )


# ## Training 
loss_obj = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

training(model=net, loss_obj=loss_obj, optimizer=optimizer, trainloader=trainloader)

# ## Testing 
evaluate(model=net, testloader=testloader)

evaluate_class(model=net, testloader=testloader)


































# ResNet =============================================================================

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, num_conv, in_channels, hidden_channels=None, kernel_size=3):
        super().__init__()

        stride = 1
        out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        # make modules and append to list
        list_ = []
        for i in range(num_conv):
            if i == 0:
                list_.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=1))
            elif i == num_conv - 1:
                list_.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=1))
            elif 0 < i < num_conv - 1:
                list_.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=1))
            else:
                pass
            list_.append(nn.ReLU(inplace=True)) # append activation function

        self.module_list = nn.ModuleList(list_)

    def forward(self, x):
        input = x
        for m in self.module_list:
            x = m(x)
        return x + input


# ResNet
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.res1 = ResidualBlock(2, 64, 64)    # Residual Block
        self.res2 = ResidualBlock(2, 64, 64)    # Residual Block
        self.res3 = ResidualBlock(2, 64, 64)    # Residual Block

        self.fc1 = nn.Linear(64*4*4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.pool(self.res1(x))
        x = self.pool(self.res2(x))
        x = self.pool(self.res3(x))

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = ResNet().to(device)

print(net(images.to(device)).shape)
print(str(net))

# net( next(iter(trainloader))[0][:10].to(device) )

# Training
loss_obj = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, momentum=0.9)

training(model=net, loss_obj=loss_obj, optimizer=optimizer, trainloader=trainloader)

# Test
evaluate(model=net, testloader=testloader)
evaluate_class(model=net, testloader=testloader)


# Weight Information --------------------------------------------------------------------
# net.res1.module_list[0].weight.shape      # wegiht: (out_channel, in_channel, kernel_width, kernel_height)
# list(net.res1.module_list[0].parameters())[0].shape       # convolution weight
# list(net.res1.module_list[0].parameters())[1].shape       # bias weight










