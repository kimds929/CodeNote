import os

import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
import torchvision.transforms as transforms


# * Dataset & DataLoader
# * Network
# * Optimizer / Loss_Function
# * Training Loop
# * Back_Propagation

batch_size = 100

transform = transforms.Compose([
    transforms.ToTensor(),                                  # Numpy → Pytorch
    transforms.Normalize(mean=(0.5), std=(0.5))  # Normalize
# (x - mean) / std => x ~ (0, 1) → (x - 0.5) / 0.5 → -1 ~ 1
# mean, std → dataset의 mean과 std
])

trainset = torchvision.datasets.MNIST(root='./Dataset', train=True, 
                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./Dataset', train=False,
                            download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                            shuffle=False, num_workers=0)

def imshow(img):    # img ~ (-1, 1)
    img = img/2 + 0.5   # img ~ (0, 1)
    npimg = img.numpy() # Tensor (Batch, Channel, Height, Width) → numpy
    print(np.transpose(npimg, (1, 2, 0)).shape) # (Batch, Height, Width, Channel)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Define Model
class NetforMNIST_Linear(nn.Module):
    def __init__(self):
        super(NetforMNIST_Linear, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 10)
    
    def forward(self, x):
        # Input: (Batchsize, 1, 28, 28) → Output: 숫자 0~9
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x
    

model = NetforMNIST_Linear()
img_sample = torch.randn(4, 1, 28, 28)
output = model(img_sample)
output  # (batch_size, 10)

# Image
pred_result = output.argmax(axis=-1)
pred_image = np.transpose(img_sample.numpy(), (0, 2, 3, 1))
for i, ai in enumerate(pred_image, 1):
    plt.subplot(2,2,i)
    plt.title(str(pred_result[i-1].numpy()))
    plt.imshow(ai.squeeze(), 'gray')
plt.show()



# Define CNN Model 
class NetforMNIST_CNN(nn.Module):
    def __init__(self):
        super(NetforMNIST_CNN, self).__init__()
        # Image Size를 보존하는 magic number
        # kernel_size, stride, padding
        # 3, 1, 1
        # 5, 1, 2
        # 7, 1, 3
        # 9, 1, 4
        # kernel_size, stride, kernel_size // 2

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.maxpool = nn.MaxPool2d((2,2))

        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = self.maxpool(F.relu(self.conv2(x)))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # output이 binary classification - sigmoid
        # output이 image - tanh
        # output이 classification - softmax (loss function과 결합되어 있음)
    

model = NetforMNIST_Linear()
model = NetforMNIST_CNN()
# model = model.cuda()  # GPU로 보내주기

img_sample = torch.randn(4, 1, 28, 28)
output = model(img_sample)
print(output.shape)

# torch.nn.Conv2d(
#     in_channels: int,
#     out_channels: int,
#     kernel_size: Union[int, Tuple[int, int]],
#     stride: Union[int, Tuple[int, int]] = 1,
#     padding: Union[int, Tuple[int, int]] = 0,
#     dilation: Union[int, Tuple[int, int]] = 1,
#     groups: int = 1,
#     bias: bool = True,
#     padding_mode: str = 'zeros',
# )


model = NetforMNIST_CNN()
n_epochs = 5
learning_rate = 0.001

loss_function = nn.CrossEntropyLoss()     # Loss_Function 안에 softmax가 포함되어있음
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # Classification 과 같은 vision task에서는 SGD with momentum > Adam
# Image Generation, reconstruction - Adam
# optimizer 정의할때 network의 learnable paramter들을 넣어주어야함


# Training
for epoch in range(n_epochs):
    running_loss = 0
    for i, (batch_x, batch_y) in enumerate(trainloader, 1):
        # GPU 로 보내기
        # batch_x = batch_x.cuda()
        # batch_y = batch_y.cuda()

        optimizer.zero_grad()       # (==) model.zero_grad()
        
        pred = model(batch_x)   # forward
        loss = loss_function(input=pred, target=batch_y)    # loss
        loss.backward()             # backward
        optimizer.step()            # update weight

        running_loss += loss.item()    # Tensor안의 값을 Return
    print(f'epochs: {epoch} / loss: {running_loss / batch_size}')


# Test_Code
correct = 0
total = 0

for test_x, test_y in testloader:
    # GPU
    # test_x = test_x.cuda()
    # test_y = test_y.cuda()

    outputs = model(test_x)
    _, output_argmax = torch.max(outputs, axis=-1) # 값, index
    # output_argmax = torch.argmax(outputs, axis=-1)

    correct += (output_argmax == test_y).sum().item()
    total += test_y.shape[0]
test_accuracy = correct / total
print(test_accuracy)



# 결과 plotting -------------------------------------
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    print(np.transpose(npimg, (1, 2, 0)).shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

outputs = model(images)
_, predicted = torch.max(outputs.data, 1)
print(predicted.numpy())
