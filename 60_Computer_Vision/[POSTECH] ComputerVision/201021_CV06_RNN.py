import os
import time
import random
import numpy as np
import pandas as pd
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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

use_cuda = True
if use_cuda and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")




# ## Dataset ================================================================
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor())

# Data loader
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


# ### Visualize Dataset
# display images
def imshow(img, std=None, mean=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(''.join('%s, ' % classes[labels[j]] for j in range(4)))



# ## Define RNN classifier
# ====== LSTM ==================================================================================
# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # create LSTM (batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # (input features=28)
        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # print(x.shape) # (batch_size, C=1, H=28, W=28)

        # change tensor shape
        x = x.view(x.shape[0], self.sequence_length, self.input_size) # (batch_size, sequence_length=28, features=28)
        # print(x.shape)

        # Set initial hidden and cell states (set correct device to tensor)
        # (num_layers, batch_size, hidden_size)
        h0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_size), device=x.device)
        c0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_size), device=x.device)

        # Forward propagate LSTM
        out, (self.h_n, self.c_n) = self.lstm(x, (h0, c0))
        # print(out.shape) # (batch_size, sequence_length=28, features=128)

        # Decode the hidden state of the last time step fc layer
        out = self.fc(out[:, -1, :])
        # print(out.shape) # (batch_size, 10)
        return out


# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10

net = RNN(input_size, sequence_length, hidden_size, num_layers, num_classes).to(device)

# Test case
# - check the default output of model before training
# - make sure that the definition of layers in your model is correct.
print(net(images.to(device)).shape)
print(str(net))

# images.shape
# net.h_n.shape   # (LSTM_Layer, batch, units)
# net.c_n.shape   # (LSTM_Layer, batch, units)



# ## Training ==================================================================================
# ### Define a Loss function and optimizer
# * Use Classification Cross-Entropy loss
# * Use Adam optimizer with learning rate 0.01

num_epochs = 2
learning_rate = 0.01

# Loss and optimizer
loss_obj = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# ## Train the network on the training data

net.train()
print('Start Training ')
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = net(images)
        loss = loss_obj(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

print('Finished Training')
net.eval()


# ## Testing ==================================================================================
# ### Show network prediction
# display ground truth
dataiter = iter(test_loader)
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
# Test the model
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))




# #### Accuracy of each class
class_correct = [0.0] * 10
class_total = [0.0] * 10
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(labels.shape[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


































# ==== Time Series Forecasting =========================================================================
# Dataset: sin πx/2T
class SineDataset(torch.utils.data.Dataset):
    def __init__(self, N=100, L=1000, T=20):
        super().__init__()

        self.T = T # period of sine wave
        self.L = L # length of sine wave
        self.N = N # number of sine waves

        # create time steps
        self.t = torch.zeros((N, L)) # (N, L)
        self.t[:] = torch.arange(end=L) + torch.randint(-4 * T, 4 * T, (N,1))

        # sine function
        self.y = torch.sin(np.pi * self.t / (2.0 * T)) # (N, L)

        # create intput, target data
        self.input = self.y[:, 0:L-1]   # 0 to L-1
        self.target = self.y[:, 1:L]    # 1 to L

    def __len__(self):
        return self.input.size(0,)

    def __getitem__(self, idx):
        return self.input[idx,:], self.target[idx,:]



# create dataset
train_dataset = SineDataset(N=512)
test_dataset = SineDataset(N=16)

# Data loader
batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)





# ### Visualize Dataset
y = train_dataset.y
t = np.arange(0,y.shape[1])

# plt.figure(figsize=(16, 9))
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title('Train data')

# plt.plot(t,y)
plt.plot(t, y[0,:], 'r', linewidth=1.0, label='red')
plt.plot(t, y[1,:], 'g', linewidth=1.0, label='green')
plt.plot(t, y[2,:], 'b', linewidth=1.0, label='blue')
plt.legend()
plt.show()






# ## Define RNN =========================================================
# Recurrent neural network (many-to-many)
# GRU: Many-to-Many → Recurrent Loop로 Class내부에서 자동으로 계산해줌
# GRU Cell → Recurrent Loop를 직접 만들어야함
class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        # define layers (hidden_size = 51)
        self.gru1 = nn.GRUCell(1, 51) # GRUCell
        self.gru2 = nn.GRUCell(51, 51) # GRUCell
        self.linear = nn.Linear(51, 1) # Linear

    def forward(self, x, future_steps=0):
        # print(input.shape) # (N, T)

        # Set initial hidden and cell states (set correct device to tensor)
        # (N(batch_size), hidden_size)
        h_t = torch.zeros((x.shape[0], 51), device=x.device)
        h_t2 = torch.zeros((x.shape[0], 51), device=x.device)

        # loop over time sequence and
        # feed input sequence
        outputs = []        # output list
        for input_t in x.split(1, dim=1):       # sequence 단위로 split 해서 for문을 돌리기 위함
            h_t = self.gru1(input_t, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]

        # loop over time sequence and
        # predict the future sequence
        # (number of steps to predict = future_steps)
        for i in range(future_steps):
            h_t = self.gru1(output, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)

            outputs += [output]

        # concatenate all time steps
        outputs = torch.cat(outputs, dim=-1) # (N, T)
        return outputs

net = RNN().to(device)

# Test case
# - check the default output of model before training
# - make sure that the definition of layers in your model is correct.
print(net(torch.tensor([[0.0]]).to(device)).shape)
print(str(net))


# ## Training
# ### Define a Loss function and optimizer
# * Use MSE loss
# * Use Adam with learning rate 0.01
# Loss and optimizer
loss_obj = nn.MSELoss()

def create_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return optimizer


# ## Train the network on the training data
def train(model, num_epochs, loss_obj):
    model.train()
    optimizer = create_optimizer(model)
    for epoch in range(num_epochs):
        for batch_idx, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = loss_obj(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data))

num_epochs = 6
train(net, num_epochs=num_epochs, loss_obj=loss_obj)


# ## Testing
def test(model, loss_obj, future_steps=1):
    model.eval()
    test_loss = 0
    input_list = []
    output_list = []
    with torch.no_grad():  
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            output = model(input, future_steps=future_steps)
            # test_loss += loss_obj(output, target).cpu().item() # sum up batch loss
            test_loss += loss_obj(output[:, :-future_steps], target).cpu().item() # compute loss of predicted sequence
    
            input_list.append(input.cpu().numpy())
            output_list.append(output.cpu().detach().numpy())

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))

    input_array = np.array(input_list) # (N_loop, N_batch, T_train)
    output_array = np.array(output_list) # (N_loop, N_batch, T_test)

    input_array = np.reshape(input_array, (-1, input_array.shape[-1])) # (N_test, T_in)
    output_array = np.reshape(output_array, (-1, output_array.shape[-1])) # (N_test, T_out)
    return input_array, output_array

future_steps = 1000
input_array, output_array = test(net, loss_obj, future_steps=future_steps)





# visualize
def visualize_output(input_array, output_array, future_steps, title=''):
    # draw the result
    plt.figure(figsize=(30, 10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('t', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)


    L_input = input_array.shape[1]
    L_output = output_array.shape[1]
    def draw(output_seq, color):

        t = np.arange(L_input)
        x = output_seq[:L_input]

        t_future = np.arange(L_input, L_output)
        x_future = output_seq[L_input:]


        plt.axhline(0, color='black')
        plt.axvline(0, color='black')

        plt.plot(t, x, color, linewidth=2.0)
        plt.plot(t_future, x_future, color + ':', linewidth=2.0)


    draw(output_array[0], 'r')
    draw(output_array[1], 'g')
    draw(output_array[2], 'b')
    plt.show()


# TODO
visualize_output(input_array, output_array, future_steps=future_steps)




