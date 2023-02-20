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

path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 7) Computer_Vision\Dataset'
origin_path = os.getcwd()
os.chdir(path)


random_seed = 4332
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

use_cuda = False
if use_cuda and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")



# Dataset -------------------------------------------------
from torch.utils.data import Dataset, DataLoader


# define dataset class
# __getitem__ returns tensors of shape (1,), (1,)
# TODO
class RegressionDataset(Dataset):
    def __init__(self, num_data = 5000):
        super().__init__()

    def __len__(self):
        return

    def __getitem__(self, idx):
        return



# crate dataset
# train: 5000 data points
# test: 1000 data points
# TODO
train_data = None
test_data = None

# Test case
#  - check the size of dataset
assert len(train_data) == 5000, "The length of training set should be 5000."
assert len(test_data) == 1000, "The length of test set should be 1000."
x, y = train_data[0]
assert tuple(x.shape) == (1,), 'The input shape should be (1,). Actual: ' + str(x.shape)
assert tuple(y.shape) == (1,),'The output shape should be (1,). Actual: ' + str(y.shape)

# create DataLoader
# train: batch size = 32
# test: batch size = 1

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)



# ### Visualize Dataset
input_list = []
output_list = []
for x, y in train_data:
    input_list.append(x)
    output_list.append(y)
x = np.array(input_list)
y = np.array(output_list)


plt.figure(figsize=(7, 7))
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title('Train data')

plt.plot(x, y, 's', markersize=3, marker='o')
plt.show()






class ANN(nn.Model):
    def __init__(self):
        super(ANN, self).__init__()

    def forward(self, x):
        return

model_single_ann = ANN().to(device)


#  - check the default output of model before training
#  - make sure that the definition of layers in your model is correct.
print(model_single_ann(torch.tensor([[0.1]])))
print(str(model_single_ann))





# ### Train the network on the training data

# #### Define a Loss function and optimizer
# * Use Mean Squared Error loss
# * Use SGD with learning rate 0.01 and momentum 0.5




# define a loss and optimizer
loss_obj = None

def create_optimizer(model):
    optimizer = None
    return optimizer


# #### Define training loop
def train(model, num_epochs, loss_obj):
    model.train()
    optimizer = create_optimizer(model)
    for epoch in range(num_epochs):
        for batch_idx, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = loss_obj(output, target.float())
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data))



# train the netwok
# num_epochs=10


def test(model, loss_obj):
    model.eval()
    test_loss = 0
    px = []
    py = []

    for input, target in test_loader:
        input, target = input.to(device), target.to(device)
        output = model(input)
        test_loss += loss_obj(output, target).cpu().item() # sum up batch loss

        px.append(input.cpu().numpy())
        py.append(output.cpu().detach().numpy())

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))

    px = np.array(px)[:,0,0]
    py = np.array(py)[:,0,0]
    return px, py



# ### Test the network on the test data
# test the network
# TODO

px, py = None,None

# visualize
def visualize_output(px, py, title=''):
    plt.figure(figsize=(7, 7))
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title(title)

    plt.plot(px, py, 's', markersize=3, marker='o')


visualize_output(px, py, 'Single Layer Neural Network')





# ## Sigmoid DNN
# * Sigmoid DNN
#     * Fully connected layer - in_features: 1, out_features: 20
#     * Fully connected layer - in_features: 20, out_features: 20
#     * Fully connected layer - in_features: 20, out_features: 1
#     * Apply Sigmoid activation function for hidden layers.
# 

class SigmoidDNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return

model_sigmoid_dnn = SigmoidDNN().to(device)


#  - check the default output of model before training
#  - make sure that the definition of layers in your model is correct.
print(model_sigmoid_dnn(torch.tensor([[0.1]])))
print(str(model_sigmoid_dnn))


# ### Train the network on the training data
train(model_sigmoid_dnn,num_epochs=10, loss_obj=loss_obj)

# ### Test the network on the test data
predict_true, predict_false = test(model_sigmoid_dnn, loss_obj)

visualize_output(predict_true, predict_false, 'Sigmoid DNN')


# ## ReLU DNN
# * ReLU
#     * Fully connected layer - in_features: 1, out_features: 20
#     * Fully connected layer - in_features: 20, out_features: 20
#     * Fully connected layer - in_features: 20, out_features: 1
#     * Apply ReLU activation function for hidden layers.


class DNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return

model_dnn = DNN().to(device)


#  - check the default output of model before training
#  - make sure that the definition of layers in your model is correct.
print(model_dnn(torch.tensor([[0.1]])))
print(str(model_dnn))


# ### Train the network on the training data

train(model_dnn,num_epochs=10, loss_obj=loss_obj)


# ### Test the network on the test data
predict_true, predict_false = test(model_dnn, loss_obj)

visualize_output(predict_true, predict_false, 'ReLU DNN')




