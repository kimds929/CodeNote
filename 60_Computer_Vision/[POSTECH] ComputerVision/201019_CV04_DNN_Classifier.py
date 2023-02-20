# # Deep Nerual Network Classification
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

random_seed = 4332
# 네트워크를 초기화
# 데이터 순서를 shuffle

# 랜덤 생성을 위해 random seed가 필요
# random seed를 고정하면 생성되는 랜덤 number를 고정할 수 있음.
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
# %matplotlib inline
# %matplotlib qt





#  
# ## Classification Dataset  =============================================================================================
# 
# 다음과 같은 classification dataset을 만든다.
# 
# * Generate random (x, y) using `torch.randn` function
#     * if x * y < 0 then label should be 0
#     * else (x * y >= 0) then label should be 1
# 
# ### How to make Dataset
# * `torch.utils.data.Dataset`을 상속받는 Class를 정의한다.
#   * 다음의 두 method는 필수적으로 정의하여야 한다.
#   * `__len__(self)` : Dataset의 크기를 return 한다.
#   * `__getitem__(self, idx)` : 데이터셋의 idx번째 데이터를 return 한다.
# 
# 
# * `torch.utils.data.DataLoader`에서 Dataset을 받아 데이터를 batch로 만드는 역할을 한다.
#   * `enumerate(dataloader)`를 통해 Dataset에 있는 data를 batch 단위로 받아올 수 있다.
#   ~~~python
#   for batch_idx, data, label in enumerate(trainloader):
#         pass
#   ~~~
# 



from torch.utils.data import Dataset, DataLoader

# define dataset class
# __getitem__ returns tensors of shape (2), (1)
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, num_data=5000):
        super(ClassificationDataset, self).__init__()
        self.points = torch.randn((num_data, 2))
    
    def __len__(self):
        return self.points.shape[0]
    
    def __getitem__(self, idx):
        point = self.points[idx, :]
        if point[0] * point[1] < 0:
            label = 0
        else:
            label = 1
        label = torch.tensor([label], dtype=torch.int)
        return point, label


# crate dataset
# train: 5000 data points
# test: 1000 data points
train_data = ClassificationDataset(num_data=5000)
test_data = ClassificationDataset(num_data=1000)


train_data.points.shape
test_data.points.shape

# Test case
#  - check the size of dataset
#  - check the contents of dataset

assert len(train_data) == 5000, "The length of training set should be 5000."
assert len(test_data) == 1000, "The length of test set should be 1000."
is_passed = True
for input, label in train_data:
    if label.dtype != torch.int:
        is_passed = False
    if label not in [0,1]:
        is_passed = False
for input, label in test_data:
    if label.dtype != torch.int:
        is_passed = False
    if label not in [0,1]:
        is_passed = False

assert is_passed, "The label of dataset should be int tensor and 0 or 1"

assert tuple(input.shape) == (2,), 'The input shape should be (2,). Actual: ' + str(input.shape)
assert tuple(label.shape) == (1,),'The output shape should be (1,). Actual: ' + str(label.shape)



# create DataLoader
# train: batch size = 32
# test: batch size = 1
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)


# ### Visualize Dataset
true_list = []
false_list = []
label_list = []
for data, label in train_data:
    if label == 1:
        true_list.append(data.numpy())
    else:
        false_list.append(data.numpy())

true_x = np.array(true_list).squeeze()[:, 0]
true_y = np.array(true_list).squeeze()[:, 1]
false_x = np.array(false_list).squeeze()[:, 0]
false_y = np.array(false_list).squeeze()[:, 1]


plt.figure(figsize=(7, 7))
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title('Train data')

plt.plot(true_x, true_y, 's', markersize=3, marker='o', color='r')
plt.plot(false_x, false_y, 's', markersize=3, marker='x', color='b')





# ## Single layer neural network =============================================================================================
# * Single layer neural network
#     * Fully connected layer - in_features: 2, out_features: 1
#     * Apply Sigmoid activation function for the output layer.

class ANN(torch.nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x

model_single_ann = ANN().to(device)



#  - check the default output of model before training
#  - make sure that the definition of layers in your model is correct.
print (model_single_ann( torch.tensor([[0.1,0.1]]).to(device) ) )
print(str(model_single_ann))







# ### Train the network on the training data ======================================

# #### Define a Loss function and optimizer
# * Use Bineary Cross-Entropy loss
# * Use SGD with learning rate 0.01 and momentum 0.5

# define a loss and optimizer
# TODO
loss_obj = nn.BCELoss()
# loss_obj = nn.CrossEntropyLoss()


def create_optimizer(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    return optimizer




# #### Define training loop ----------------------------------------------------------
def train(model, num_epochs, loss_obj):
    model.train()
    optimizer = create_optimizer(model)
    for epoch in range(num_epochs):
        for batch_idx, (input, target) in enumerate(train_loader):
            # move tensors to device
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
        print()
# model_single_ann( next(iter(train_loader))[0].to(device) )

# train the network
# model_single_ann = ANN().to(device)
# epoch = 10
train(model=model_single_ann, num_epochs=10, loss_obj=loss_obj)






# ### Define test loop ----------------------------------------------------------
def test(model, loss_obj):
    predict_true = []
    predict_false = []

    model.eval()
    test_loss = 0
    correct = 0
    for input, target in test_loader:
        input, target = input.to(device), target.to(device)

        output = model(input)

        test_loss += loss_obj(output, target.float()).data # sum up batch loss
        pred = torch.round(output).data # get the index of the max log-probability
        correct += pred.eq(target.data.float().view_as(pred)).cpu().sum()
        if pred.sum() == 0:
            predict_false.append(input.data.tolist())
        else:
            predict_true.append(input.data.tolist())

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return predict_true, predict_false


# ### Test the network on the test data
# test the network
predict_true, predict_false = test(model_single_ann, loss_obj)



# visualize
def visualize_output(predict_true, predict_false, title=''):
    predict_true_x = np.array(predict_true).squeeze()[:, 0]
    predict_true_y = np.array(predict_true).squeeze()[:, 1]
    predict_false_x = np.array(predict_false).squeeze()[:, 0]
    predict_false_y = np.array(predict_false).squeeze()[:, 1]

    plt.figure(figsize=(7, 7))

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    plt.title(title)

    plt.plot(predict_true_x, predict_true_y, 's', markersize=3, marker='o', color='r')
    plt.plot(predict_false_x, predict_false_y, 's', markersize=3, marker='x', color='b')

visualize_output(predict_true, predict_false, 'Single Layer Neural Network')






# =======================================================================
# ## Linear DNN
# * Linear DNN
#     * Fully connected layer - in_features: 2, out_features: 20
#     * Fully connected layer - in_features: 20, out_features: 20
#     * Fully connected layer - in_features: 20, out_features: 1
#     * Apply no activation function for hidden layers.
#     * Apply Sigmoid activation function for the output layer.

class LinearDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

model_linear_dnn = LinearDNN().to(device)

#  - check the default output of model before training
#  - make sure that the definition of layers in your model is correct.
print( model_linear_dnn( torch.tensor([[0.1,0.1]]).to(device) ) )
print(str(model_linear_dnn))

# ### Train the network on the training data
train(model_linear_dnn, 10, loss_obj)

# ### Test the network on the test data
predict_true, predict_false = test(model_linear_dnn, loss_obj)

visualize_output(predict_true, predict_false, 'Linear DNN')




# ## DNN
# * DNN
#     * Fully connected layer - in_features: 2, out_features: 20
#     * Fully connected layer - in_features: 20, out_features: 20
#     * Fully connected layer - in_features: 20, out_features: 1
#     * Apply ReLU activation function for hidden layers.
#     * Apply Sigmoid activation function for the output layer.



# Deep Neural Network with Relu
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

model_dnn = DNN().to(device)


#  - check the default output of model before training
#  - make sure that the definition of layers in your model is correct.
print( model_dnn( torch.tensor([[0.1,0.1]]).to(device) ) )
print(str(model_dnn))


# ### Train the network on the training data
train(model_dnn, 10, loss_obj)

# ### Test the network on the test data
predict_true, predict_false = test(model_dnn, loss_obj)

visualize_output(predict_true, predict_false, 'Linear DNN')


