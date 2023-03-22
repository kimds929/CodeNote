import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression

# import tensorflow as tf
import torch
from torch import nn

from IPython.display import clear_output
# clear_output(wait=True)


dataset_path = 'C:/Users/Admin/Desktop/DataScience'
torch.cuda.is_available()
torch.__version__


# Random Seed  ============================================================================
random_seed = 4332
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

torch.backends.cudnn.deterministic = True   # GPU 사용시 연산결과가 달라질 수 있음.
torch.backends.cudnn.benchmark = False

np.random.seed(random_seed)
random.seed(random_seed)

# GPU Setting ============================================================================
use_cuda = False
# if use_cuda and torch.cuda.is_available():
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)    


# Dataset Load ============================================================================
# test_df = pd.read_clipboard()
test_dict = {'y1': [2, 12,  6, 19, 5,  5, 14, 8, 12, 20,  1, 10],
            'y2': [0, 0,  1, 1, 0,  0, 1, 0, 1, 1,  0, 1],
            'x1': [ 5,  5, 35, 38,  9, 19, 30,  2, 49,  30,  0, 14],
            'x2': ['a', 'c', 'a', 'b', 'b', 'b', 'a', 'c', 'c', 'a', 'b', 'c'],
            'x3': [46, 23, 23,  3, 36, 10, 14, 28,  5, 19, 42, 32],
            'x4': ['g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g2', 'g2']
            }

test_df = pd.DataFrame(test_dict)

y1_col = ['y1']     # Regressor
y2_col = ['y2']     # Classifier
x_col = ['x1']

y1 = test_df[y1_col]    # Regressor
y2 = test_df[y2_col]    # Classifier
X = test_df[x_col]

y1_np = y1.to_numpy()
y2_np = y2.to_numpy()    # Classifier
X_np = X.to_numpy()

plt.figure(figsize=(10,3))
plt.subplot(121)
plt.title('Regressor')
plt.plot(X, y1, 'o')

plt.subplot(122)
plt.title('Classifier')
plt.plot(X, y2, 'o')
plt.show()


# xp, yp
Xp = np.linspace(np.min(X_np), np.max(X_np), 100).reshape(-1,1)


# [ Sklearn ] ============================================================================
LR = LinearRegression()
LR.fit(X,y1)
print(LR.coef_[0,0], LR.intercept_[0])

plt.plot(X, y1, 'o')
plt.plot(X, X*LR.coef_[0,0] + LR.intercept_[0], linestyle='-', color='orange', alpha=0.5)
plt.show()


# Dataset -----------------------------------------------------------------------------------------------
# https://wikidocs.net/book/2788

n_batch = 3
n_shuffle = 3       # X_np.shape[0]

X_torch = torch.FloatTensor(X_np)
y_torch = torch.FloatTensor(y1_np)


print(np.concatenate([X_np, y1_np], axis=1))

# Dataset
train_ds = torch.utils.data.TensorDataset(X_torch, y_torch)
# ?torch.utils.data.TensorDataset
next(iter(train_ds))
train_ds.tensors

for i, (tx, ty) in enumerate(train_ds, 1):
    print(i, tx.numpy().ravel(), ty.numpy().ravel())

# DataLoader
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=n_batch, shuffle=True)
# ?torch.utils.data.DataLoader
# next(iter(train_loader))
# list(train_loader.dataset)

for i, (tx, ty) in enumerate(train_loader, 1):
    print(i, tx.numpy().ravel(), ty.numpy().ravel())


# Custom Dataset
# class CustomDataset(torch.utils.data.Dataset): 
#   def __init__(self):
#   데이터셋의 전처리를 해주는 부분

#   def __len__(self):
#   데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분

#   def __getitem__(self, idx): 
#   데이터셋에서 특정 1개의 샘플을 가져오는 함수

# # Dataset 상속
# class CustomDataset(Dataset): 
#   def __init__(self):
#     self.x_data = [[73, 80, 75],
#                    [93, 88, 93],
#                    [89, 91, 90],
#                    [96, 98, 100],
#                    [73, 66, 70]]
#     self.y_data = [[152], [185], [180], [196], [142]]

#   # 총 데이터의 개수를 리턴
#   def __len__(self): 
#     return len(self.x_data)

#   # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
#   def __getitem__(self, idx): 
#     x = torch.FloatTensor(self.x_data[idx])
#     y = torch.FloatTensor(self.y_data[idx])
#     return x, y



# model = nn.DataParallel(model)      # Model 복수 GPU사용





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##  (Pytorch) LinearRegression
dataset_path = r'C:\Users\Admin\Desktop\DataScience\Code9) Dataset'
data_wine = pd.read_csv(f"{dataset_path}/datasets_wine.csv", encoding='utf-8-sig')
data_wine

import torch

#  Returns a bool indicating if CUDA is currently available.
torch.cuda.is_available()   #  True
 
#  Returns the index of a currently selected device.
torch.cuda.current_device() #  0
 
#  Returns the number of GPUs available.
torch.cuda.device_count()   #  1
 
#  Gets the name of a device.
torch.cuda.get_device_name(0)   #  'GeForce GTX 1060'
 
#  Context-manager that changes the selected device.
#  device (torch.device or int) – device index to select. 
torch.cuda.device(0)

# Default CUDA device
device = torch.device('cuda')
 
# allocates a tensor on default GPU
a = torch.tensor([1., 2.], device=device)
 
# transfers a tensor from 'C'PU to 'G'PU
b = torch.tensor([1., 2.]).cuda()
 
# Same with .cuda()
b2 = torch.tensor([1., 2.]).to(device=device)




seed = 1
no_cuda = False
torch.manual_seed(seed)
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# device = 'cpu'

kwargs = {'num_workers': 1, 'pin_memory':True} if use_cuda else {}

data_y = data_wine['Aroma']
data_X = data_wine[data_wine.columns.drop('Aroma')]

from sklearn.preprocessing import StandardScaler
SE_X = StandardScaler()
data_X_nor = SE_X.fit_transform(data_X)
SE_y = StandardScaler()
data_y_nor = SE_y.fit_transform(data_y.to_frame())


data_y_torch = torch.FloatTensor(np.array(data_y_nor))
data_X_torch = torch.FloatTensor(np.array(data_X_nor))

from sklearn.model_selection import train_test_split
train_y_torch, test_y_torch, train_X_torch, test_X_torch = train_test_split(data_y_torch, data_X_torch, test_size=0.2, random_state=0)

train_X_torch = train_X_torch.to(device)
train_y_torch = train_y_torch.to(device)

test_X_torch = test_X_torch.to(device)
test_y_torch  = test_y_torch.to(device)

n_batch = 4
train_dataset = torch.utils.data.TensorDataset(train_X_torch, train_y_torch)
test_dataset = torch.utils.data.TensorDataset(test_X_torch, test_y_torch)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=n_batch, shuffle=True)

# train_X_torch
# train_X_torch.shape
# layer_nn1 = torch.nn.Linear(9,2)
# layer_nn1(train_X_torch)


class Net01(torch.nn.Module):
    def __init__(self):
        super(Net01, self).__init__()
        self.nn1 = torch.nn.Linear(9, 18)
        self.nn2 = torch.nn.Linear(18, 36)
        self.nn3 = torch.nn.Linear(36, 1)
        
    def forward(self, x):
        # x1 = self.nn1(x)
        x1 = torch.nn.functional.relu( self.nn1(x) )
        # x2 = self.nn2(x1)
        x2 = torch.nn.functional.relu( self.nn2(x1) )
        x3 = self.nn3(x2) 
        return x3
        
lr = 0.01
momentum = 0.5
epochs = 50

model = Net01().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# loss_function = torch.nn.MSELoss()

loss_hist = []
# https://nuguziii.github.io/dev/dev-002/
for epoch in range(1, epochs+1):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = torch.nn.functional.mse_loss(pred, batch_y)
        # loss = loss_function(pred, batch_y)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        print(f"epoch: {epoch}, loss: {loss}")
        loss_hist.append(float(loss.clone().cpu().detach().numpy()))

plt.plot(loss_hist)




with torch.no_grad():
    result = model(train_X_torch)

plt.plot(result)
plt.plot(train_y_torch)

