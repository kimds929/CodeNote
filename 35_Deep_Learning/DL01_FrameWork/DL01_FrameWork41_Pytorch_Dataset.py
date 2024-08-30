import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import torch
import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from glob import glob
from PIL import Image
import os
# from skimage import io, transform

# https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html


# Start Setting ------------------------------------------------------
seed = 1
no_cuda = False
torch.manual_seed(seed)
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory':True} if use_cuda else {}

# Ready Dataset =========================================================================================================
# (cifar from tensorflow url) ---------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train_torch = torch.FloatTensor(x_train).permute(0,3,1,2)[:1000]
y_train_torch = torch.tensor(y_train).squeeze().long()[:1000]
x_test_torch = torch.FloatTensor(x_test).permute(0,3,1,2)[:200]
y_test_torch = torch.tensor(y_test).squeeze().long()[:200]

# Dataset
train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
test_dataset = torch.utils.data.TensorDataset(x_test_torch, y_test_torch)

# (next(iter(train_dataset))[0].shape, next(iter(train_dataset))[1].shape)
# for i, (tx, ty) in enumerate(train_dataset, 1):
#     print(i, tx.numpy().ravel(), ty.numpy().ravel())
#     break


n_batch = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=n_batch, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=n_batch, shuffle=False)

# (next(iter(train_loader))[0].shape, next(iter(train_loader))[1].shape)
# for i, (tx, ty) in enumerate(train_loader, 1):
#     print(i, tx.numpy().ravel(), ty.numpy().ravel())
#     break


# (cifar from image_file) ---------------------------------------------------------------------
folder_path = r'D:\Python\강의) [FastCampus] 딥러닝 올인원 패키지'
absolute_path = r'D:\Python\강의) [FastCampus] 딥러닝 올인원 패키지\dataset'

current_path = os.getcwd()
os.chdir(absolute_path)

train_paths = glob('./cifar/train/*.png')
test_paths = glob('./cifar/test/*.png')
# os.chdir(current_path)

# path label 확인
path = train_paths[0]
label_name = os.path.basename(path).split('_')[-1].replace('.png','')
label_name

# get train_label
def get_label_name(path):
    return os.path.basename(path).split('_')[-1].replace('.png','')
label_names = [get_label_name(path) for path in train_paths]
classes = np.unique(label_names)
classes

# ohe = np.array(classes == label_name, np.uint8)   # onehot_encode
# ohe
# lb = np.argmax(ohe)   # label_encode
# lb

# # Image_load
# image = plt.imread(path)
# image.shape



# DataSet - Class (Custom Dataset)
class Torch_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

        # get label
        self.label_names = [os.path.basename(path).split('_')[-1].replace('.png','') for path in self.data_paths]
        self.classes = np.unique(label_names)
        self.labels = [np.argmax(self.classes == label_name) for label_name in self.label_names]
    
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        # read Image
        path = self.data_paths[index]
        image = Image.open(path)
        # image = Image.open(path).convert("L") # gray scale
        label = self.labels[index]  # label

        if self.transform:
            image = np.array(self.transform(image))
        else:
            image = np.array(torch.tensor(np.array(image)).permute(2,0,1))
        return image, label                                         # batch, channel, height, width
        # return np.array(torch.tensor(image).permute(1,2,0)), label  # batch, height, width, channel


n_batch = 64  
train_loader = torch.utils.data.DataLoader(
    Torch_Dataset(train_paths[:1000],
        torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.406], std=[0.225])
            ])
        ),
    batch_size=n_batch, shuffle=True, **kwargs
    )

test_loader = torch.utils.data.DataLoader(
    Torch_Dataset(test_paths[:200], 
        torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.406], std=[0.225])
            ])
        ),
    batch_size=n_batch, shuffle=False, **kwargs
    )

(next(iter(train_loader))[0].shape, next(iter(train_loader))[1].shape)  # batch_data.shape
# for i, (tx, ty) in enumerate(train_loader, 1):
#     print(i, tx.numpy().ravel(), ty.numpy().ravel())
#     break

# sample_image
plt.title(classes[next(iter(train_loader))[1][0]])
plt.imshow(next(iter(train_loader))[0][0].permute(1,2,0))               # image
next(iter(train_loader))[1][:10].type()

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





# Model Learning =========================================================================================================
# modeling
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1)       # channel
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = torch.nn.Linear(in_features=5*5*50, out_features=500)
        self.fc2 = torch.nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 5*5*50)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)
torch.nn.functional.max_pool2d
lr = 0.001
momentum = 0.5
epochs = 5
log_interval = 100

model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


# Tensorboard ****
# from torch.utils.tensorboard import SummaryWriter
# writer = torch.utils.tensorboard.SummaryWriter(log_dir=None)    # runs folder 생성 및 현시각 폴더 생성하여 저장
writer = torch.utils.tensorboard.SummaryWriter(log_dir='torch_tensorboard')

# Learning_Rate Scheduler ****
# from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)

for epoch in range(1, epochs+1):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()       # gradient_zeors(initialize)
        pred = model(batch_x)       # forward
        loss = torch.nn.functional.nll_loss(input=pred, target=batch_y)
        loss.backward()             # backward
        optimizer.step()            # weight_update
    
    if epoch % 1 == 0:
        with torch.no_grad():
            test_loss = 0
            test_accuracy = 0
            for test_batch, (batch_test_x, batch_test_y) in enumerate(test_loader,1):
                test_pred = model(batch_test_x)
                test_pred_result = test_pred.argmax(axis=1, keepdim=True)
                # test_pred_result = test_pred.argmax(dim=1, keepdim=True)

                test_loss += torch.nn.functional.nll_loss(input=test_pred, target=batch_test_y, reduction='sum').item()
                test_accuracy += test_pred_result.eq(batch_test_y.view_as(test_pred_result)).sum().item()
                    # batch_test_y.vew_as(test_pred_result) : batch_test_y의 차원을 test_pred_result와 동일하게 맞춰준다
            test_loss /= test_batch
            test_accuracy /= test_batch
            print(f'epoch {format(epoch, "4d")}) test_loss: {format(test_loss, ".2f")} test_accuracy: {format(test_accuracy, ".2f")}')

    # Add to Scheduler ****
    scheduler.step(test_accuracy, epoch)

    print(f'\nTest set: Average loss: {format(test_loss,".4f")}, Accuracy: {format(test_accuracy, ".2f")}')
    
    # Add to TensorBoard ****
    if epoch == 0:
        grid = torchvision.utils.make_grid(batch_x)
        writer.add_image('images', grid, epoch)
        writer.add_graph(model, batch_x)
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', test_accuracy, epoch)

writer.close()



# tensorboard --logdir=D:/Python/"강의) [FastCampus] 딥러닝 올인원 패키지"/runs/Sep22_17-35-46_KN19S7023 --port 8008
# tensorboard --logdir=D:/Python/"강의) [FastCampus] 딥러닝 올인원 패키지"/torch_tensorboard --port 8008




# Weight Save & Load **** ---------------------------------------------------------------------------------------------------

# Weight Save
save_path = './model/'
torch.save(model.state_dict(), save_path + 'model_weight.pt')            # Weight 저장하기
# model.state_dict()
# model.state_dict().keys()


# Load Weight
load_model = Net()
load_weight_dict = torch.load(save_path + 'model_weight.pt')            # Weight 불러오기
load_model.load_state_dict(load_weight_dict)        # 불러온 Weight 적용

# load_model.parameters
# load_model.eval()
# load_model.train()




# Model Save & Load **** ---------------------------------------------------------------------------------------------------

# Save Model
# save_path = './model/model_weight.pt'
torch.save(model, save_path + 'model.pt')       # Model 저장하기

# Load Model
load_model = torch.load(save_path + 'model.pt') # Model 불러오기




# Save, Load and Resuming Training **** ---------------------------------------------------------------------------------------------------

# Save CheckPoint
torch.save({
    'epoch':epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
    }, save_path + 'model_checkpoint.pt')


# LoadCheckPoint
load_model = Net()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
load_checkpoint = torch.load(save_path + 'model_checkpoint.pt')
load_checkpoint.keys()

load_model.load_state_dict(load_checkpoint['model_state_dict'])             # Weight Load

load_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
load_optimizer.load_state_dict(load_checkpoint['optimizer_state_dict'])     # Optimizer Load






















