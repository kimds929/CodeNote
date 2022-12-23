import os 
import numpy as np
import pandas as pd

import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
# import tensorflow as tf

from PIL import Image
import cv2

apple_path = r'D:\작업방\업무 - 자동차 ★★★\Workspace_Python\DataSet_Image\Apple'
train_folder_path = r'D:\작업방\업무 - 자동차 ★★★\Workspace_Python\DataSet_Image\Fruit360_Sample\train_set'
files_path = [f"{train_folder_path}/{f}" for f in os.listdir(train_folder_path) if 'apple' in f]


# (function) make_batch_from_path ------------------------------------------------------------------
def make_batch_from_path(paths, labels=None, batch_size=None, shuffle=False, random_state=None, shape_format="BHWC"):
    batch_size = len(paths) if batch_size is None else batch_size
    
    if shuffle is True:
        index = np.arange(len(paths))
        rng = np.random.default_rng(random_state)
        index_permute = rng.permutation(index)

        loading_paths = np.array(paths)[index_permute]
        if labels is not None:
            loading_labels = np.array(labels)[index_permute]
    else:
        loading_paths = np.array(paths)
        if labels is not None:
            loading_labels = np.array(labels)

    shape_format_dict = {'B':0, 'H':1, 'W':2, 'C':3}
    shape_format_list = [shape_format_dict[c] for c in shape_format]

    for i in range(0, len(loading_paths), batch_size):
        if labels is None:
            yield np.stack([plt.imread(p) for p in loading_paths[i:i+batch_size]]).transpose(*shape_format_list)
        else:
            yield (np.stack([plt.imread(p) for p in loading_paths[i:i+batch_size]]).transpose(*shape_format_list), loading_labels[i:i+batch_size])
# --------------------------------------------------------------------------------------------------------------------


train_loader = make_batch_from_path(files_path)
batch_img = next(train_loader)
norm_img = np.array([cv2.resize(img, (25,25))/255 for img in batch_img])
torch_img = torch.tensor(norm_img.transpose(0,3,1,2), dtype=torch.float32)


a0 = plt.imread(f"{train_folder_path}/apple_0.jpg")
a1 = cv2.resize(a0, (25,25))
a1.shape
a2 = torch.tensor(a1.transpose(2,0,1)[np.newaxis,...]/255,dtype=torch.float32)
a2.shape
a3 = (a2.numpy().transpose(0,2,3,1)*255).astype(np.uint8)[0]
a3.shape
plt.imshow(a3)




class FlattenTranspose(torch.nn.Module):
    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape
    
    def forward(self,x):
        return x.view(x.shape[0], *list(self.out_shape))

## ANN_AutoEncoder ###############################################################################
class Torch_AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            OrderedDict([
                ('en01_flatten', torch.nn.Flatten()),
                ('en02_Linear', torch.nn.Linear(25*25*3, 1024)),
                ('en03_Linear', torch.nn.Linear(1024, 256)),
                ('en04_Linear', torch.nn.Linear(256, 16)),
                ('en05_Linear', torch.nn.Linear(16, 4)),
                ('en06_Linear', torch.nn.Linear(4, 2))
            ])
        )
        self.decoder = torch.nn.Sequential(
            OrderedDict([
                ('de01_Linear', torch.nn.Linear(2, 4)),
                ('de02_Linear', torch.nn.Linear(4, 16)),
                ('de03_Linear', torch.nn.Linear(16, 256)),
                ('de04_Linear', torch.nn.Linear(256, 1024)),
                ('de05_Linear', torch.nn.Linear(1024, 25*25*3)),
                ('de06_flattenT', FlattenTranspose((3,25,25)))
            ])
        )
    
    def forward(self, X):
        self.letent = self.encoder(X)
        self.restruct = self.decoder(self.letent)
        return self.restruct


## ANN_AutoEncoder (Simple) ###############################################################################
class Torch_AutoEncoder2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            OrderedDict([
                ('en01_flatten', torch.nn.Flatten()),
                ('en02_Linear', torch.nn.Linear(25*25*3, 256)),
                ('en04_Linear', torch.nn.Linear(256, 16)),
                ('en06_Linear', torch.nn.Linear(16, 2))
            ])
        )
        self.decoder = torch.nn.Sequential(
            OrderedDict([
                ('de01_Linear', torch.nn.Linear(2, 16)),
                ('de03_Linear', torch.nn.Linear(16, 256)),
                ('de05_Linear', torch.nn.Linear(256, 25*25*3)),
                ('de06_flattenT', FlattenTranspose((3,25,25)))
            ])
        )
    
    def forward(self, X):
        self.letent = self.encoder(X)
        self.restruct = self.decoder(self.letent)
        return self.restruct

## ANN_AutoEncoder with ReLU ###############################################################################
class Torch_AutoEncoder3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            OrderedDict([
                ('en01_flatten', torch.nn.Flatten()),
                ('en02_Linear', torch.nn.Linear(25*25*3, 256)),
                ('en03_Act', torch.nn.ReLU()),
                ('en04_Linear', torch.nn.Linear(256, 16)),
                ('en05_Act', torch.nn.ReLU()),
                ('en06_Linear', torch.nn.Linear(16, 2))
            ])
        )
        self.decoder = torch.nn.Sequential(
            OrderedDict([
                ('de01_Linear', torch.nn.Linear(2, 16)),
                ('de02_Act', torch.nn.ReLU()),
                ('de03_Linear', torch.nn.Linear(16, 256)),
                ('de04_Act', torch.nn.ReLU()),
                ('de05_Linear', torch.nn.Linear(256, 25*25*3)),
                ('de06_flattenT', FlattenTranspose((3,25,25)))
            ])
        )
    
    def forward(self, X):
        self.letent = self.encoder(X)
        self.restruct = self.decoder(self.letent)
        return self.restruct



# < Convolution >
# output = (input + 2·padding - kernel_size)/stride + 1
# < Transpose Convolution >
# output = (input - 1) × stride + kernel_size - 2·padding

## CNN_AutoEncoder ###############################################################################
class Torch_CNNAutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            OrderedDict([
                ('en01_Conv', torch.nn.Conv2d(in_channels=3, out_channels=32, 
                                              kernel_size=(3,3), stride=2, padding=1)),
                # ('en02_Act', torch.nn.ReLU()),
                ('en03_Conv', torch.nn.Conv2d(32, 64, (3,3), 2, 1)),
                # ('en04_Act', torch.nn.ReLU()),
                ('en05_Flatten', torch.nn.Flatten()),
                ('en06_Linear', torch.nn.Linear(64*7*7, 2))
            ])
        )
        self.decoder = torch.nn.Sequential(
            OrderedDict([
                ('de01_Linear', torch.nn.Linear(2, 32*7*7)),
                ('de02_FlattenT', FlattenTranspose((32, 7, 7))),
                ('de03_ConvT', torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, 
                                                         kernel_size=(3,3), stride=2, padding=1)),
                # ('de04_Act', torch.nn.ReLU()),
                ('de05_ConvT', torch.nn.ConvTranspose2d(64, 32, (3,3), 2, 1)),
                # ('de06_Act', torch.nn.ReLU()),
                ('de07_ConvT', torch.nn.ConvTranspose2d(32, 3, (3,3), 1, 1))
            ])
        )

    def forward(self, X):
        self.letent = self.encoder(X)
        self.restruct = self.decoder(self.letent)
        return self.restruct
    

## CNN_AutoEncoder with ReLU ###############################################################################
class Torch_CNNAutoEncoder2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            OrderedDict([
                ('en01_Conv', torch.nn.Conv2d(in_channels=3, out_channels=32, 
                                              kernel_size=(3,3), stride=2, padding=1)),
                ('en02_Act', torch.nn.ReLU()),
                ('en03_Conv', torch.nn.Conv2d(32, 64, (3,3), 2, 1)),
                ('en04_Act', torch.nn.ReLU()),
                ('en05_Flatten', torch.nn.Flatten()),
                ('en06_Linear', torch.nn.Linear(64*7*7, 2))
            ])
        )
        self.decoder = torch.nn.Sequential(
            OrderedDict([
                ('de01_Linear', torch.nn.Linear(2, 32*7*7)),
                ('de02_FlattenT', FlattenTranspose((32, 7, 7))),
                ('de03_ConvT', torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, 
                                                         kernel_size=(3,3), stride=2, padding=1)),
                ('de04_Act', torch.nn.ReLU()),
                ('de05_ConvT', torch.nn.ConvTranspose2d(64, 32, (3,3), 2, 1)),
                ('de06_Act', torch.nn.ReLU()),
                ('de07_ConvT', torch.nn.ConvTranspose2d(32, 3, (3,3), 1, 1))
            ])
        )

    def forward(self, X):
        self.letent = self.encoder(X)
        self.restruct = self.decoder(self.letent)
        return self.restruct
    
    
#########################################################################################################

# np.array(dir(torch.nn)[-150:])


AE_Model = Torch_AutoEncoder()
AE_Model = Torch_AutoEncoder2()
AE_Model = Torch_AutoEncoder3()
AE_Model = Torch_CNNAutoEncoder()
AE_Model = Torch_CNNAutoEncoder2()
# AE_Model(torch_img)
# r1 = torch.reshape(AE(a2), (3,25,25))
# r1.shape
# r1_numpy = r1.detach().numpy().transpose(1,2,0)


torch_img = torch.tensor(norm_img.transpose(0,3,1,2), dtype=torch.float32)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(AE_Model.parameters())
epochs = 1000

losses = []

for e in range(epochs):
    optimizer.zero_grad()
    pred = AE_Model(torch_img)
    loss = loss_function(pred, torch_img)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    print(f"({e+1} epoch) loss: {losses[-1]}")
    # print(f"({e+1} epoch) loss: {losses[-1]}", end='\r')
    # time.sleep(0.3)



# encoder - decoder
a1 = AE_Model(torch_img[1,:].reshape(1,3,25,25)).detach()
a1_img = (a1.numpy()*255).astype(np.uint8)[0].transpose(1,2,0)
plt.imshow(a1_img)

# encoder
AE_Model.encoder(torch_img[1,:].reshape(1,3,25,25))

# decoder
latent = torch.tensor([[0,1]], dtype=torch.float32)
a2 = AE_Model.decoder(latent).detach()
a2_img = (a2.numpy()*255).astype(np.uint8)[0].transpose(1,2,0)
plt.imshow(a2_img)


a1 = torch.tensor(np.random.random((1,32, 7, 7)), dtype=torch.float32)
l1 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=2, padding=1)
r2 = l1(a1)



l2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=2, padding=1)
l2(r2).shape








