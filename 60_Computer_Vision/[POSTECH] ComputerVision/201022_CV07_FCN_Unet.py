import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv
import PIL
from PIL import Image
import collections
import os.path as osp
import scipy
import scipy.misc
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 7) Computer_Vision\Dataset'
path = r'/home/pirl/data'
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

use_cuda = True
if use_cuda and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# ① Reshape
# ② Transpose Convolution   # Image Segmentation
# ③ Upsampling              # GAN 에서 많이 쓰임
# ④ Convolution

class_names = np.array(['background','aeroplane','bicycle','bird','boat','bottle','bus',
                        'car','cat','chair','cow','diningtable','dog','horse','motorbike','person',
                        'potted plant','sheep','sofa','train','tv/monitor',
                       ])



imgsets_file = osp.join('Kitti', '%s.txt'% 'train')
for did in open(imgsets_file):
    did = did.strip()
    print(did)
    did = did.split()








class KITTIdataset(torch.utils.data.Dataset):
    class_names = np.array(['background', 'road'])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform
        dataset_dir = osp.join(self.root, 'Kitti')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(dataset_dir, '%s.txt'%split)
            for did in open(imgsets_file):
                did = did.strip()
                did = did.split()
                img_file = osp.join(dataset_dir, 'data_road/%s' % did[0])
                lbl_file = osp.join(dataset_dir, 'data_road/%s' % did[1])
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = 1
        
        if self._transform:
            img, lbl = self.transform(img, lbl)
            return img, lbl
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1) # H W C -> C H W
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl






train_loader = torch.utils.data.DataLoader(KITTIdataset(root = './', split = 'train', transform = True), 
                                           batch_size = 1, shuffle = True)

val_loader = torch.utils.data.DataLoader(KITTIdataset(root = './', split = 'val', transform = True), 
                                         batch_size = 1, shuffle = False)

vgg16 = torchvision.models.vgg16(pretrained = True)
print(vgg16)









# # Define the Network
# -VGG16
# - FCN model
# ![convnet](./resources/fcn_upsampling.png "Variable")
class FCN(nn.Module):
    def __init__(self, num_class = 21):
        super(FCN, self).__init__()
        
        ## Why padding 100?? https://github.com/shelhamer/fcn.berkeleyvision.org
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.features3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.features4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
                
        self.features5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace = True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace = True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_class, 1))
        
        self.upscore2 = nn.ConvTranspose2d(num_class, num_class, kernel_size = 4, stride = 2, bias = False)
        self.upscore4 = nn.ConvTranspose2d(num_class, num_class, kernel_size = 4, stride = 2, bias = False)
        self.upscore8 = nn.ConvTranspose2d(num_class, num_class, kernel_size = 16, stride = 8, bias = False)
        
        self.score_pool4 = nn.Conv2d(512, num_class, 1)
        self.score_pool3 = nn.Conv2d(256, num_class, 1)
        
        self.params = [self.features1, self.features2, self.features3, 
                       self.features4, self.features5]
        
    def upsample(self, x, size):
        return nn.functional.upsample(x, size = size, mode = 'bilinear')
                             
    def forward(self, inputs):
        x = self.features1(inputs)
        pool1 = self.maxpool(x)
        x = self.features2(pool1)
        pool2 = self.maxpool(x)
        x = self.features3(pool2)
        pool3 = self.maxpool(x)
        x = self.features4(pool3)
        pool4 = self.maxpool(x)
        x = self.features5(pool4)
        pool5 = self.maxpool(x)
        x = self.classifier(pool5)
        
        # also use getattr with for loop ...
        x = self.upscore2(x)
        
        pool4 = self.score_pool4(pool4)
        pool4 = pool4[:, :, 5:5 + x.size()[2], 5:5 + x.size()[3]]
        x = torch.add(x, pool4)
        
        x = self.upscore4(x)
        
        pool3 = self.score_pool3(pool3)
        pool3 = pool3[:, :, 9:9 + x.size()[2], 9:9 + x.size()[3]]
        x = torch.add(x, pool3)
        
        x = self.upscore8(x)
        x = x[:, :, 33:33 + inputs.size()[2], 33:33 + inputs.size()[3]]
        return x
    
    def copy_params(self, vgg):
        for l1, l2 in zip(vgg.features, self.params):
            if (isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data







m1 = nn.ConvTranspose2d(2, 2, kernel_size = 4, stride = 2, bias = False)
m2 = nn.ConvTranspose2d(2, 2, kernel_size = 4, stride = 2, bias = False)
m3 = nn.ConvTranspose2d(2, 2, kernel_size = 16, stride = 8, bias = False)
temp_input = torch.randn(1,2,50,50)
temp_output = m1(temp_input)
print(temp_input.size())
print(temp_output.size())



model_type = 'FCN'
if not model_type in ['unet', 'FCN']:
    raise Exception('unsupproted model type.')

if model_type == 'unet':
    model = UNet(21)
elif model_type == 'FCN':
    model = FCN(21)
    
print(model)















# # U-Net =================================================================================
# -  U-Net model
# ![unet](./resources/unet.png "Variable")



class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)   # skip connection 줄때, concat
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)  # shape: (batch, channel, height, width)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))  # dim=1: channel단위로 concat하겠다.
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        return F.upsample(final, x.size()[2:], mode='bilinear')



m1 = nn.ConvTranspose2d(2, 2, kernel_size = 4, stride = 2, bias = False)
m2 = nn.ConvTranspose2d(2, 2, kernel_size = 4, stride = 2, bias = False)
m3 = nn.ConvTranspose2d(2, 2, kernel_size = 16, stride = 8, bias = False)
temp_input = torch.randn(1,2,50,50)
temp_output = m1(temp_input)
print(temp_input.size())
print(temp_output.size())



model_type = 'unet'
if not model_type in ['unet', 'FCN']:
    raise Exception('unsupproted model type.')

if model_type == 'unet':
    model = UNet(21)
elif model_type == 'FCN':
    model = FCN(21)
    
print(model)










# ## Measure accuracy and visualization =========================================

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def visualization(net, image, epoch, device):
    net.to('cpu')
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    img = image
    img = np.array(img, dtype = np.uint8)
    img = img[:, :, ::-1] # channel RGB -> BGR
    img = img.astype(np.float64)
    img -= mean_bgr
    img = img.transpose(2, 0, 1) # H W C -> C H W
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)

    score = net(img)
    lbl_pred = score.data.max(1)[1].cpu().numpy()
    lbl_pred = np.squeeze(lbl_pred)
    imageio.imwrite('./pred/mask_'+str(epoch+1)+'.png', lbl_pred)
    # scipy.misc.imsave('./pred/mask_'+str(epoch+1)+'.png', lbl_pred)

    input_img = image
    input_img = np.array(input_img, dtype = np.uint8)
    color = [0, 255, 0, 127] 
    color = np.array(color).reshape(1, 4)
    shape = input_img.shape
    segmentation = lbl_pred.reshape(shape[0], shape[1], 1)
    output = np.dot(segmentation, color)

    # output = scipy.misc.toimage(output, mode = "RGBA")
    # background = scipy.misc.toimage(input_img)
    # background.paste(output, box = None, mask = output)

    output = Image.fromarray(output.astype(np.uint8))
    background = Image.fromarray(input_img.astype(np.uint8))
    background.paste(output, box=None, mask=output)
    background.save('./overlay/overlay_'+str(epoch+1)+'.png')

    # imageio.imwrite('./overlay/overlay_'+str(epoch+1)+'.png', np.array(background))
    # scipy.misc.imsave('./overlay/overlay_'+str(epoch+1)+'.png', np.array(background))
    net.to(device)
    return background














# # Train ===============================================================

import torch.optim as optim

if model_type == 'unet':
    net = UNet(21)
elif model_type == 'FCN':
    vgg16 = torchvision.models.vgg16(pretrained = True)
    net = FCN(num_class = 2)
    net.copy_params(vgg16)
    del vgg16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

training_epochs = 5
learning_rate = 1e-8

criterion = nn.CrossEntropyLoss(size_average=False) 
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 0.0005)
# optimizer = optim.Adam(net.parameters(), lr=learning_rate)

best_iou = 0
num_class = len(train_loader.dataset.class_names)
j=0

for epoch in range(training_epochs):
    print ('current epoch : %d'%(epoch))
    running_loss = 0.0
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # load data, forward
        data, target = data.to(device), target.to(device)
        score = net(data)
        
#         import pdb; pdb.set_trace()
        loss = criterion(score, target)
        if batch_idx % 20 ==0:
            print ('batch : %d, loss : %f'%(batch_idx, loss.item()))
        j += 1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #validation
    net.eval()
    
    val_loss = 0
    lbl_trues = []
    lbl_preds = []
    metrics = []

    for batch_idx, (data, target) in enumerate(val_loader):
        # load data, forward
        data, target = data.to(device), target.to(device)
        score = net(data)

        # calc val loss, accuracy
        loss = criterion(score, target)
        val_loss += loss.item() / len(data)

        lbl_pred = score.data.max(1)[1].cpu().numpy()  #[0]에는 value가 [1]에는 index가 존재
        lbl_true = target.data.cpu()

        for lt, lp in zip(lbl_true, lbl_pred):
            tmp = label_accuracy_score(lt.numpy(), lp, num_class)
            metrics.append(tmp)
            
    val_loss /= len(val_loader)
    metrics = np.mean(metrics, axis = 0)
    mean_iou = metrics[2]
    print ('val loss : %f, mean_iou : %f'%(val_loss, mean_iou))
    
    if best_iou < mean_iou:
        best_iou = mean_iou
        print("Best model saved")
        torch.save(net.state_dict(), './model_best.pth')
        
    #visualization
    img = PIL.Image.open('./Kitti/data_road/testing/image_2/um_000058.png')
    visualization(net, img, epoch, device)
    
    net.train()
        
print('Finished Training')

# visualization(net, img, epoch, device)

next(iter(train_loader))[0].shape
image = next(iter(train_loader))[0].numpy()[0]
label = next(iter(train_loader))[1].numpy()[0]

np.min(image), np.max(image)
plt.imshow(label)



# plt.imshow(np.transpose(image, [1,2,0]))


