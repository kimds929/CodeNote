import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torchvision
import torchvision.models as models

##############################################################


# load dataset
# os.chdir()
image_path = r'D:\DataScience\강의) 강의자료\강의) [FastCampus] 55. 한 번에 끝내는 컴퓨터비전 초격차 패키지 Online\Part 2. 컴퓨터비전 특화 이론과 실습\Part 2. 컴퓨터비전 특화 이론과 실습\Chapter_3. 딥러닝과 컴퓨터비전\Code\data'
image = f'{image_path}/house.jpg'
image = Image.open(image).convert('RGB')

plt.figure()
plt.imshow(image)
plt.show()


## convert pytorch tensor and normalize image
to_tensor = torchvision.transforms.ToTensor()
normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

compose_preprocessing = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                ])


# image_norm = normalizer(to_tensor(image))
image_norm = compose_preprocessing(image)

# plt.imshow(image.permute(1,2,0))

##### torchvision pretrained model ######################

# np.array(dir(models))
pd.Series(np.array(dir(models))).to_clipboard(index=False)
# models.alexnet(pretrained=False)
image_norm = image_norm.unsqueeze(0) ## batchify ## [B, C, H, W]

print(image_norm.shape)

alexnet = models.resnet18(pretrained=True)      # download pretrained model
alexnet.eval() ## freeze weights.
alexnet.train() ## trainable weights.

# Load dataset ###########################################
dataset_path = r'D:\DataBase\Data_Image'
cifar10 = torchvision.datasets.CIFAR10(root=dataset_path, download=True)

len(cifar10)

# Image
for img, gt in cifar10:
    print(img.size, gt)
    plt.imshow(img)
    plt.show()

    img = normalizer(to_tensor(img))
    img = img.unsqueeze(0)
    
    print(img.shape)    
    break

