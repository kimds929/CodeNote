
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
imageset_path =r'C:\Users\Admin\Desktop\DataBase\Image_Data'



transform = transforms.Compose(
    [
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

cifar10_train = torchvision.datasets.CIFAR10(root=imageset_path, train=True,
                                        download=True, transform=transform)

cifar10_test = torchvision.datasets.CIFAR10(root=imageset_path, train=False,
                                       download=True, transform=transform)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


batch_size = 64
train_loader = DataLoader(dataset=cifar10_train,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

test_loader = DataLoader(dataset=cifar10_test,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)







