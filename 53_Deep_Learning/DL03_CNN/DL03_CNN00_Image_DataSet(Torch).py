import torchvision
import torchvision.transforms as transforms

imageset_path =r'C:\Users\Admin\Desktop\DataBase\Image_Data'

# MNIST (63.5 MB) ------------------------------------------------------------------
mnist_train = torchvision.datasets.MNIST(root = imageset_path,
                            train=True,
                            download=True,
                            transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root = imageset_path,
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())
# print('number of training data : ', len(train_data))
# print('number of test data : ', len(test_data))


# FashionMNIST (81.8 MB) -------------------------------------------------------------
transform = transforms.Compose([
                            #    transforms.Resize((35,35)),
                               transforms.ToTensor(),
                               ])

mnist_fashion_train = torchvision.datasets.FashionMNIST(root=imageset_path,
                          train=True,
                          transform=transform,
                          download=True)

mnist_fashion_test = torchvision.datasets.FashionMNIST(root=imageset_path,
                         train=False,
                         transform=transform,
                         download=True)
# classes = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 
#          5: 'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle-boot'}


# CIFAR10 (177 MB) -------------------------------------------------------------
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

# STL10 (2.9 GB) -------------------------------------------------------------
stl10_train = torchvision.datasets.STL10(root=path2data, split='train', download=True, transform=transforms.ToTensor())
stl10_test = torchvision.datasets.STL10(root=path2data, split='test', download=True, transform=transforms.ToTensor())


# batch_size = 4
# train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)
# test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

