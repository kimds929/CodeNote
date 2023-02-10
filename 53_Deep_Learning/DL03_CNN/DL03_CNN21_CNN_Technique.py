# pip install httpimport
import httpimport
url = 'https://raw.githubusercontent.com/kimds929/DS_Library/main/'
# url = 'https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_DeepLearning.py?token=GHSAT0AAAAAAB4QVBFFTGI5FZ7OMEJJWNPYY7DZOZQ'
with httpimport.remote_repo(f"{url}/DS_DeepLearning.py"):
    from DS_DeepLearning import EarlyStopping

import time
import copy

import torch
import torchvision
import torchvision.transforms as transforms

imageset_path =r'C:\Users\Admin\Desktop\DataBase\Image_Data'

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

batch_size = 32
train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


for batch_X, batch_y in trainloader:
    break

print(batch_X.shape, batch_y.shape)

X_sample = batch_X[:3]
y_sample = batch_y[:3]

####################################################################################################################
# (Global Average Pooling) ***

x = torch.randn((256, 96, 128, 128)).cuda()

# Layer
gap_layer = torch.nn.AdaptiveAvgPool2d((1,1))
gap_layer(x)        # 256, 96, 1, 1
gap_layer(x).shape

# function
torch.nn.functional.avg_pool2d(x, x.size()[2:])     # 256, 96, 1, 1
torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # 256, 96, 1, 1
torch.mean(x.view(x.size(0), x.size(1), -1), dim=2) #


###################################################################################################################

# use_cuda = False
#  if use_cuda and torch.cuda.is_available():
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

###################################################################################################################
class CNN_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1):
        super().__init__()
        self.cnn_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2,2)     # width/2, height/2
            )
    def forward(self, X):
        return self.cnn_block(X)       

# valid_score: 1.8174
class CNN_Basic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block1 = CNN_Block(3, 32)
        self.cnn_block2 = CNN_Block(32, 64)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64*8*8, 10)
    
    def forward(self, X):
        self.cb1 = self.cnn_block1(X)
        self.cb2 = self.cnn_block2(self.cb1)
        self.flat = self.flatten(self.cb2)
        self.output = torch.nn.functional.softmax(self.fc1(self.flat), dim=1)
        return self.output

# valid_score: 2.0131
class CNN_GAP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block1 = CNN_Block(3, 32)
        self.cnn_block2 = CNN_Block(32, 64)
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))    # ★
        self.fc1 = torch.nn.Linear(64, 10)
    
    def forward(self, X):
        self.cb1 = self.cnn_block1(X)
        self.cb2 = self.cnn_block2(self.cb1)
        self.gap = self.global_avg_pool(self.cb2).squeeze() # ★
        self.output = torch.nn.functional.softmax(self.fc1(self.gap), dim=1)
        return self.output

# X_sample.shape    # (3,3,32,32)
(3,32,16,16)
(3,64,8,8)

# model_basic = CNN_Basic().to(device)
model_gap = CNN_GAP().to(device)
# model_gap(X_sample.to(device))



# load_state_dict ------------------------------------------------------
# model_basic.load_state_dict(es.optimum[2])
# model_gap.load_state_dict(es.optimum[2])
# predict ------------------------------------------------------

model_basic.eval()
with torch.no_grad():
    pred = model_basic(X_sample.to(device))
torch.argmax(pred,dim=1)
y_sample
# --------------------------------------------------------------


model = copy.deepcopy(model_gap)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 10

es = EarlyStopping()

train_losses = []
valid_losses = []
for e in range(epochs):
    start_time = time.time() # 시작 시간 기록
    # train_set learning*
    model.train()
    train_epoch_loss = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()                   # wegiht initialize
        pred = model(batch_X.to(device))                   # predict
        loss = loss_function(pred, batch_y.to(device))     # loss
        loss.backward()                         # backward
        optimizer.step()                        # update_weight

        with torch.no_grad():
            train_batch_loss = loss.to('cpu').detach().numpy()
            train_epoch_loss.append( train_batch_loss )

    # valid_set evaluation *
    valid_epoch_loss = []
    with torch.no_grad():
        model.eval() 
        for batch_X, batch_y in test_loader:
        # for batch_X, batch_y in valid_loader:
            pred = model(batch_X.to(device))                   # predict
            loss = loss_function(pred, batch_y.to(device))     # loss
            valid_batch_loss = loss.to('cpu').detach().numpy()
            valid_epoch_loss.append( valid_batch_loss )

    with torch.no_grad():
        train_loss = np.mean(train_epoch_loss)
        valid_loss = np.mean(valid_epoch_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # print(f'Epoch: {e + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}')
        # print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {np.exp(valid_loss):.3f}')
        early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(), verbose=2)

        if early_stop == 'break':
            break





