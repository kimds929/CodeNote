

import torch.nn as nn

################################################################################################################################
# nn.Sequential 
# 계층(layer)들을 순차적으로 쌓아 올릴 수 있도록 해줍니다. 이 안에 정의된 계층들은 순차적으로 실행됩니다.
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(32*14*14, 10)
)

# 새 레이어 추가
new_layer = nn.Linear(20, 30)
model = nn.Sequential(*list(model) + [new_layer])
print(model)

# 특정 레이어 삭제 (예: 마지막 레이어 삭제)
model = nn.Sequential(*list(model)[:-1])

print(model)

################################################################################################################################
# nn.ModuleList
# 파이썬 리스트와 유사하지만, 리스트에 추가되는 모듈들이 PyTorch 모델의 일부로 자동 등록됩니다. 여러 계층이나 블록을 리스트로 묶어서 사용할 때 유용합니다.
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
        ])

        # 레이어 추가
        self.layers.append(nn.Linear(30, 20))

        # 특정 레이어 삭제 (예: 첫 번째 레이어 삭제)
        del self.layers[0]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


################################################################################################################################
# nn.ModuleDict
# 이름이 지정된 모듈들을 딕셔너리 형태로 저장할 수 있는 컨테이너입니다. 이름을 통해 모듈을 쉽게 참조할 수 있으며, 다양한 계층들을 관리할 때 유용합니다.
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.ModuleDict({
            'conv1': nn.Conv2d(1, 32, kernel_size=3),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(32, 64, kernel_size=3),
            'relu2': nn.ReLU(),
        })

        # 레이어 추가
        self.layers['conv3'] = nn.Conv2d(64, 32, kernel_size=3)

        # 특정 레이어 삭제
        del model['conv3']
    
    def forward(self, x):
        x = self.layers['conv1'](x)
        x = self.layers['relu1'](x)
        x = self.layers['conv2'](x)
        x = self.layers['relu2'](x)
        return x



################################################################################################################################
# nn.Module
# 모든 신경망 계층의 기본 클래스입니다. 사용자 정의 모델이나 계층을 만들 때 이 클래스를 상속받아야 합니다. 모델의 구조와 forward 패스를 정의하는데 사용됩니다.
class MyCustomModel(nn.Module):
    def __init__(self):
        super(MyCustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32*26*26, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x




################################################################################################################################
# nn.ParameterList : 파라미터들을 리스트로 관리한다.
# nn.ParameterDict : 파라미터들을 딕셔너리로 관리한다.
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.param_list = nn.ParameterList([
            nn.Parameter(torch.randn(10, 10)),
            nn.Parameter(torch.randn(10, 10))
        ])
        
        self.param_dict = nn.ParameterDict({
            'weight1': nn.Parameter(torch.randn(10, 10)),
            'weight2': nn.Parameter(torch.randn(20, 20))
        })
    
    def forward(self, x):
        x = x.mm(self.param_list[0])  # matrix multiplication with first parameter
        x = x.mm(self.param_dict['weight1'])  # matrix multiplication with named parameter
        return x



