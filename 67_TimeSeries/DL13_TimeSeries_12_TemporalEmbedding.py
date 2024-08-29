import torch
import torch.nn as nn
import numpy as np

class Time2Vec(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        # Linear Component
        self.linear_weights = nn.Parameter(torch.randn(input_dim, 1))
        self.linear_bias = nn.Parameter(torch.randn(input_dim, 1))
        
        # Periodic Components
        self.periodic_weights = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))
        self.periodic_bias = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))

        # NonLinear Purse Periodic Component
        self.nonlinear_weights = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))
        self.nonlinear_bias = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))

    def forward(self, x):
        # Linear Component
        linear_term = x @ self.linear_weights + self.linear_bias
        
        # Periodic Component
        periodic_term = torch.sin(x @ self.periodic_weights + self.periodic_bias)

        # NonLinear Purse Periodic Component
        nonlinear_term = torch.sign(torch.sin(x @ self.nonlinear_weights + self.nonlinear_bias))
        
        # Combine All Components
        return torch.cat([linear_term, periodic_term, nonlinear_term], dim=-1)


# Time2Vec 모델 초기화
in_dim = 1
em_dim = 7
time2vec = Time2Vec(input_dim=in_dim, embed_dim=em_dim)

# 월요일(0)부터 일요일(6)까지 7일을 24*60분 단위로 나눈 시간 데이터 생성
time_data = np.linspace(0, 7, 7 * 60*24, endpoint=False).reshape(-1, 1)
time_data_tensor = torch.tensor(time_data, dtype=torch.float32)

# Time2Vec 모델을 통해 시간 데이터를 임베딩
embedding = time2vec(time_data_tensor)
print(embedding.shape)

for ei in range(em_dim):
    plt.plot(time_data.ravel(), embedding[:,ei].ravel().detach().numpy(), label=ei)
plt.legend()
plt.show()



class TimePredictModel(nn.Module):
    def __init__(self, input_dim=1, embed_dim=5, hidden_dim=32, output_dim=1):
        super().__init__()
        self.time2vec = Time2Vec(input_dim, embed_dim)

        self.fc_block = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        time_embed = self.time2vec(x)
        output = self.fc_block(time_embed)
        return output



# 데이터 생성: 주기 함수를 예시로 사용 ##################################################
def sine_wave(x):
    return np.sin(2 * np.pi * x)

# 데이터 생성: 사각파 함수를 예시로 사용
def square_wave(x, period=1.0):
    return np.sign(np.sin(2 * np.pi * x / period))

# plt.plot(periodic_improve(time_data*24*60))

# time_data_x = np.linspace(0, 7, 7 * 60*24, endpoint=False)
time_data_x = np.linspace(0, 7, 7 * 24*6, endpoint=False)

# period_y = sine_wave(time_data_x)
# period_y = square_wave(time_data_x)
period_y = periodic_improve(time_data_x*60*24)

train_x = torch.tensor(time_data_x.astype(np.float32).reshape(-1, 1))
train_y = torch.tensor(period_y.astype(np.float32).reshape(-1,1))

plt.figure()
plt.plot(time_data_x, period_y)
plt.show()

# Dataset and DataLoader
batch_size = 64
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

##################################################################################

import httpimport
remote_library_url = 'https://raw.githubusercontent.com/kimds929/'

with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
    from DS_DeepLearning import EarlyStopping

with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML


# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 모델 초기화
model = TimePredictModel(input_dim=1, embed_dim=5, hidden_dim=32, output_dim=1) 


# loss_mse = nn.MSELoss()
def mse_loss(model, x, y):
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    return loss

optimizer = optim.Adam(model.parameters(), lr=1e-2)


tm = TorchModeling(model=model, device=device)
tm.compile(optimizer=optimizer
            , loss_function = mse_loss
            , scheduler = scheduler_pathtime
            , early_stop_loss = EarlyStopping(patience=5)
            )
tm.train_model(train_loader=train_loader, epochs=100, display_earlystop_result=True, early_stop=False)
# tm.test_model(test_loader=test_loader)
# tm.recompile(optimizer=optim.Adam(model.parameters(), lr=1e-4))


with torch.no_grad():
    model.eval()
    pred = model(train_x.to(device))
    pred_arr = pred.to('cpu').numpy()

plt.figure(figsize=(15,3))
plt.plot(train_x.ravel(), pred_arr.ravel(), alpha=0.5, label='pred')
plt.plot(train_x.ravel(), train_y.ravel(), alpha=0.5, label='true')
plt.legend(loc='upper right')
plt.show()








import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Time2Vec 클래스 정의
class Time2Vec(nn.Module):
    def __init__(self, in_features, out_features):
        super(Time2Vec, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 선형 성분에 대한 가중치
        self.linear_weights = nn.Parameter(torch.randn(in_features, 1))
        self.linear_bias = nn.Parameter(torch.randn(in_features, 1))
        
        # 주기 성분에 대한 가중치
        self.periodic_weights = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.periodic_bias = nn.Parameter(torch.randn(in_features, out_features - 1))

    def forward(self, x):
        # 선형 성분
        linear_term = x @ self.linear_weights + self.linear_bias
        
        # 주기 성분
        periodic_term = torch.sin(x @ self.periodic_weights + self.periodic_bias)
        
        # 선형 성분과 주기 성분을 결합
        return torch.cat([linear_term, periodic_term], dim=-1)

# Regression 모델 정의
class RegressionModel(nn.Module):
    def __init__(self, time2vec_output_dim):
        super(RegressionModel, self).__init__()
        Time2Vec()

        self.fc_block = nn.Sequential(

        )
        self.fc1 = nn.Linear(time2vec_output_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 데이터 생성: 주기 함수를 예시로 사용 (예: 사인 함수)
def target_function(x):
    return np.sin(2 * np.pi * x)

# 시간 데이터를 월요일(0)부터 일요일(6)까지 24분 단위로 생성
time_data = np.linspace(0, 7, 7 * 24, endpoint=False).reshape(-1, 1)
target_data = target_function(time_data).reshape(-1, 1)

# 텐서로 변환
time_data_tensor = torch.tensor(time_data, dtype=torch.float32)
target_data_tensor = torch.tensor(target_data, dtype=torch.float32)

# 모델 초기화
time2vec = Time2Vec(in_features=1, out_features=3)  # in_features=1, out_features=3 (임베딩 차원)
regression_model = RegressionModel(time2vec_output_dim=3)

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(list(time2vec.parameters()) + list(regression_model.parameters()), lr=0.01)


# 학습 과정
num_epochs = 1000
for epoch in range(num_epochs):
    # Time2Vec 임베딩 생성
    time_embedding = time2vec(time_data_tensor)
    
    # 회귀 모델로 예측
    predictions = regression_model(time_embedding)
    
    # 손실 계산
    loss = criterion(predictions, target_data_tensor)
    
    # 역전파 및 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 학습 과정 출력
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 학습된 모델로 예측
time_embedding = time2vec(time_data_tensor)
predictions = regression_model(time_embedding).detach().numpy()

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(time_data, target_data, label='True Function', color='blue')
plt.plot(time_data, predictions, label='Predicted Function', color='red', linestyle='dashed')
plt.xlabel('Time (days)')
plt.ylabel('Value')
plt.title('True vs Predicted Function')
plt.legend()
plt.show()
