import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

################################################################################################################
# 1차원 Dataset 생성1 ------------------------------------------------------------
sample_size=200
group_size = 3
mu_list = np.random.rand(group_size)*8 -4
sigma_list = np.random.rand(group_size)*0.5

data_sample = []
for mu, sigma in zip(mu_list, sigma_list):
    data_sample.append(np.random.normal(loc=mu, scale=sigma, size=sample_size))

plt.hist(data_sample, bins=30)
plt.show()

data_np_cat = np.stack(data_sample).reshape(-1,1)
samples_X = torch.FloatTensor(data_np_cat)

# visualize
plt.hist(samples_X.ravel(), bins=50, color='skyblue', edgecolor='gray', alpha=0.5)
plt.show()



# 1차원 Dataset 생성2 ------------------------------------------------------------
# 1. True Distribution (Mixture of Gaussians)
def get_true_samples(n):
    # -3과 3에 봉우리가 있는 혼합 분포
    m = torch.distributions.Categorical(torch.tensor([0.5, 0.5]))
    comps = torch.distributions.Normal(torch.tensor([-3.0, 3.0]), torch.tensor([1.0, 1.0]))
    data = torch.distributions.MixtureSameFamily(m, comps)
    return data.sample((n, 1))

# 실제 PDF 계산 (시각화용)
def true_pdf(x):
    return 0.5 * (1/np.sqrt(2*np.pi)) * (np.exp(-0.5*(x+3)**2) + np.exp(-0.5*(x-3)**2))

# dataset
datasize = 500
samples_X = get_true_samples(datasize)

samples_truepdf = true_pdf(samples_X)
idx = torch.argsort(samples_X, axis=0).ravel()


# visualize
plt.hist(samples_X.ravel(), bins=50, color='skyblue', edgecolor='gray', alpha=0.5)
ax = plt.twinx()
ax.plot(samples_X.ravel()[idx], samples_truepdf.ravel()[idx])
plt.show()


# Dataset & DataLoader #################################################################
from torch.utils.data import TensorDataset, DataLoader

data_torch = torch.FloatTensor(samples_X)
dataset = TensorDataset(data_torch)
train_loader = DataLoader(dataset, shuffle=True, batch_size=128)

for batch in train_loader:
    break



################################################################################################################
# 2. 모델 정의 (Energy-Based Model)
# log q(x) = -E(x) - log Z (여기서 E가 신경망)
class EnergyNet(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(), # 매끄러운 2차 미분을 위해 Tanh나 Softplus 사용
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)
# -------------------------------------------------------------------------------------------------------------- 

# 1D 수치 적분을 통한 Partition Function Z 계산 (MLE용)
def get_log_Z(model):
    x_range = torch.linspace(-10, 10, 500).view(-1, 1).to(next(model.parameters()).device)
    energy = model(x_range)
    Z = torch.trapz(torch.exp(-energy).flatten(), x_range.flatten())    # datapoint로 적분을 근사 torch.trapz(y, x)
    return torch.log(Z)

# KL Divergence (MLE) Loss
def kl_loss(model, x):
    energy = model(x)
    # return energy.mean()
    log_Z = get_log_Z(model)
    # minimize -log q(x) = E(x) + log Z
    return (energy + log_Z).mean()


def get_log_score_Z(model):
    x_range = torch.linspace(-10, 10, 500).view(-1, 1).to(next(model.parameters()).device).requires_grad_(True)
    range_energy = model(x_range)
    range_score = -torch.autograd.grad(range_energy.sum(), x_range, create_graph=True)[0]
    range_term1 = 0.5 * range_score**2
    
    range_term2 = torch.autograd.grad(range_score.sum(), x_range, create_graph=True)[0]
    return -range_term1 + range_term2

# Fisher Divergence (Score Matching) Loss
def fisher_loss(model, x):
    x.requires_grad_(True)
    energy = model(x)
    
    # Score s(x) = d/dx log q(x) = -d/dx E(x)
    score = -torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
    
    # 1. 0.5 * ||score||^2
    term1 = 0.5 * score**2
    
    # 2. Laplacian (d/dx score)
    # 1차원에서는 스코어를 다시 x로 미분
    term2 = torch.autograd.grad(score.sum(), x, create_graph=True)[0]
    
    # range_term
    # range_term = get_log_score_Z(model)
    
    # log_Z = get_log_Z(model)
    return (term1 + term2).mean() #- (range_term.mean())
# -------------------------------------------------------------------------------------------------------------- 

model_kl = EnergyNet(hidden_dim=64).to(device)
model_fisher = EnergyNet(hidden_dim=64).to(device)

opt_kl = optim.Adam(model_kl.parameters(), lr=1e-4)
opt_fisher = optim.Adam(model_fisher.parameters(), lr=1e-4)

EPOCHS = 100

for i in tqdm(range(EPOCHS)):
    for batch in train_loader:
        batch_x = batch[0].to(device)
        # KL 학습
        opt_kl.zero_grad()
        l_kl = kl_loss(model_kl, batch_x)
        l_kl.backward()
        opt_kl.step()
        
        # Fisher 학습
        opt_fisher.zero_grad()
        l_f = fisher_loss(model_fisher, batch_x)
        l_f.backward()
        opt_fisher.step()

# --- 시각화 ---
x_test = torch.linspace(-8, 8, 200).view(-1, 1).to(device)
with torch.no_grad():
    # KL PDF 계산
    pdf_kl = torch.exp(-model_kl(x_test) - get_log_Z(model_kl)).cpu().numpy()
    # Fisher PDF 계산
    pdf_fisher = torch.exp(-model_fisher(x_test) - get_log_Z(model_fisher)).cpu().numpy()

plt.figure(figsize=(10, 5))
plt.title("Density Estimation: KL vs Fisher Divergence")
plt.hist(samples_X.ravel(), bins=50, color='skyblue', edgecolor='gray', alpha=0.5)
ax = plt.twinx()
# ax.plot(x_test.cpu(), true_pdf(x_test.cpu()), 'k--', label='True Distribution', alpha=0.5)
ax.plot(x_test.cpu(), pdf_kl, 'r-', label='Learned via KL (MLE)')
ax.plot(x_test.cpu(), pdf_fisher, 'b-', label='Learned via Fisher (Score Matching)')
# ax.fill_between(x_test.cpu().flatten(), 0, true_pdf(x_test.cpu()).flatten(), color='gray', alpha=0.1)
ax.legend()
plt.show()

