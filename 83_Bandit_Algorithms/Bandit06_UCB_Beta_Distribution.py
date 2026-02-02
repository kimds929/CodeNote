import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.stats import beta





########################################################################################################
# (Beta Distribution)

alpha_ = 20
beta_ = 20
beta.ppf(1-0.05, alpha_, beta_) - beta.ppf(0.05, alpha_, beta_)
(beta.ppf(1-0.05, alpha_, beta_) + beta.ppf(0.05, alpha_, beta_))/2


def plot_beta_distributions(alpha_, beta_, color=None):
    x = np.linspace(0, 1, 100)
    
    # plt.figure(figsize=(8, 5))
    
    y = beta.pdf(x, alpha_, beta_)
    plt.plot(x, y, label=f'alpha={alpha_}, beta={beta_}', color=color)
    
    plt.title('Beta Distributions')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()


alpha_list = [1, 4, 5, 10]
beta_list = [1, 2, 8, 10]


for alpha_, beta_ in zip(alpha_list, beta_list):
    plot_beta_distributions(alpha_, beta_, color=(1-alpha_/10, 1-beta_/10, 0))
plt.legend(loc='upper right', bbox_to_anchor=(1.4,1))


########################################################################################################


from IPython.display import clear_output
import time
rng = np.random.RandomState()    


p = 0.6

alpha_ = 0
beta_ = 0


x = np.linspace(0, 1, 100)
for i in range(30):
    sample_p = rng.rand()
    # np.random.binomial(n=1, p=0.5)
    
    print(int(sample_p < p))
    if sample_p < p:
        alpha_ += 1
    else:
        beta_ += 1
    y = beta.pdf(x, alpha_, beta_)
    
    plt.plot(x, y, label=f'alpha={alpha_}, beta={beta_}')
    plt.axvline(p, color='red', ls='--', alpha=0.5)
    plt.show()
    clear_output(wait=True)
    time.sleep(0.1)




###########################################################################################################
# (Beta Distributino Class)
from scipy.stats import beta

class Beta:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.beta = beta

    def add_alpha(self, inc=1):
        self.a += inc

    def add_beta(self, inc=1):
        self.b += inc

    def init_params(self, a=None, b=None):
        a = self.a if a is None else a
        b = self.b if b is None else b
        return a, b
    
    def __repr__(self):
        return f"Beta(a={self.a}, b={self.b})"

    # ----------------------------
    # sampling / density
    # ----------------------------
    def rvs(self, a=None, b=None, loc=0, scale=1, size=1, random_state=None):
        a, b = self.init_params(a, b)
        return self.beta.rvs(a=a, b=b, loc=loc, scale=scale, size=size, random_state=random_state)

    def pdf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.pdf(x, a=a, b=b, loc=loc, scale=scale)

    def logpdf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.logpdf(x, a=a, b=b, loc=loc, scale=scale)

    # ----------------------------
    # CDF / SF / inverse
    # ----------------------------
    def cdf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.cdf(x, a=a, b=b, loc=loc, scale=scale)

    def logcdf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.logcdf(x, a=a, b=b, loc=loc, scale=scale)

    def sf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.sf(x, a=a, b=b, loc=loc, scale=scale)

    def logsf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.logsf(x, a=a, b=b, loc=loc, scale=scale)

    def ppf(self, q, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.ppf(q, a=a, b=b, loc=loc, scale=scale)

    def isf(self, q, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.isf(q, a=a, b=b, loc=loc, scale=scale)

    # ----------------------------
    # moments / stats
    # ----------------------------
    def moment(self, order, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.moment(order, a=a, b=b, loc=loc, scale=scale)

    def stats(self, a=None, b=None, loc=0, scale=1, moments='mv'):
        a, b = self.init_params(a, b)
        return self.beta.stats(a=a, b=b, loc=loc, scale=scale, moments=moments)

    def entropy(self, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.entropy(a=a, b=b, loc=loc, scale=scale)

    def expect(self, func, a=None, b=None, loc=0, scale=1,
               lb=None, ub=None, conditional=False, **kwds):
        a, b = self.init_params(a, b)
        # SciPy expect는 args=(a,b) 형태로 shape를 전달
        return self.beta.expect(func, args=(a, b), loc=loc, scale=scale,
                                lb=lb, ub=ub, conditional=conditional, **kwds)

    # ----------------------------
    # point estimates
    # ----------------------------
    def median(self, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.median(a=a, b=b, loc=loc, scale=scale)

    def mean(self, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.mean(a=a, b=b, loc=loc, scale=scale)

    def var(self, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.var(a=a, b=b, loc=loc, scale=scale)

    def std(self, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.std(a=a, b=b, loc=loc, scale=scale)

    def interval(self, confidence, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.interval(confidence, a=a, b=b, loc=loc, scale=scale)

    # ----------------------------
    # fit
    # ----------------------------
    def fit(self, data, update=False, **kwargs):
        """
        SciPy beta.fit(data)는 (a, b, loc, scale)를 반환.
        - update=False: 추정치만 반환
        - update=True : self.a, self.b도 업데이트 (loc/scale은 별도 관리 안 함)
        """
        a_hat, b_hat, loc_hat, scale_hat = self.beta.fit(data, **kwargs)
        if update:
            self.a = a_hat
            self.b = b_hat
        return a_hat, b_hat, loc_hat, scale_hat




########################################################################################################
########################################################################################################
########################################################################################################
# (Environment Setting)
rng = np.random.RandomState(6)

def random_noise(scale=0.2):
    return rng.rand()*scale - scale/2


n_process = rng.randint(3,8)
n_fac_in_process = rng.randint(1, 5, size=n_process)
max_fac = max(n_fac_in_process)

# ---------------------------------------------------------------------------------------------------
# make position
def make_pos(n_fac_in_process):
    x_pos = [[np.nan] + [ei+1]*f  for ei, f in enumerate(n_fac_in_process)]
    y_pos = [[np.nan] + (np.arange(1, f+1)/(f+1) * (max_fac+2) + random_noise() ).tolist()  for f in n_fac_in_process]
    return x_pos, y_pos

x_pos, y_pos = make_pos(n_fac_in_process)
print(x_pos)
print(y_pos)



# ---------------------------------------------------------------------------------------------------
# visualize graph
def make_fac_plot(n_fac_in_process, pathes=[], label=[], abnormal=[]):
    x_pos, y_pos = make_pos(n_fac_in_process)
    
    # point
    colors20 = {ei: c for ei, c in enumerate(plt.cm.tab20.colors)}
    for x_list, y_list in zip(x_pos, y_pos):
        for yi, (x, y) in enumerate(zip(x_list, y_list)):
            if not np.isnan(x):
                plt.scatter(x, y, color = 'gray')
                # plt.scatter(x, y, color = colors20.get(x))
                plt.text(x-0.3, y+0.1, f"x{x}_y{yi}", fontsize=8, alpha=0.7)
    
    # abnormal
    for (axi, ayi, p) in abnormal:
        scatter  = plt.scatter(x_pos[axi-1][ayi], y_pos[axi-1][ayi], color=((p/2), 0, 0))
        scateter_color = scatter.get_facecolor()
        plt.text(x_pos[axi-1][ayi]+0.05, y_pos[axi-1][ayi], f"{p:.2f}", fontsize=8, color=scateter_color)
    
    # path
    for pi, path in enumerate(pathes):
        path_pos = [(x[p], y[p]) for x, y, p in zip(x_pos, y_pos, path) if not np.isnan(x[p])]
        path_pos_T = np.array(path_pos).T
        line, = plt.plot(*path_pos_T, alpha=0.5)
        line_color = line.get_color()
        
        if len(label) == len(pathes):
            plt.text(*path_pos_T[:, -1], f"MTL_{pi} ({label[pi]})", fontsize=8, color=line_color)
        else:
            plt.text(*path_pos_T[:, -1], f"MTL_{pi}", fontsize=8, color=line_color)
    plt.xlim(0.3, None)
        
# make_fac_plot(n_fac_in_process)
# plt.show()

# ---------------------------------------------------------------------------------------------------
# generate random path
def random_path_generator(n=1):
    return np.stack([[int(rng.choice(np.arange(f+1))) for f in n_fac_in_process] for _ in range(n)])

random_paths = random_path_generator(n=3)
print(random_paths)
# make_fac_plot(n_fac_in_process, random_paths)


make_fac_plot(n_fac_in_process, random_paths)
plt.show()

###################################################################################################
abnormal_p = 0.7
    
# ★ random path generate
X_random_paths = random_path_generator(n=100)
print(X_random_paths)





# ---------------------------------------------------------------------------------------------------
# # random label generate
# label = (rng.random( (len(X_random_paths),1) ) > abnormal_p).astype(int)
# print(label.flatten())


# designated label generate
def generate_abnormal_xy(n_fac_in_process, n=1):
    abnormal_ps = np.round((rng.random(size=n) /2 + 0.5), 2)
    abnormal_xyp = []
    for i, p in zip(range(n), abnormal_ps):
        abnormal_idx = rng.rand() * sum(n_fac_in_process)
        abnormal_x = np.where(np.cumsum(n_fac_in_process) > abnormal_idx)[0][0]+1
        abnormal_y = (n_fac_in_process - (np.cumsum(n_fac_in_process) - abnormal_idx)).astype(int)[abnormal_x-1]+1
        abnormal_xyp.append( (abnormal_x, abnormal_y, p))
    return abnormal_xyp
        

abnormal_xyp = generate_abnormal_xy(n_fac_in_process, n=1)
make_fac_plot(n_fac_in_process, abnormal=abnormal_xyp)
plt.show()
# ---------------------------------------------------------------------------------------------------



def generate_path_probs_and_labels(
    random_paths: np.ndarray,
    n_fac_in_process: np.ndarray,
    abnormal_xyp: list,   # [(abnormal_x, abnormal_y, p0), ...]
    base_abnormal_p: float = 0.03,
    scale: float = 100.0
):
    """
    Returns
    -------
    p_path : (N,) float
        path별 샘플링된 불량확률
    y      : (N,) int
        0=양호, 1=불량
    """
    yp_true = []
    yp_obs = []

    # 안전장치: abnormal_xyp가 공정/시설 범위 밖이면 에러 대신 무시하고 싶다면 continue로 바꿔도 됨
    for i, path in enumerate(random_paths):
        p_total = base_abnormal_p
        for (ax, ay, p0) in abnormal_xyp:
            if not (1 <= ax <= len(n_fac_in_process)):
                raise ValueError(f"abnormal_x={ax} is out of range (1..{len(n_fac_in_process)})")
            if not (1 <= ay <= n_fac_in_process[ax-1]):
                raise ValueError(f"abnormal_y={ay} is out of range (1..{n_fac_in_process[ax-1]}) for process {ax}")

            if int(path[ax-1]) == int(ay):
                p_total = float(1 - (1-p0)* (1-p_total))
        
        yp_sample = rng.beta(p_total*scale, (1-p_total)*scale)
        
        yp_true.append(p_total)
        yp_obs.append(yp_sample)

    return np.array(yp_true), np.array(yp_obs)



yp_true, yp_obs = generate_path_probs_and_labels(X_random_paths, n_fac_in_process, abnormal_xyp)

make_fac_plot(n_fac_in_process, X_random_paths[yp_true > 0.5], abnormal=abnormal_xyp)
plt.show()



# ★ y_observation (0/1)
y_obs = rng.binomial(1, yp_obs)


# visualize
idx = rng.randint(len(X_random_paths))
make_fac_plot(n_fac_in_process, X_random_paths[[idx]], label=y_obs[[idx]], abnormal=abnormal_xyp)



###########################################################################################################


process_param_alpha = [np.bincount(xg) for xg in X_random_paths[np.where(y_obs==1)[0]].T]
process_param_beta = [np.bincount(xg) for xg in X_random_paths[np.where(y_obs==0)[0]].T]


# beta distributions for each process
process_beta_dist = []
for x_alpha_, x_beta_ in zip(process_param_alpha, process_param_beta):
    beta_dist = [Beta(alpha_, beta_) for alpha_, beta_ in zip(x_alpha_, x_beta_)]
    process_beta_dist.append(beta_dist)


# [[y.mean() for y in x] for x in process_beta_dist]      # mean
# [[float(y.interval(0.90)[0]) > 0.5 for y in x] for x in process_beta_dist]    # lcb
# [[float(y.interval(0.90)[1]) < 0.5 for y in x] for x in process_beta_dist]    # ucb




















# ----------------------------------------------------------------------------------------------------------------
# beta distribution을 Deep Neural Network가 학습해서 binary classification의 uncertainty도 학습할 수 있을까? loss funciton을 beta distribution기반으로 해야할까?
# 방금 질문한 beta distribution을 Deep Neural Network가 학습해서 binary classification의 uncertainty도 학습하는 과정을 
# 간단한 예제 데이터 기반으로 torch 코드를 보여줘.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ===== 1. 예제 데이터 생성 =====
torch.manual_seed(42)
n_samples = 200
X = torch.rand(n_samples, 1) * 2 - 1  # [-1, 1] 범위
y = (X[:, 0] > 0).float()  # 0 또는 1 라벨

# ===== 2. 모델 정의 =====
class BetaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc_alpha = nn.Linear(16, 1)
        self.fc_beta = nn.Linear(16, 1)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        alpha = F.softplus(self.fc_alpha(h)) + 1e-6  # 양수 제약
        beta = F.softplus(self.fc_beta(h)) + 1e-6
        return alpha, beta

# ===== 3. Loss 함수 (Beta-Bernoulli NLL) =====
def beta_bernoulli_nll(alpha, beta, target):
    p = alpha / (alpha + beta)
    nll = - (target * torch.log(p + 1e-8) + (1 - target) * torch.log(1 - p + 1e-8))
    return nll.mean()

# ===== 4. 학습 =====
model = BetaNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    alpha_pred, beta_pred = model(X)
    loss = beta_bernoulli_nll(alpha_pred, beta_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ===== 5. 예측 및 불확실성 계산 =====
with torch.no_grad():
    alpha_pred, beta_pred = model(X)
    mean_pred = alpha_pred / (alpha_pred + beta_pred)
    var_pred = (alpha_pred * beta_pred) / ((alpha_pred + beta_pred)**2 * (alpha_pred + beta_pred + 1))
    
# 일부 샘플 출력
for i in range(5):
    print(f"X={X[i].item():.2f}, y={y[i].item()}, "
          f"mean={mean_pred[i].item():.3f}, var={var_pred[i].item():.4f}")
    
    