
################################################################################################
# Generate Dataset ##########################################################################
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg') 
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

# Example data (replace this with your actual data)
batch_size = 64
input_dim_init = 1    # Dimension of input data
hidden_dim = 10   # Number of hidden units
output_dim = 1    # Dimension of latent space

# (sample data) x_train / y_train --------------------------------------------------
class UnknownFuncion():
    def __init__(self, n_polynorm=1, tneta_scale=1, error_scale=None, normalize=True):
        self.n_polynorm = n_polynorm
        self.normalize = normalize
        self.y_mu = None
        self.y_std = None

        self.true_theta = torch.randn((self.n_polynorm+1,1)) * tneta_scale
        self.error_scale = torch.rand((1,))*0.3+0.1 if error_scale is None else error_scale
    
    def normalize_setting(self, train_x):
        if (self.y_mu is None) and (self.y_std is None):
            outputs = self.true_f(train_x)
            self.y_mu = torch.mean(outputs)
            self.y_std = torch.std(outputs)

    def true_f(self, x):
        for i in range(self.n_polynorm+1):
            response = self.true_theta[i] * (x**i)

            if (self.normalize) and (self.y_mu is not None) and (self.y_std is not None):
                response = (response - self.y_mu)/self.y_std
        return response

    def forward(self, x):
        if (self.normalize) and (self.y_mu is not None) and (self.y_std is not None):
            return self.true_f(x) + self.error_scale * torch.randn((x.shape[0],1))
        else:
            return self.true_f(x) + self.true_f(x).mean()*self.error_scale * torch.randn((x.shape[0],1))

    def __call__(self, x):
        return self.forward(x)

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# scale = 1
# shift = 0
# error_scale = 0.3
scale = torch.randint(0,100 ,size=(1,))
shift = torch.randint(0,500 ,size=(1,))
# error_scale = torch.rand((1,))*0.3+0.1

x_train = torch.randn(1000, input_dim_init) *scale + shift   # 1000 samples of dimension 10
x_train_add_const = torch.cat([x_train, torch.ones_like(x_train)], axis=1)
input_dim = x_train_add_const.shape[1]

# f = UnknownFuncion()
# f = UnknownFuncion(n_polynorm=2, normalize=True)
# f = UnknownFuncion(n_polynorm=3)
# f = UnknownFuncion(n_polynorm=4)
f = UnknownFuncion(n_polynorm=5)
# f = UnknownFuncion(n_polynorm=6)
# f = UnknownFuncion(n_polynorm=7)
# f = UnknownFuncion(n_polynorm=8)
# f = UnknownFuncion(n_polynorm=9)
f.normalize_setting(x_train)

y_true = f.true_f(x_train)
y_train = f(x_train)
error_sigma = (y_train - y_true).std()
# true_theta = torch.randn((input_dim,1))
# y_true = x_train_add_const @ true_theta
# y_train = y_true + error_scale*scale*torch.randn((x_train_add_const.shape[0],1))


# Dataset and DataLoader
dataset = TensorDataset(x_train_add_const, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# visualize
x_lin = torch.linspace(x_train.min(),x_train.max(),300).reshape(-1,1)
x_lin_add_const = torch.concat([x_lin, torch.ones_like(x_lin)], axis=1)

plt.figure()
plt.scatter(x_train, y_train, label='obs')
plt.plot(x_lin, f.true_f(x_lin), color='orange', label='true')
plt.legend()
plt.show()

# -----------------------------------------------------------------------------------------------
# (NOTE) ★ model is the most powerful for performance as well as easy to learn
#  1. BNN_DirectEnsemble2
#  2. BNN_Model_2
# -----------------------------------------------------------------------------------------------




################################################################################################
# DirectlyInferenceNN ##########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class BNN_Direct(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.direct_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, output_dim*2)
        )   

        self.output_dim = output_dim

    def forward(self, x):
        # Predict mu and log(sigma^2)
        mu_logvar = self.direct_block(x)
        mu, logvar = torch.split(mu_logvar, self.output_dim, dim=1)
        logvar = torch.clamp(logvar, min=-5, max=10) 
        
        return mu, logvar


# -----------------------------------------------------------------------------------------------
class BNN_DirectEnsemble1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_models=10):
        super().__init__()
        self.models = nn.ModuleList([BNN_Direct(input_dim, hidden_dim, output_dim) for _ in range(n_models)])

    def forward(self, x):
        ensemble_outputs = torch.stack([torch.stack(model(x)) for model in self.models])
        mu, logvar = torch.mean(ensemble_outputs, dim=0)
        # logvar = torch.clamp(logvar, min=-5, max=10) 
        return mu, logvar

# ★Mu/Var Ensemble only last layer
class BNN_DirectEnsemble2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_models=10):
        super().__init__()
        self.basic_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, output_dim*2*n_models)
        )
        self.n_models = n_models

        self.output_dim = output_dim

    # train step
    def train_forward(self, x):
        mu_logvar = self.basic_block(x)
        mu, logvar = torch.split(mu_logvar, self.n_models, dim=1)
        logvar = torch.clamp(logvar, min=-10, max=20) 
        return mu, logvar

    # eval step : 여러 번 샘플링하여 평균과 분산 계산
    def predict(self, x, idx=None):
        mu, logvar = self.train_forward(x)
        if idx is None:
            mu_mean = torch.mean(mu, dim=1, keepdims=True)
            logvar_mean = torch.mean(logvar, dim=1, keepdims=True)
        else:
            mu_mean = torch.mean(mu[:, idx], dim=1, keepdims=True)
            logvar_mean = torch.mean(logvar[:, idx], dim=1, keepdims=True)
        return  mu_mean, logvar_mean

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x)

################################################################################################

# Initialize the model, optimizer
# model = BNN_Direct(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
# model = BNN_DirectEnsemble1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10)
model = BNN_DirectEnsemble2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=5)
model.to(device)

# -----------------------------------------------------------------------------------------------
# Hyperparameters
learning_rate = 1e-3
beta = 0.5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_gaussian = nn.GaussianNLLLoss()
# mu_loss_weight = 0.5  # mu에 대한 손실 가중치
# logvar_loss_weight = 0.1  # log_var에 대한 손실 가중치

num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        
        # Forward pass ----------------------------------------
        mu, logvar = model(batch_x)
        std = torch.exp(0.5*logvar)
        
        # (Loss Parameters) ----------------------------------------
        # # fixed variance : # Reparameterization trick: z = mu + sigma * epsilon
        # z = mu + std * torch.randn_like(std)    
        # loss = torch.sum( (z - batch_y)**2)

        # # variational variance : neg_log_likelihood
        # loss = torch.sum(0.5 * torch.log(std**2) + 0.5 * (1/std**2) * (batch_y - mu) ** 2)
        # loss = -torch.distributions.Normal(mu, std).log_prob(batch_y).sum()
        loss = loss_gaussian(mu, batch_y, std**2)

        # # weigted variantional variance
        # gaussian_loss = torch.exp(-logvar) * (batch_y - mu)**2 + logvar
        # loss = mu_loss_weight * torch.mean((batch_y - mu)**2) + logvar_loss_weight * torch.mean(gaussian_loss)

        
        # (KL-Divergence Loss) For anchoring response range \hat{y} ~ N(0,1) 
        # loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # loss_kl = 0.5 * torch.sum(mu**2 + std**2 - torch.log(std**2) - 1)
        
        # Compute loss ----------------------------------------
        # loss = loss + beta * loss_kl
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
   
    if epoch % 25 ==0:
        with torch.no_grad():
            model.eval()
            eval_mu, eval_logvar = model(x_train_add_const.to(device))
            eval_mu, eval_logvar = eval_mu.to('cpu'), eval_logvar.to('cpu')
            eval_std = torch.exp(0.5*eval_logvar)
            residual_sigma = (eval_mu - y_train).std()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}, mu: {torch.mean(mu):.2f}, resid_std: {residual_sigma:.2f}')
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}, Loss_recon: {loss_recon:.2f}, Loss_KL: {loss_kl:.2f}')
# -----------------------------------------------------------------------------------------------


# visualize
x_lin = torch.linspace(x_train.min(),x_train.max(),300).reshape(-1,1)
x_lin_add_const = torch.concat([x_lin, torch.ones_like(x_lin)], axis=1)

with torch.no_grad():
    model.eval()
    y_mu, y_logvar = model(x_lin_add_const.to(device))
    y_mu = y_mu.to('cpu')
    y_std = torch.exp(0.5*y_logvar).to('cpu')

plt.figure()
plt.scatter(x_train, y_train, label='obs', alpha=0.5)
plt.plot(x_lin, f.true_f(x_lin), color='orange', label='true', alpha=0.5)

plt.plot(x_lin, y_mu, alpha=0.5, color='mediumseagreen', label='pred_mu')
plt.fill_between(x_lin.flatten(), (y_mu-y_std).flatten(), (y_mu+y_std).flatten(), color='mediumseagreen', alpha=0.2, label='pred_var')
plt.legend()
plt.show()

# valid std
with torch.no_grad():
    model.eval()
    eval_mu, eval_logvar = model(x_train_add_const.to(device))
    eval_mu, eval_logvar = eval_mu.to('cpu'), eval_logvar.to('cpu')
    eval_std = torch.exp(0.5*eval_logvar)
    residual_sigma = (eval_mu - y_train).std()
    print(f"residual_sigma : {residual_sigma:.3f}", end =" ")
print(f"/ error_sigma :{error_sigma:.3f}")
# print(f"/ error_sigma :{f.error_scale:.3f}")










################################################################################################
# BayesianNN (WeightEnsemble) ##################################################################


# ---------------------------------------------------------------------------------------
# pip install torchbnn
import torch.nn as nn
import torchbnn as bnn

# BNN model : ensemble only when evaluation
class BNN_Weight_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_models=10):
        super().__init__()
        self.bayes_block = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_dim, out_features=hidden_dim)
            ,nn.ReLU()
            ,bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=hidden_dim)
            ,nn.ReLU()
            ,bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=hidden_dim)
            ,nn.ReLU()
            ,bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=output_dim)
        )

        self.n_models = n_models
    
    # train step
    def train_forward(self, x):
        x = self.bayes_block(x)
        return x, None

    # eval step : 여러 번 샘플링하여 평균과 분산 계산
    def predict(self, x, n_models=None):
        n_models = self.n_models if n_models is None else n_models

        ensemble_outputs = torch.stack([self.train_forward(x)[0] for _ in range(n_models)])
        mu = torch.mean(ensemble_outputs, dim=0)
        var = torch.var(ensemble_outputs, dim=0)
        logvar = torch.log(var)
        return mu, logvar

    def forward(self, x, n_models=None):
        n_models = self.n_models if n_models is None else n_models
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x, n_models)

# ---------------------------------------------------------------------------------------
# pip install torchbnn
import torch.nn as nn
import torchbnn as bnn

# BNN model : ensemble when both training & evaluation
class BNN_Weight_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_models=10):
        super().__init__()
        self.sub_bnn = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_dim, out_features=hidden_dim)
            ,nn.ReLU()
            ,bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=hidden_dim)
            ,nn.ReLU()
            ,bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=hidden_dim)
            ,nn.ReLU()
            ,bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=output_dim)
        )

        self.n_models = n_models
    
    def forward(self, x):
        ensemble_outputs = torch.stack([self.sub_bnn(x) for _ in range(self.n_models)])
        mu = torch.mean(ensemble_outputs, dim=0)
        var = torch.var(ensemble_outputs, dim=0)
        logvar = torch.log(var)
        return mu, logvar






#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
import torch
import torch.nn as nn

# Bayesian Linear Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight mean and log variance
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(mean=0,std=0.1))
        # self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).normal_(mean=0,std=0.1))
        # self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        
        # Bias mean and log variance
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(mean=0,std=0.1))
        # self.bias_logvar = nn.Parameter(torch.Tensor(out_features).normal_(mean=0,std=0.1))
        # self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_mu)
        bound = 1 / torch.sqrt(torch.tensor(self.in_features + self.out_features, dtype=torch.float32))
        nn.init.uniform_(self.bias_mu, -bound, bound)
        # nn.init.xavier_uniform_(self.bias_mu)

    def forward(self, x):
        # Reparameterization trick
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        
        # Sample from normal distribution
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
               
        weight = self.weight_mu + weight_std * weight_eps   # re-parameterization_trick
        bias = self.bias_mu + bias_std * bias_eps      # re-parameterization_trick
        return x @ weight.T + bias



# BNN model : ensemble only when evaluation
class BNN_Weight_3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_models=10):
        super().__init__()
        self.bayes_block = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim)
            ,nn.ReLU()
            ,BayesianLinear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,BayesianLinear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,BayesianLinear(hidden_dim, output_dim)
            )
        self.n_models = n_models
    
    # train step
    def train_forward(self, x):
        x = self.bayes_block(x)
        return x, None

    # eval step : 여러 번 샘플링하여 평균과 분산 계산
    def predict(self, x, n_models=None):
        n_models = self.n_models if n_models is None else n_models

        ensemble_outputs = torch.stack([self.train_forward(x)[0] for _ in range(n_models)])
        mu = torch.mean(ensemble_outputs, dim=0)
        var = torch.var(ensemble_outputs, dim=0)
        logvar = torch.log(var)
        return mu, logvar

    def forward(self, x, n_models=None):
        n_models = self.n_models if n_models is None else n_models
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x, n_models)

# ---------------------------------------------------------------------------------------
import torch
import torch.nn as nn

# BNN model : ensemble when both training & evaluation
class BNN_Weight_4(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_models=10):
        super().__init__()
        self.sub_bnn = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim)
            ,nn.ReLU()
            ,BayesianLinear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,BayesianLinear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,BayesianLinear(hidden_dim, output_dim)
        )
        self.n_models = n_models

    def forward(self, x):
        ensemble_outputs = torch.stack([self.sub_bnn(x) for _ in range(self.n_models)])
        mu = torch.mean(ensemble_outputs, dim=0)
        var = torch.var(ensemble_outputs, dim=0)
        logvar = torch.log(var)
        return mu, logvar


################################################################################################

# Initialize the model, optimizer
# model = BNN_Weight_1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10)
model = BNN_Weight_2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10)
# model = BNN_Weight_3(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10)
# model = BNN_Weight_4(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10)
model.to(device)


# -----------------------------------------------------------------------------------------------
# Hyperparameters
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_mse = nn.MSELoss()     # for BNN_Weight_1, BNN_Weight_3
loss_gaussian = nn.GaussianNLLLoss()    # for BNN_Weight_2, BNN_Weight_4

num_epochs = 300

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        
        # Forward pass ----------------------------------------
        mu, logvar = model(batch_x)
        if logvar is not None:
            std = torch.exp(0.5*logvar)

        # Compute loss ----------------------------------------
        if logvar is None:
            # # (Not assumption of error distribution) nn.MSELoss()
            # loss = torch.sum( (batch_y - mu)**2)
            loss = loss_mse(mu, batch_y)
        
        else:
            # # (Assumption of error distribution as Gaussian : Negative Log-Likelihood) nn.GaussianNLLLoss()
            # loss = torch.sum(0.5 * torch.log(std**2) + 0.5 * (1/std**2) * (batch_y - mu) ** 2)
            # loss = -torch.distributions.Normal(mu, std).log_prob(batch_y).sum()
            loss = loss_gaussian(mu, batch_y, std**2)
        
        # Backward pass and optimize --------------------------
        loss.backward()
        optimizer.step()

    if epoch % 25 ==0:
        with torch.no_grad():
            model.eval()
            eval_mu, eval_logvar = model(x_train_add_const.to(device))
            eval_mu, eval_logvar = eval_mu.to('cpu'), eval_logvar.to('cpu')
            eval_std = torch.exp(0.5*eval_logvar)
            residual_sigma = (eval_mu - y_train).std()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}, mu: {torch.mean(mu):.2f}, resid_std: {residual_sigma:.2f}')
# --------------------------------------------------------------------------------------------------


# visualize
x_lin = torch.linspace(x_train.min(),x_train.max(),300).reshape(-1,1)
x_lin_add_const = torch.concat([x_lin, torch.ones_like(x_lin)], axis=1)

with torch.no_grad():
    model.eval()
    y_mu, y_logvar = model(x_lin_add_const.to(device))
    y_mu = y_mu.to('cpu')
    y_std = torch.exp(0.5*y_logvar).to('cpu')

plt.figure()
plt.scatter(x_train, y_train, label='obs', alpha=0.5)
plt.plot(x_lin, f.true_f(x_lin), color='orange', label='true', alpha=0.5)

plt.plot(x_lin, y_mu, alpha=0.5, color='mediumseagreen', label='pred_mu')
plt.fill_between(x_lin.flatten(), (y_mu-y_std).flatten(), (y_mu+y_std).flatten(), color='mediumseagreen', alpha=0.2, label='pred_var')
plt.legend()
plt.show()

# valid std
with torch.no_grad():
    model.eval()
    eval_mu, eval_logvar = model(x_train_add_const.to(device))
    eval_mu, eval_logvar = eval_mu.to('cpu'), eval_logvar.to('cpu')
    eval_std = torch.exp(0.5*eval_logvar)
    residual_sigma = (eval_mu - y_train).std()
    print(f"residual_sigma : {residual_sigma:.3f}", end =" ")
print(f"/ error_sigma :{error_sigma:.3f}")
# print(f"/ error_sigma :{f.error_scale:.3f}")

















####################################################################################################
# ModelEnsemble Based ##############################################################################
# Base Network
class BasicNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.basic_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.basic_block(x)

# Ensemble Model
class BNN_Model_1(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, n_models=10):
        super().__init__()
        hidden_dims = torch.randint(low=8,high=128,size=(n_models,)) if hidden_dim is None else torch.ones((n_models,)).type(torch.int)*n_models

        self.models = nn.ModuleList([BasicNetwork(input_dim, int(h_dim), output_dim) for _, h_dim in zip(range(n_models), hidden_dims)])
    
    def forward(self, x):
        ensemble_outputs = torch.stack([model(x) for model in self.models])
        mu = torch.mean(ensemble_outputs, dim=0)
        var = torch.var(ensemble_outputs, dim=0)
        logvar = torch.log(var)
        return mu, logvar

# ★ Sample Ensemble only last layers Model
class BNN_Model_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_models=10):
        super().__init__()
        self.basic_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, output_dim*n_models)
        )
    
    def forward(self, x):
        ensemble_outputs = self.basic_block(x)
        # return ensemble_outputs
        mu = torch.mean(ensemble_outputs, dim=1, keepdims=True)
        var = torch.var(ensemble_outputs, dim=1, keepdims=True)
        logvar = torch.log(var)
        return mu, logvar

################################################################################################
# model = BNN_Model_1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10)    # initial weight ensemble
# model = BNN_Model_1(input_dim=input_dim, hidden_dim=None, output_dim=output_dim, n_models=10)    # architecture ensemble
model = BNN_Model_2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10)    # initial weight ensemble
model.to(device)
model



# -----------------------------------------------------------------------------------------------
# Hyperparameters
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_mse = nn.MSELoss()
loss_gaussian = nn.GaussianNLLLoss()

num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        
        # Forward pass ----------------------------------------
        mu, logvar = model(batch_x)
        std = torch.exp(0.5*logvar)

        # Compute loss ----------------------------------------
        # # (Not assumption of error distribution) nn.MSELoss()
        # loss = torch.sum( (batch_y - mu)**2)
        # loss = loss_mse(mu, batch_y)

        # # (Assumption of error distribution as Gaussian : Negative Log-Likelihood) nn.GaussianNLLLoss()
        # loss = torch.sum(0.5 * torch.log(std**2) + 0.5 * (1/std**2) * (batch_y - mu) ** 2)
        # loss = -torch.distributions.Normal(mu, std).log_prob(batch_y).sum()
        loss = loss_gaussian(mu, batch_y, std**2)
        
        # Backward pass and optimize --------------------------
        loss.backward()
        optimizer.step()

    if epoch % 25 ==0:
        with torch.no_grad():
            model.eval()
            eval_mu, eval_logvar = model(x_train_add_const.to(device))
            eval_mu, eval_logvar = eval_mu.to('cpu'), eval_logvar.to('cpu')
            eval_std = torch.exp(0.5*eval_logvar)
            residual_sigma = (eval_mu - y_train).std()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}, mu: {torch.mean(mu):.2f}, resid_std: {residual_sigma:.2f}')
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}, Loss_recon: {loss_recon:.2f}, Loss_KL: {loss_kl:.2f}')
# -----------------------------------------------------------------------------------------------


# visualize
x_lin = torch.linspace(x_train.min(),x_train.max(),300).reshape(-1,1)
x_lin_add_const = torch.concat([x_lin, torch.ones_like(x_lin)], axis=1)

with torch.no_grad():
    model.eval()
    y_mu, y_logvar = model(x_lin_add_const.to(device))
    y_mu = y_mu.to('cpu')
    y_std = torch.exp(0.5*y_logvar).to('cpu')

plt.figure()
plt.scatter(x_train, y_train, label='obs', alpha=0.5)
plt.plot(x_lin, f.true_f(x_lin), color='orange', label='true', alpha=0.5)

plt.plot(x_lin, y_mu, alpha=0.5, color='mediumseagreen', label='pred_mu')
plt.fill_between(x_lin.flatten(), (y_mu-y_std).flatten(), (y_mu+y_std).flatten(), color='mediumseagreen', alpha=0.2, label='pred_var')
plt.legend()
plt.show()

# valid std
with torch.no_grad():
    model.eval()
    eval_mu, eval_logvar = model(x_train_add_const.to(device))
    eval_mu, eval_logvar = eval_mu.to('cpu'), eval_logvar.to('cpu')
    eval_std = torch.exp(0.5*eval_logvar)
    residual_sigma = (eval_mu - y_train).std()
    print(f"residual_sigma : {residual_sigma:.3f}", end =" ")
print(f"/ error_sigma :{error_sigma:.3f}")
# print(f"/ error_sigma :{f.error_scale:.3f}")





























###############################################################################################
# Full Ensemble  ##############################################################################

# Bayesian Linear Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight mean and log variance
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(mean=0,std=0.1))
        # self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).normal_(mean=0,std=0.1))
        # self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        
        # Bias mean and log variance
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(mean=0,std=0.1))
        # self.bias_logvar = nn.Parameter(torch.Tensor(out_features).normal_(mean=0,std=0.1))
        # self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_mu)
        bound = 1 / torch.sqrt(torch.tensor(self.in_features + self.out_features, dtype=torch.float32))
        nn.init.uniform_(self.bias_mu, -bound, bound)
        # nn.init.xavier_uniform_(self.bias_mu)

    def forward(self, x):
        # Reparameterization trick
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        
        # Sample from normal distribution
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        
        if self.training:
            weight = self.weight_mu + weight_std * weight_eps   # re-parameterization_trick
            bias = self.bias_mu + bias_std * bias_eps      # re-parameterization_trick
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return x @ weight.T + bias



# Base Network
class BayesianBasicNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.bayes_block = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim)
            # bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_dim, out_features=hidden_dim)
            ,nn.ReLU()
            ,BayesianLinear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,BayesianLinear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,BayesianLinear(hidden_dim, output_dim)
            # ,bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=output_dim)
        )
    
    def forward(self, x):
        return self.bayes_block(x)

# Ensemble Model
class BNN_FullEnsemble(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, n_models=10):
        super().__init__()
        hidden_dims = torch.randint(low=8,high=128,size=(n_models,)) if hidden_dim is None else torch.ones((n_models,)).type(torch.int)*n_models
        self.models = nn.ModuleList([BayesianBasicNetwork(input_dim, int(h_dim), output_dim) for _, h_dim in zip(range(n_models), hidden_dims)])
    
    def forward(self, x):
        ensemble_outputs = torch.stack([model(x) for model in self.models])
        mu = torch.mean(ensemble_outputs, dim=0)
        var = torch.var(ensemble_outputs, dim=0)
        logvar = torch.log(var)
        return mu, logvar

################################################################################################
model = BNN_FullEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10)    # initial weight ensemble
# model = BNN_FullEnsemble(input_dim=input_dim, hidden_dim=None, output_dim=output_dim, n_models=10)    # architecture ensemble
model.to(device)
# model


# -----------------------------------------------------------------------------------------------
# Hyperparameters
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_mse = nn.MSELoss()
loss_gaussian = nn.GaussianNLLLoss()

num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        
        # Forward pass ----------------------------------------
        mu, logvar = model(batch_x)
        std = torch.exp(0.5*logvar)

        # Compute loss ----------------------------------------
        # # (Not assumption of error distribution) nn.MSELoss()
        # loss = torch.sum( (batch_y - mu)**2)
        # loss = loss_mse(mu, batch_y)

        # # (Assumption of error distribution as Gaussian : Negative Log-Likelihood) nn.GaussianNLLLoss()
        # loss = torch.sum(0.5 * torch.log(std**2) + 0.5 * (1/std**2) * (batch_y - mu) ** 2)
        # loss = -torch.distributions.Normal(mu, std).log_prob(batch_y).sum()
        loss = loss_gaussian(mu, batch_y, std**2)
        
        # Backward pass and optimize --------------------------
        loss.backward()
        optimizer.step()

    if epoch % 25 ==0:
        with torch.no_grad():
            model.eval()
            eval_mu, eval_logvar = model(x_train_add_const.to(device))
            eval_mu, eval_logvar = eval_mu.to('cpu'), eval_logvar.to('cpu')
            eval_std = torch.exp(0.5*eval_logvar)
            residual_sigma = (eval_mu - y_train).std()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}, mu: {torch.mean(mu):.2f}, resid_std: {residual_sigma:.2f}')
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}, Loss_recon: {loss_recon:.2f}, Loss_KL: {loss_kl:.2f}')
# -----------------------------------------------------------------------------------------------


# visualize
x_lin = torch.linspace(x_train.min(),x_train.max(),300).reshape(-1,1)
x_lin_add_const = torch.concat([x_lin, torch.ones_like(x_lin)], axis=1)

with torch.no_grad():
    model.eval()
    y_mu, y_logvar = model(x_lin_add_const.to(device))
    y_mu = y_mu.to('cpu')
    y_std = torch.exp(0.5*y_logvar).to('cpu')

plt.figure()
plt.scatter(x_train, y_train, label='obs', alpha=0.5)
plt.plot(x_lin, f.true_f(x_lin), color='orange', label='true', alpha=0.5)

plt.plot(x_lin, y_mu, alpha=0.5, color='mediumseagreen', label='pred_mu')
plt.fill_between(x_lin.flatten(), (y_mu-y_std).flatten(), (y_mu+y_std).flatten(), color='mediumseagreen', alpha=0.2, label='pred_var')
plt.legend()
plt.show()

# valid std
with torch.no_grad():
    model.eval()
    eval_mu, eval_logvar = model(x_train_add_const.to(device))
    eval_mu, eval_logvar = eval_mu.to('cpu'), eval_logvar.to('cpu')
    eval_std = torch.exp(0.5*eval_logvar)
    residual_sigma = (eval_mu - y_train).std()
    print(f"residual_sigma : {residual_sigma:.3f}", end =" ")
print(f"/ error_sigma :{error_sigma:.3f}")
# print(f"/ error_sigma :{f.error_scale:.3f}")