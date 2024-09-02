
################################################################################################
# Generate Dataset ##########################################################################
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg') 
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

example = False

if example:
    # Example data (replace this with your actual data)
    batch_size = 64
    input_dim_init = 1    # Dimension of input data
    hidden_dim = 10   # Number of hidden units
    output_dim = 1    # Dimension of latent space

# (sample data) x_train / y_train --------------------------------------------------
class UnknownFuncion():
    def __init__(self, n_polynorm=1, theta_scale=1, error_scale=None, normalize=True):
        self.n_polynorm = n_polynorm
        self.normalize = normalize
        self.y_mu = None
        self.y_std = None

        self.true_theta = torch.randn((self.n_polynorm+1,1)) * theta_scale
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

class UnknownBernoulliFunction():
    def __init__(self, n_polynorm=1, theta_scale=1, error_scale=None, normalize=True):
        self.n_polynorm = n_polynorm
        self.normalize = normalize
        self.y_mu = None
        self.y_std = None

        self.true_theta = torch.randn((self.n_polynorm+1,1)) * theta_scale
        self.error_scale = torch.rand((1,))*0.3+0.1 if error_scale is None else error_scale
    
    def normalize_setting(self, train_x):
        if (self.y_mu is None) and (self.y_std is None):
            outputs = self.true_f(train_x)
            self.y_mu = torch.mean(outputs)
            self.y_std = torch.std(outputs)

    def true_z(self, x):
        for i in range(self.n_polynorm+1):
            response = self.true_theta[i] * (x**i)

            if (self.normalize) and (self.y_mu is not None) and (self.y_std is not None):
                response = (response - self.y_mu)/self.y_std
        return response

    def sigmoid_f(self, x):
        return 1/(1 + torch.exp(x))

    def true_f(self, x):
        response = self.true_z(x)
        return self.sigmoid_f(-response)

    def forward_z(self, x):
        if (self.normalize) and (self.y_mu is not None) and (self.y_std is not None):
            noise_z = self.true_z(x) + self.error_scale * torch.randn((x.shape[0],1))
        else:
            noise_z = self.true_z(x) + self.true_z(x).mean()*self.error_scale * torch.randn((x.shape[0],1))
        return noise_z

    def forward(self, x):
        noise_z = self.forward_z(x)
        probs = self.sigmoid_f(-noise_z)
        bernoulli_dist = torch.distributions.Bernoulli(probs=probs)
        return bernoulli_dist.sample()

    def __call__(self, x):
        return self.forward(x)


if example:
    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    scale = 200
    shift = 0
    # error_scale = 0.3
    # scale = torch.randint(0,100 ,size=(1,))
    # shift = torch.randint(0,500 ,size=(1,))
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
    # f = RewardFunctionTorch()
    # f = UnknownBernoulliFunction()
    f.normalize_setting(x_train)

    y_true = f.true_f(x_train)
    y_train = f(x_train)
    error_sigma = (y_train - y_true).std()
    # true_theta = torch.randn((input_dim,1))
    # y_true = x_train_add_const @ true_theta
    # y_train = y_true + error_scale*scale*torch.randn((x_train_add_const.shape[0],1))

    # Dataset and DataLoader
    train_dataset = TensorDataset(x_train_add_const, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Valid DataSet
    x_valid = torch.randn(300, input_dim_init) *scale + shift   # 300 samples of validation set
    x_valid_add_const = torch.cat([x_valid, torch.ones_like(x_valid)], axis=1)
    y_valid = f(x_valid)
    valid_dataset = TensorDataset(x_train_add_const, y_train)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # Test DataSet
    x_test = torch.randn(200, input_dim_init) *scale + shift   # 200 samples of test set
    x_test_add_const = torch.cat([x_test, torch.ones_like(x_test)], axis=1)
    y_test = f(x_test)
    test_dataset = TensorDataset(x_train_add_const, y_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    # visualize
    x_lin = torch.linspace(x_train.min(),x_train.max(),300).reshape(-1,1)
    x_lin_add_const = torch.concat([x_lin, torch.ones_like(x_lin)], axis=1)

    plt.figure()
    plt.scatter(x_train, y_train, label='obs')
    plt.plot(x_lin, f.true_f(x_lin), color='orange', label='true')
    plt.legend()
    plt.show()


    # plt.plot(x_lin, f.true_z(x_lin), color='orange', label='true')

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

class DirectEstimate(nn.Module):
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
class DirectEnsemble1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_models=10):
        super().__init__()
        self.models = nn.ModuleList([BNN_Direct(input_dim, hidden_dim, output_dim) for _ in range(n_models)])

    def forward(self, x):
        ensemble_outputs = torch.stack([torch.stack(model(x)) for model in self.models])
        mu, logvar = torch.mean(ensemble_outputs, dim=0)
        # logvar = torch.clamp(logvar, min=-5, max=10) 
        return mu, logvar

# ★Mu/Var Ensemble only last layer
class DirectEnsemble2(nn.Module):
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

# ★Mu/Var Ensemble only last layer
class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(),
                batchNorm=True,  dropout=0.5):
        super().__init__()
        ff_block = [nn.Linear(input_dim, output_dim)]
        if activation:
            ff_block.append(activation)
        if batchNorm:
            ff_block.append(nn.BatchNorm1d(output_dim))
        if dropout > 0:
            ff_block.append(nn.Dropout(dropout))
        self.ff_block = nn.Sequential(*ff_block)
    
    def forward(self, x):
        return self.ff_block(x)

class DirectEnsemble3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, n_models=10):
        super().__init__()
        
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, output_dim*2*n_models, activation=False, batchNorm=False, dropout=0)
        self.n_layers = n_layers
        self.n_models = n_models
        self.output_dim = output_dim

    # train step
    def train_forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection

        mu_logvar = (x)
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

if example:
    # Initialize the model, optimizer
    # model = DirectEstimate(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    # model = DirectEnsemble1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10)
    # model = DirectEnsemble2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=5)
    model = DirectEnsemble3(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=5)
    model.to(device)


    # -----------------------------------------------------------------------------------------------
    # Hyperparameters
    learning_rate = 1e-3
    # beta = 0.5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_gaussian = nn.GaussianNLLLoss()
    # mu_loss_weight = 0.5  # mu에 대한 손실 가중치
    # logvar_loss_weight = 0.1  # log_var에 대한 손실 가중치

    num_epochs = 100

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            
            # Forward pass ----------------------------------------
            loss = gaussian_loss(model, batch_x, batch_y)
            # mu, logvar = model(batch_x)
            # std = torch.exp(0.5*logvar)
            
            # (Loss Parameters) ----------------------------------------
            # # fixed variance : # Reparameterization trick: z = mu + sigma * epsilon
            # z = mu + std * torch.randn_like(std)    
            # loss = torch.sum( (z - batch_y)**2)

            # # variational variance : neg_log_likelihood
            # loss = torch.sum(0.5 * torch.log(std**2) + 0.5 * (1/std**2) * (batch_y - mu) ** 2)
            # loss = -torch.distributions.Normal(mu, std).log_prob(batch_y).sum()
            # loss = loss_gaussian(mu, batch_y, std**2)

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






























###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
from six.moves import cPickle
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.notebook import tqdm
from IPython.display import clear_output, display, update_display

class TorchModeling():
    def __init__(self, model, device='cpu'):
        self.now_date = datetime.strftime(datetime.now(), '%y%m%d_%H')

        self.model = model.to(device)
        self.device = device
        self.t = 1

        self.train_losses = []
        self.train_metrics = []
        self.valid_losses = []
        self.valid_metrics = []
        self.test_losses = []
        self.test_metrics = [] 

        self.train_info = []
        self.test_info = []
    
    def get_save_path(self):
        return f"{os.getcwd()}/{self.now_date}_{self.model._get_name()}"

    def fun_decimal_point(self, value):
        if type(value) == str or type(value) == int:
            return value
        else:
            if value == 0:
                return 3
            try:
                point_log10 = np.floor(np.log10(abs(value)))
                point = int((point_log10 - 3)* -1) if point_log10 >= 0 else int((point_log10 - 2)* -1)
            except:
                point = 0
            return np.round(value, point)

    def compile(self, optimizer, loss_function, metric_function=None, scheduler=None,
                early_stop_loss=None, early_stop_metrics=None):
        """
        loss_function(model, x, y) -> loss
        """
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics_function = metric_function
        self.scheduler = scheduler
        self.early_stop_loss = early_stop_loss
        self.early_stop_metrics = early_stop_metrics

    def recompile(self, optimizer=None, loss_function=None, metric_function=None, scheduler=None,
                early_stop_loss=None, early_stop_metrics=None):
        if scheduler is not None:
            self.scheduler = scheduler
            self.scheduler.optimizer = self.optimizer

        if optimizer is not None:
            self.optimizer = optimizer
            self.scheduler.optimizer = self.optimizer

        if loss_function is not None:
            self.loss_function = loss_function
        
        if metric_function is not None:
            self.metrics_function = metric_function

        if early_stop_loss is not None:
            self.early_stop_loss.patience = early_stop_loss.patience
            self.early_stop_loss.optimize = early_stop_loss.optimize
            early_stop_loss.load(self.early_stop_loss)
            self.early_stop_loss = early_stop_loss

        if early_stop_metrics is not None:
            self.early_stop_metrics.patience = early_stop_metrics.patience
            self.early_stop_metrics.optimize = early_stop_metrics.optimize
            early_stop_metrics.load(self.early_stop_metrics)
            self.early_stop_metrics = early_stop_metrics

    def train_model(self, train_loader, valid_loader=None, epochs=10, tqdm_display=False,
                early_stop=True, save_parameters=False, display_earlystop_result=False):
        final_epcohs = self.t + epochs - 1
        # [START of Epochs Loop] ############################################################################################
        epochs_iter = tqdm(range(self.t, self.t + epochs), desc="Epochs", total=epochs, position=0, leave=True) if tqdm_display else range(self.t, self.t + epochs)
        for epoch in epochs_iter:
            print_info = {}

            # train Loop --------------------------------------------------------------
            self.model.train()
            train_epoch_loss = []
            train_epoch_metrics = []
            train_iter = tqdm(enumerate(train_loader), desc="Train Batch", total=len(train_loader), position=1, leave=False) if tqdm_display else enumerate(train_loader)
            for batch_idx, (batch_x, batch_y) in train_iter:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_function(self.model, batch_x, batch_y)
                loss.backward()
                self.optimizer.step()
            
                with torch.no_grad():
                    train_epoch_loss.append( loss.to('cpu').detach().numpy() )
                    if self.metrics_function is not None:
                        train_epoch_metrics.append( self.metric_f(self.model, batch_x, batch_y) )

            with torch.no_grad():
                print_info['train_loss'] = np.mean(train_epoch_loss)
                self.train_losses.append(print_info['train_loss'])
                if self.metrics_function is not None:
                    print_info['train_metrics'] = np.mean(train_epoch_metrics)
                    self.train_metrics.append(print_info['train_metrics'])

            # scheduler ---------------------------------------------------------
            if self.scheduler is not None:
                self.scheduler.step()

            with torch.no_grad():
                # valid Loop ---------------------------------------------------------
                if valid_loader is not None and len(valid_loader) > 0:
                    self.model.eval()
                    valid_epoch_loss = []
                    valid_epoch_metrics = []
                    valid_iter = tqdm(enumerate(valid_loader), desc="Valid Batch", total=len(valid_loader), position=1, leave=False) if tqdm_display else enumerate(valid_loader)
                    for batch_idx, (batch_x, batch_y) in valid_iter:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        
                        loss = self.loss_function(self.model, batch_x, batch_y)
                    
                        valid_epoch_loss.append( loss.to('cpu').detach().numpy() )
                        if self.metrics_function is not None:
                            valid_epoch_metrics.append( self.metric_f(self.model, batch_x, batch_y) )

                    print_info['valid_loss'] = np.mean(valid_epoch_loss)
                    self.valid_losses.append(print_info['valid_loss'])
                    if self.metrics_function is not None:
                        print_info['valid_metrics'] = np.mean(valid_epoch_metrics)
                        self.valid_metrics.append(print_info['valid_metrics'])
            
                # print_info ---------------------------------------------------------
                self.train_info.append(print_info)
                print_sentences = ",  ".join([f"{k}: {str(self.fun_decimal_point(v))}" for k, v in print_info.items()])
                
                # print(f"[Epoch: {epoch}/{final_epcohs}] {print_sentences}")
                if final_epcohs - epoch + 1 == epochs:
                    display(f"[Epoch: {epoch}/{final_epcohs}] {print_sentences}", display_id="epoch_result")
                else:
                    update_display(f"[Epoch: {epoch}/{final_epcohs}] {print_sentences}", display_id="epoch_result")

                # early_stop ---------------------------------------------------------
                early_stop_TF = None
                if self.early_stop_loss is not None:
                    score = print_info['valid_loss'] if (valid_loader is not None and len(valid_loader) > 0) else print_info['train_loss']
                    reference_score = print_info['train_loss'] if (valid_loader is not None and len(valid_loader) > 0) else None
                    params = self.model.state_dict() if save_parameters else None
                    early_stop_TF = self.early_stop_loss.early_stop(score=score, reference_score=reference_score,save=params, verbose=0)

                    if save_parameters:
                        path_save_loss = f"{self.get_save_path()}_earlystop_loss.pth"
                        cPickle.dump(self.early_stop_loss, open(path_save_loss, 'wb'))      # save earlystop loss

                if self.metrics_function is not None and self.early_stop_metrics is not None:
                    score = print_info['valid_metrics'] if (valid_loader is not None and len(valid_loader) > 0) else print_info['train_metrics']
                    reference_score = print_info['train_metrics'] if (valid_loader is not None and len(valid_loader) > 0) else None
                    params = self.model.state_dict() if save_parameters else None
                    self.early_stop_loss.early_stop(score=score, reference_score=reference_score, save=params, verbose=0)

                    if save_parameters:
                        path_save_metrics = f"{self.get_save_path()}_earlystop_metrics.pth"
                        cPickle.dump(self.early_stop_metrics, open(path_save_metrics, 'wb'))      # save earlystop metrics

                # save_parameters ---------------------------------------------------------
                if save_parameters:
                    path_save_weight = f"{self.get_save_path()}_weights.pth"
                    cPickle.dump(self.model.state_dict(), open(path_save_weight, 'wb'))      # save earlystop weights

                # step update ---------------------------------------------------------
                self.t += 1

                # early_stop break ---------------------------------------------------------
                if early_stop is True and early_stop_TF == 'break':
                        break
        
        if display_earlystop_result:
            if self.early_stop_loss is not None:
                display(self.early_stop_loss.plot)
            if self.metrics_function is not None and self.early_stop_metrics is not None:
                display(self.early_stop_metrics.plot)
        # [END of Epochs Loop] ############################################################################################

    def test_model(self, test_loader, tqdm_display=False):
        with torch.no_grad():
            print_info = {"epoch":self.t-1}
            # test Loop ---------------------------------------------------------
            if test_loader is not None and len(test_loader) > 0:
                self.model.eval()
                test_epoch_loss = []
                test_epoch_metrics = []
                test_iter = tqdm(enumerate(test_loader), desc="Valid Batch", total=len(test_loader), position=1, leave=False) if tqdm_display else enumerate(test_loader)
                for batch_idx, (batch_x, batch_y) in test_iter:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    loss = self.loss_function(self.model, batch_x, batch_y)
                
                    test_epoch_loss.append( loss.to('cpu').detach().numpy() )
                    if self.metrics_function is not None:
                        test_epoch_metrics.append( self.metric_f(self.model, batch_x, batch_y) )

                print_info['test_loss'] = np.mean(test_epoch_loss)
                self.test_losses.append(print_info['test_loss'])
                if self.metrics_function is not None:
                    print_info['test_metrics'] = np.mean(test_epoch_metrics)
                    self.test_metrics.append(print_info['test_metrics'])
            print_sentences = ",  ".join([f"{k}: {str(self.fun_decimal_point(v))}" for k, v in print_info.items() if k != 'epoch'])
            print(f"[After {self.t-1} epoch test performances] {print_sentences}")
            self.test_info.append(print_info)


# import importlib
# import requests
# importlib.reload(httpimport)


# response_DS_DataFrame = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_DataFrame.py")
# response_DS_Plot = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_Plot.py")
# response_DS_MachineLearning = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_MachineLearning.py")
# response_DS_DeepLearning = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_DeepLearning.py")
# response_DS_Torch = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_Torch.py")
# exec(response_DS_DataFrame.text)
# exec(response_DS_Plot.text)
# exec(response_DS_MachineLearning.text)
# exec(response_DS_DeepLearning.text)
# exec(DS_Torch.text)


import numpy as np
import pandas as pd
import missingno as msno
import httpimport
# from datetime import datetime

if example:
    remote_library_url = 'https://raw.githubusercontent.com/kimds929/'

    with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
        from DS_DeepLearning import EarlyStopping

    with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
        from DS_Torch import TorchDataLoader, TorchModeling, AutoML

    # with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
    #     from DS_TorchModule import ScaledDotProductAttention, MultiHeadAttentionLayer





# loss_mse = nn.MSELoss()
def mse_loss(model, x, y):
    logmu = model(x)
    mu = torch.exp(logmu)
    loss = torch.nn.functional.mse_loss(mu, y)
    return loss


# loss_gaussian = nn.GaussianNLLLoss()
def gaussian_loss(model, x, y):
    mu, logvar = model(x)
    std = torch.exp(0.5*logvar)
    loss = torch.nn.functional.gaussian_nll_loss(mu, y, std**2)
    # loss = loss_gaussian(mu, y, std**2)
    return loss

def bernoulli_loss(model, x, y):
    mu, logvar = model(x)
    std = torch.exp(0.5*logvar)
    logit = torch.log((y + 1e-10) / (1 - y + 1e-10))
    loss = torch.nn.functional.gaussian_nll_loss(mu, logit, std**2)
    # loss = loss_gaussian(mu, logit, std**2)
    return loss

if example:
    f.true_z(x_train).numpy().astype(int)
    # f.forward_z(x_train).numpy().astype(int)
    # f.true_z(x_train).numpy().astype(int)
    model(x_train_add_const.to(device))

    # model = DirectEnsemble2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=5)
    model = DirectEnsemble3(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10, n_layers=5)
    # model = SampleEnsemble2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_samples=10, n_layers=5)
    # [name for name, layer in model.EnsembleBlock.named_children()]


    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    tm = TorchModeling(model=model, device=device)
    tm.compile(optimizer=optimizer
                # ,loss_function = gaussian_loss
                ,loss_function = bernoulli_loss
                , scheduler=scheduler
                , early_stop_loss = EarlyStopping(patience=5)
                )
    # tm.early_stop_loss = None
    # tm.early_stop_loss = EarlyStopping(patience=20)
    # tm.early_stop_loss.reset_patience_scores()

    # tm.training(train_loader=train_loader, valid_loader=valid_loader, epochs=100, display_earlystop_result=True)
    tm.train_model(train_loader=train_loader, valid_loader=valid_loader, epochs=100, display_earlystop_result=True, early_stop=False)
    # tm.train_model(train_loader=train_loader, valid_loader=valid_loader, epochs=10, tqdm_display=True, display_earlystop_result=True, early_stop=False)
    tm.test_model(test_loader=test_loader)
    # tm.early_stop_loss.generate_plot(figsize=(15,4))

    # tm.optimizer
    # tm.recompile(early_stop_loss = EarlyStopping(patience=4))
    # tm.recompile(optimizer = optim.Adam(model.parameters(), lr=1e-3))
    # tm.recompile(optimizer = optim.Adam(model.parameters(), lr=1e-4))
    # tm.recompile(scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2))




################################################################################################################################

if example:
    # visualize
    x_lin = torch.linspace(x_train.min(),x_train.max(),300).reshape(-1,1)
    x_lin_add_const = torch.concat([x_lin, torch.ones_like(x_lin)], axis=1)

    with torch.no_grad():
        model.eval()
        y_mu, y_logvar = model(x_lin_add_const.to(device))
        y_mu = y_mu.to('cpu')
        y_std = torch.exp(0.5*y_logvar).to('cpu')

    # # Gaussian
    # plt.figure()
    # plt.scatter(x_train, y_train, label='obs', alpha=0.5)
    # plt.plot(x_lin, f.true_f(x_lin), color='orange', label='true', alpha=0.5)

    # plt.plot(x_lin, y_mu, alpha=0.5, color='mediumseagreen', label='pred_mu')
    # plt.fill_between(x_lin.flatten(), (y_mu-y_std).flatten(), (y_mu+y_std).flatten(), color='mediumseagreen', alpha=0.2, label='pred_var')
    # plt.legend()
    # plt.show()

    # Logit
    plt.figure()
    plt.scatter(x_train, y_train, label='obs', alpha=0.5)
    plt.plot(x_lin, f.true_f(x_lin), color='orange', label='true', alpha=0.5)

    plt.plot(x_lin, 1/(1+torch.exp(-y_mu)), alpha=0.5, color='mediumseagreen', label='pred_mu')
    plt.fill_between(x_lin.flatten(), 1/(1+torch.exp(-(y_mu-y_std))).flatten(), 1/(1+torch.exp(-(y_mu+y_std))).flatten(), color='mediumseagreen', alpha=0.2, label='pred_var')
    plt.legend()
    plt.show()



if example:
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

###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################






























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

if example:
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
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
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
class SampleEnsemble1(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, n_samples=10):
        super().__init__()
        hidden_dims = torch.randint(low=8,high=128,size=(n_samples,)) if hidden_dim is None else torch.ones((n_models,)).type(torch.int)*n_models

        self.models = nn.ModuleList([BasicNetwork(input_dim, int(h_dim), output_dim) for _, h_dim in zip(range(n_models), hidden_dims)])
    
    def forward(self, x):
        ensemble_outputs = torch.stack([model(x) for model in self.models])
        mu = torch.mean(ensemble_outputs, dim=0)
        var = torch.var(ensemble_outputs, dim=0)
        logvar = torch.log(var)
        return mu, logvar

# ★ Sample Ensemble only last layers Model
class SampleEnsemble2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_samples=10):
        super().__init__()
        self.basic_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, output_dim*n_samples)
        )
    
    def forward(self, x):
        ensemble_outputs = self.basic_block(x)
        # return ensemble_outputs
        mu = torch.mean(ensemble_outputs, dim=1, keepdims=True)
        var = torch.var(ensemble_outputs, dim=1, keepdims=True)
        logvar = torch.log(var)
        return mu, logvar


# ★ Sample Ensemble only last layers Model
class SampleEnsemble2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, n_samples=10):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, output_dim*n_samples, activation=False, batchNorm=False, dropout=0)
        self.n_layers = n_layers
        self.n_samples = n_samples
        self.output_dim = output_dim

    
    def forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection

        # return ensemble_outputs
        mu = torch.mean(x, dim=1, keepdims=True)
        var = torch.var(x, dim=1, keepdims=True)
        logvar = torch.log(var)
        return mu, logvar

################################################################################################

if example:
    # model = SampleEnsemble1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10)    # initial weight ensemble
    # model = SampleEnsemble1(input_dim=input_dim, hidden_dim=None, output_dim=output_dim, n_models=10)    # architecture ensemble
    model = SampleEnsemble2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=10)    # initial weight ensemble
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
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
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

if example:
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
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
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