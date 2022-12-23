import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
from IPython.display import clear_output
from collections import OrderedDict

# DeepLearning MDL Predict
class PredictDL():
    def __init__(self, model, input='torch'):
        self.model = model
        self.input = input
    
    def predict(self, x):
        if self.input == 'torch':
            return self.model.predict(torch.FloatTensor(np.array(x))).numpy().ravel()




class AutoML(torch.nn.Module):
    '''
    【 Required Library 】torch, from collections import OrderedDict
    【 Required Customized Class 】
    
     < Method >
     . __init__:
     . create_architecture:
     . create_model:
     . forward:
     . predict:
    '''
    def __init__(self, X, y=None, hidden_layers=3, hidden_nodes=None, structure_type=None,
                 layer_structure={'Linear':'io','BatchNorm1d': 'o', 'ReLU': None}):
        super(AutoML, self).__init__()
        self.x_shape = X.shape
        self.y_ndim = None if y is None else y.ndim
        self.y_shape = None if y is None else y.shape
        
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.structure_type = structure_type
        
        self.model_dict = OrderedDict()
        
        self.layer_structure = layer_structure
        self.create_architecture()      # AutoML Architecture
        self.create_model()      # AutoML Architecture
        
        self.model = torch.nn.Sequential(self.model_dict)
        
        self.predicts = {}
    
    def create_architecture(self):
        n = self.x_shape[1]
        
        if self.hidden_nodes is None:
            # AutoML with Hidden Layer 
            hidden_nodes = [n]
            
            if self.structure_type is None or self.structure_type == 1:
                for i in range(self.hidden_layers):
                    n *= 2
                    hidden_nodes.append(n)
                self.hidden_nodes = hidden_nodes
            elif self.structure_type == 2:
                for i in range( (self.hidden_layers-1)//2):
                    n *= 2
                    hidden_nodes.append(n)
                self.hidden_nodes = hidden_nodes.copy()
                if (self.hidden_layers-1) % 2 == 1:
                    self.hidden_nodes.append(n*2)
                self.hidden_nodes = self.hidden_nodes + hidden_nodes[::-1]
            
        else:
            # AutoML with Hidden Nodes 
            self.hidden_layers = len(self.hidden_nodes)
            self.hidden_nodes = [n] + self.hidden_nodes
        
        if self.y_ndim is None:
            self.hidden_nodes.append(1)
        else:
            self.hidden_nodes.append(1 if  self.y_ndim == 1 else self.y_shape[1])
   
    def create_model(self, layer_structure=None):
        layer_structure = self.layer_structure if layer_structure is None else layer_structure
            
        for hi, hn in enumerate(self.hidden_nodes):
            if hi < len(self.hidden_nodes)-2:
                n_input = self.hidden_nodes[hi]
                n_output = self.hidden_nodes[hi+1]
                
                for ls, ls_io in layer_structure.items():
                    if ls_io == 'io':
                        self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}({n_input}, {n_output})") })
                    elif ls_io == 'i':
                        self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}({n_input})") })
                    elif ls_io == 'o':
                        self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}({n_output})") })
                    elif ls_io is None:
                        self.model_dict.update({f"l{hi}_{ls}": eval(f"torch.nn.{ls}()") })
            elif hi == len(self.hidden_nodes)-2:
                self.model_dict.update({f"nn{hi}": torch.nn.Linear(self.hidden_nodes[hi], self.hidden_nodes[hi+1])})

    def forward(self, x, training=True):
        if training is True:
            for layer_name, layer in zip(self.model_dict.keys(), self.model):
                x = layer(x)
                self.predicts[layer_name] = x
            
        elif training is False:
            with torch.no_grad():
                for layer_name, layer in zip(self.model_dict.keys(), self.model):
                    x = layer(x)
                    self.predicts[layer_name] = x
        return x

    def predict(self, x):
        return self.forward(x, training=False)
