import torch
import torch.nn as nn


# BaroNowModel
class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(6, 10)
            ,nn.Linear(10,1)
            )
        self.linear2 =  nn.ModuleList([nn.Linear(6,1)])
        self.linear3 = nn.ModuleDict({'Linear':nn.Linear(6,1)})
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=6, nhead=2, dim_feedforward=16, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        self.linear4 = nn.Linear(6,1)

        nn.init.constant_(self.linear1[0].weight, 0.5)
        nn.init.constant_(self.linear1[0].bias, 0)
        nn.init.constant_(self.linear2[0].weight, 0.5)
        nn.init.constant_(self.linear2[0].bias, 0)
        nn.init.constant_(self.linear3['Linear'].weight, 0.5)
        nn.init.constant_(self.linear3['Linear'].bias, 0)

    def weight_activate(self, activate=None, deactivate=None, all_activate=None):
        # all weight activate
        if (all_activate is True) or (all_activate is False):
            for param in self.parameters():
                param.requires_grad = all_activate

        if (activate is not None) and isinstance(activate, nn.Module):
            for name, param in activate.named_parameters():
                param.requires_grad = True

        if (deactivate is not None)and isinstance(deactivate, nn.Module):
            for name, param in deactivate.named_parameters():
                param.requires_grad = False

    def requires_grad_status(self, layer=None, verbose=0):
        if layer is None:
            layer = self

        require_grad_list = {}
        if isinstance(layer, nn.Module):
            for name, param in layer.named_parameters():
                require_grad_list[name] = param.requires_grad
                
                if verbose > 0:
                    print(f"({name}) {param.requires_grad}")
            return require_grad_list

    def forward(self, x, layer=1):
        if layer == 1:
            out = self.linear1(x)
            self.weight_activate(activate=self.linear1, all_activate=False)

        elif layer == 2:
            out = self.linear2[0](x)
            self.weight_activate(activate=self.linear2, all_activate=False)
        elif layer == 3:
            out = self.linear3['Linear'](x)
            self.weight_activate(activate=self.linear3, all_activate=False)
        elif layer ==4 :
            out = self.linear4(self.transformer_encoder(x))
            self.weight_activate(all_activate=False)
            self.weight_activate(activate=self.transformer_encoder_layer)
            self.weight_activate(activate=self.transformer_encoder)
            self.weight_activate(activate=self.linear4)
            
        return out


# model = TorchModel()
# model.requires_grad_status()
# model(torch.rand(10,6), layer=1).shape

