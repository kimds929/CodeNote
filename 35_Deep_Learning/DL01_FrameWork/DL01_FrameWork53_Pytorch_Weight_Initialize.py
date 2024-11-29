import torch
import torch.nn as nn
import torch.nn.init as init


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_block = nn.Sequential(
            nn.Linear(1,8)
            ,nn.Linear(8,8)
            ,nn.Linear(8,8)
            ,nn.Linear(8,8)
            ,nn.Linear(8,8)
            ,nn.Linear(8,8)
        )
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=8, batch_first=True, dropout=0) # dropout=0, for consistent result 
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)

        self.ouput_layer = nn.Linear(8,1)


        with torch.no_grad():
            nn.init.zeros_(self.linear_block[0].weight)
            nn.init.ones_(self.linear_block[0].bias)

            nn.init.constant_(self.linear_block[1].weight, 0.5)
            nn.init.constant_(self.linear_block[1].weight, 1.0)

            nn.init.uniform_(self.linear_block[2].bias, a=-0.5, b=0.5)
            nn.init.normal_(self.linear_block[2].weight, mean=0.0, std=0.01)

            self.linear_block[3].weight.uniform_(-1/self.linear_block[2].in_features, 1/self.linear_block[2].in_features)
            self.linear_block[3].weight.fill_(0.01)

            # nn.init.xavier_uniform_(self.linear_block[4].weight)
            nn.init.xavier_normal_(self.linear_block[4].weight)
            nn.init.zeros_(self.linear_block[4].bias)

            # nn.init.kaiming_uniform_(self.linear_block[5].weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.linear_block[5].weight, nonlinearity='relu')
            nn.init.zeros_(self.linear_block[5].bias)
            
            # initialize_transformer weight
            self._initialize_transformer_weights()

    def _initialize_transformer_weights(self):
        # Initialize only Transformer layers
        for module in self.modules():
            # Check if the module is part of TransformerEncoder or TransformerEncoderLayer
            if isinstance(module, nn.TransformerEncoderLayer) or isinstance(module, nn.TransformerEncoder):
                for sub_module in module.modules():
                    if isinstance(sub_module, nn.Linear):
                        nn.init.constant_(sub_module.weight, 0.5)
                        if sub_module.bias is not None:
                            nn.init.constant_(sub_module.bias, 0.1)
                    elif isinstance(sub_module, nn.LayerNorm):
                        nn.init.constant_(sub_module.weight, 0.5)
                        nn.init.constant_(sub_module.bias, 0.1)
                    elif isinstance(sub_module, nn.MultiheadAttention):
                        nn.init.constant_(sub_module.in_proj_weight, 0.5)
                        nn.init.constant_(sub_module.out_proj.weight, 0.5)
                        nn.init.constant_(sub_module.out_proj.bias, 0.1)
    
    def forward(self, x, layer=1):
        unsqueeze_x = x.unsqueeze(-1)
        x_embed = self.linear_block(unsqueeze_x)

        x_transformer = self.transformer_encoder(x_embed)
        output = self.ouput_layer(x_transformer).squeeze()
        output_last = output[...,[-1]]

        return output_last

model = TorchModel()
model(torch.rand(20,5))

