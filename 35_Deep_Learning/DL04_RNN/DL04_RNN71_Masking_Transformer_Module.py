import torch
import torch.nn as nn


class MaskingTransformer(nn.Module):
    def __init__(self, embed_dim=4):
        super().__init__()

        self.linear_embedding = nn.Linear(1, embed_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, dim_feedforward=embed_dim, batch_first=True, dropout=0) # dropout=0, for consistent result 
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.output_layer = nn.Linear(embed_dim, 1)
        # nn.init.constant_(self.linear_embedding.weight, 1)
        # nn.init.constant_(self.linear_embedding.bias, 1)
        # self._initialize_transformer_weights()

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
                    
    def forward(self, x, mask=True):
        unsqueeze_x = x.unsqueeze(-1)
        embed_x = self.linear_embedding(unsqueeze_x)

        # Create a causal mask to prevent future information leakage
        if mask is True:
            seq_len = embed_x.size(-2)  # (batch, seq, embed)
            casual_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            # casual_mask = casual_mask.masked_fill(casual_mask == 1, float('-inf'))
            output_transformer = self.transformer_encoder(src=embed_x, mask=casual_mask)
        else:
            output_transformer = self.transformer_encoder(src=embed_x)
        output = self.output_layer(output_transformer).squeeze(-1)
        return output

# seq_len = 5
# torch.triu(torch.ones(seq_len, seq_len))

x1 = torch.arange(3).type(torch.float32).tile(10,1)
x2 = torch.arange(5).type(torch.float32).tile(10,1)
x3 = torch.arange(8).type(torch.float32).tile(10,1)

model = MaskingTransformer()

model(x1)
model(x2[:,:3])
model(x2, mask=True)
model(x2, mask=False)
model(x3, mask=True)
model(x3, mask=False)
