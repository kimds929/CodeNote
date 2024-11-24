import torch
import torch.nn as nn
import torch.nn.init as init


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_block = nn.Sequential(
            nn.Linear(5,10)
            ,nn.Linear(10,20)
            ,nn.Linear(20,20)
            ,nn.Linear(20,20)
            ,nn.Linear(20,20)
            ,nn.Linear(20,20)
            ,nn.Linear(20,1)
        )

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
        
    def forward(self, x, layer=1):
        return self.linear_block(x)

model = TorchModel()
model(torch.rand(20,5))

