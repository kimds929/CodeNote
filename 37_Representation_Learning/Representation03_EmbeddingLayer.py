import torch
import torch.nn as nn


########################################################################################

class LinearEmbedding(nn.Module):
    def __init__(self, feature_dim, embed_dim):
        super().__init__()
        # 각 feature별로 weight와 bias를 직접 파라미터로 선언
        self.weight = nn.Parameter(torch.randn(feature_dim, embed_dim, 1))  # (feature_dim, embed_dim, 1)
        self.bias = nn.Parameter(torch.randn(feature_dim, embed_dim))       # (feature_dim, embed_dim)
        self.embed_dim = embed_dim

        # weight 초기화
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')
        fan_in = torch.tensor(1.0)  # 각 feature별로 입력이 1개이므로
        bound = 1.0 / torch.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (batch_dim, feature_dim)
        x = x.unsqueeze(-1).unsqueeze(-1)  # (batch_dim, feature_dim, 1, 1)
        out = torch.matmul(x, self.weight.transpose(1, 2))  # (batch_dim, feature_dim, 1, embed_dim)
        out = out.squeeze(-2) + self.bias      # (batch_dim, feature_dim, embed_dim)
        return out

# le = LinearEmbedding(3,2)
# le(torch.rand(10,3)).shape


class SineEmbedding(nn.Module):
    def __init__(self, feature_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # 각 feature별로 k개의 진폭, 주기, 위상 파라미터 선언
        self.amplitude = nn.Parameter(torch.ones(feature_dim, embed_dim))      # (feature_dim, embed_dim)
        self.frequency = nn.Parameter(torch.ones(feature_dim, embed_dim))      # (feature_dim, embed_dim)
        self.phase     = nn.Parameter(torch.zeros(feature_dim, embed_dim))     # (feature_dim, embed_dim)

        # 진폭, 주기, 위상 초기화
        nn.init.normal_(self.amplitude, mean=1.0, std=0.1)
        nn.init.normal_(self.frequency, mean=1.0, std=0.1)
        nn.init.normal_(self.phase, mean=0.0, std=0.1)

    def forward(self, x):
        # x: (batch_dim, feature_dim)
        x = x.unsqueeze(-1)  # (batch_dim, feature_dim, 1)
        # (batch_dim, feature_dim, 1) * (feature_dim, embed_dim) -> (batch_dim, feature_dim, embed_dim)
        out = self.amplitude * torch.sin(self.frequency * x + self.phase)  # (batch_dim, feature_dim, embed_dim)
        return out

# se = SineEmbedding(3,2)
# se(torch.rand(3))

########################################################################################
