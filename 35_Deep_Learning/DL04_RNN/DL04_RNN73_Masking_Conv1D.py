import torch
import torch.nn as nn




class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=100, stride=1, dilation=1):
        """
        kernel_size : kerner_size - 1 개의 과거정보만 참조
        """
        super(CausalConv1D, self).__init__()
        self.padding = (kernel_size - 1) * dilation  # Ensure causality
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=self.padding, 
            dilation=dilation
        )

    def forward(self, x):
        x.unsqueeze(-1)
        # x: (batch_size, seq_length, embedding_dim)
        x = x.transpose(-2, -1)  # Change to (batch_size, embedding_dim, seq_length)
        x = self.conv(x)
        x = x[:, :, :-self.padding]  # Remove future-padded values
        x = x.transpose(-2, -1)  # Back to (batch_size, seq_length, embedding_dim)
        return x


class MaskingConv1D(nn.Module):
    def __init__(self, embed_dim=4, window_size=100):
        super().__init__()
        self.linear_embedding = nn.Linear(1, embed_dim)
        self.MaskingConv1D = CausalConv1D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=window_size, stride=1, dilation=1)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, x):
        unsqueeze_x = x.unsqueeze(-1)
        embed_x = self.linear_embedding(unsqueeze_x)
        output_conv1d = self.MaskingConv1D(embed_x)
        output = self.output_layer(output_conv1d).squeeze(-1)
        return output


x1 = torch.arange(3).type(torch.float32).tile(10,1)
x2 = torch.arange(5).type(torch.float32).tile(10,1)
x3 = torch.arange(8).type(torch.float32).tile(10,1)

model = MaskingConv1D()

model(x1)
model(x2[:,:3])
model(x2)
model(x3)