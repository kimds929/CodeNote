import torch
import torch.nn as nn

# SlidingWindowLayer
class SlidingWindowLayer(nn.Module):
    def __init__(self, embed_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.fc = nn.Linear(window_size * embed_dim, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_length, embed_dim)
        batch_size = x.shape[0]
        seq_length = x.shape[-2]
        # batch_size, seq_length, embed_dim = x.size()

        padding = (self.window_size-1, 0)
        padded_x = nn.functional.pad(input=x, pad=(0,0, *padding))  # Pad the start

        windows = padded_x.unfold(dimension=-2, size=self.window_size, step=1)  # (batch_size, seq_length, window_size, embed_dim)
        print(windows.shape)
        windows = windows.contiguous().view(batch_size, seq_length, -1)  # Flatten window
        embed_output = self.fc(windows)
        return embed_output


# # LearnableWindowLayer
# class LearnableWindowLayer(nn.Module):
#     def __init__(self, embed_dim, max_window_size=100):
#         super().__init__()
#         self.max_window_size = max_window_size
#         self.logits = nn.Parameter(torch.zeros(max_window_size))
#         # self.window_size_float = nn.Parameter(torch.tensor(0.1))

#     # def get_window_size(self):
#     #     window_size = self.window_size_float*100
#     #     return torch.clamp(window_size.round().int(), min=1, max=self.max_window_size)

#     def get_window_weights(self):
#         # 확률 분포 계산
#         weights = nn.functional.softmax(self.logits, dim=0)  # (max_window_size,)
#         return weights

#     def forward(self, x):
#         # x: (batch_size, seq_length, embed_dim)
#         batch_size = x.shape[0]
#         seq_length = x.shape[-2]
#         # batch_size, seq_length, embed_dim = x.size()

#         window_weights = self.get_window_weights()
#         window_size = window_weights.shape[0] 
#         # window_size = self.get_window_size()
#         padding = (window_size-1, 0)
#         padded_x = nn.functional.pad(input=x, pad=(0, 0, *padding))  # Pad the start

#         windows = padded_x.unfold(dimension=-2, size=window_size, step=1)  # (batch_size, seq_length, window_size, embed_dim)
#         windows_mean = windows.mean(dim=-1)
        
#         return windows_mean


# Numerical Input
class MaskingSlidingWindow(nn.Module):
    def __init__(self, embed_dim=4, window_size=100, pad_token=np.inf):
        super().__init__()
        self.pad_token = float(pad_token)

        self.embedding_layer = nn.Linear(1, embed_dim)
        self.MaskingSWL = SlidingWindowLayer(embed_dim=embed_dim, window_size=window_size)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, x, return_last_seq=True):
        unsqueeze_x = x.unsqueeze(-1)
        
        embed_x = self.embedding_layer(unsqueeze_x)
        output_swl = self.MaskingSWL(embed_x)
        output = self.output_layer(output_swl).squeeze(-1)
        if return_last_seq is True:
            return output[torch.arange(x.size(0)),(~(x==self.pad_token)).sum(dim=-1)-1].view(-1,1)
        else:
            return output

x1 = torch.arange(3).type(torch.float32).tile(10,1)
x2 = torch.arange(5).type(torch.float32).tile(10,1)
x3 = torch.arange(8).type(torch.float32).tile(10,1)



model = MaskingSlidingWindow()
model( torch.tensor([[0,np.inf,np.inf], [1,2,np.inf]]) )
model( torch.tensor([[0,np.inf,np.inf], [1,2,np.inf]]), return_last_seq=False)
model(x1, return_last_seq=False)
model(x2[:,:3], return_last_seq=False)
model(x2, return_last_seq=False)
model(x3, return_last_seq=False)


# Categorical Input
class MaskingSlidingWindow(nn.Module):
    def __init__(self, embed_dim=4, window_size=100, pad_token=0):
        super().__init__()
        self.pad_token = int(pad_token)
        # self.embedding_layer = nn.Linear(1, embed_dim)
        self.embedding_layer = nn.Embedding(num_embeddings=7, embedding_dim=embed_dim)
        self.MaskingSWL = SlidingWindowLayer(embed_dim=embed_dim, window_size=window_size)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, x, return_last_seq=True):
        embed_x = self.embedding_layer(x)
        output_mha = self.MaskingSWL(embed_x)
        output = self.output_layer(output_mha).squeeze(-1)

        if return_last_seq is True:
            return output[torch.arange(x.size(0)),(~(x==self.pad_token)).sum(dim=-1)-1].view(-1,1)
        else:
            return output


model = MaskingSlidingWindow()
model(X_tensor[:10,:3], return_last_seq=False)
model(X_tensor[:10,:5], return_last_seq=False)
model(X_tensor[:10,:8], return_last_seq=False)

model(X_tensor[:10,:3], return_last_seq=True)
model(X_tensor[:10,:5], return_last_seq=True)
model(X_tensor[:10,:8], return_last_seq=True)




# # ---------------------------------------------------------------------------
# from torch.utils.data import DataLoader, TensorDataset
# from tqdm.auto import tqdm
# y_tensor = torch.rand(X_tensor.size(0)).view(-1,1)
# dataset = TensorDataset(X_tensor, y_tensor)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# optimizer = torch.optim.Adam(model.parameters())

# n_epoch = 3
# for e in range(n_epoch):
#     batch_loss_list = []
#     with tqdm(total=len(dataloader), desc=f"Epoch {e+1}/{n_epoch}") as pbar:
#         for batch_x, batch_y in dataloader:
#             optimizer.zero_grad()

#             pred = model(batch_x)
#             loss = nn.functional.mse_loss(pred, batch_y)

#             loss.backward()
#             optimizer.step()

#             with torch.no_grad():
#                 batch_loss_list.append( loss.item() )
            
#             pbar.set_postfix(Loss=f"{np.mean(batch_loss_list).item():.2f}")
#             pbar.update(1)








