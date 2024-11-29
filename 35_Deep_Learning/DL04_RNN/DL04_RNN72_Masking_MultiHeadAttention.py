import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CausalSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = (embed_dim // num_heads) ** -0.5

    def forward(self, x, mask=None):
        # x: (batch_size, seq_length, embed_dim)
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)  # (batch_size, seq_length, 3 * embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_length, seq_length)
        # Generate causal mask
        if mask is True:
            mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).to(x.device)
            attn_weights = attn_weights.masked_fill(mask == 1, float('-inf'))
        elif mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 1, float('-inf'))

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)
        attn_output = torch.matmul(attn_probs, v)  # (batch_size, num_heads, seq_length, head_dim)

        # Combine heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        return self.out_proj(attn_output)


class MaskingMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=4, num_heads=2):
        super().__init__()
        self.linear_embedding = nn.Linear(1, embed_dim)
        self.MaskingMHA = CausalSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, x, mask=True):
        unsqueeze_x = x.unsqueeze(-1)
        
        embed_x = self.linear_embedding(unsqueeze_x)

        if mask is True:
            seq_len = embed_x.size(-2)  # (batch, seq, embed)
            casual_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            output_mha = self.MaskingMHA(embed_x, mask=casual_mask)
        else:
            output_mha = self.MaskingMHA(embed_x)
        output = self.output_layer(output_mha).squeeze(-1)
        return output


x1 = torch.arange(3).type(torch.float32).tile(10,1)
x2 = torch.arange(5).type(torch.float32).tile(10,1)
x3 = torch.arange(8).type(torch.float32).tile(10,1)


model = MaskingMultiHeadAttention()

model(x1)
model(x2[:,:3])
model(x2, mask=True)
model(x2, mask=False)
model(x3, mask=True)
model(x3, mask=False)