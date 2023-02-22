import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionModule(nn.Module):
    def __init__(self, num_channels, num_heads=4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(batch_first=True, embed_dim=num_channels, num_heads=num_heads)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=num_channels)
        
    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x)
        y = y.reshape(b, c, h * w).permute(0, 2, 1)
        y, _ = self.attention(query=y, key=y, value=y)
        y = y.permute(0, 2, 1).reshape(b, c, h, w)
        out = x + y
        return out
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim=256, downsample=False):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.embedding_fc = nn.Linear(in_features=embedding_dim, out_features=in_channels)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.silu = nn.SiLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
    def forward(self, x, t):
        '''
        Gatekept by unreadable code HAHAHAHAHAHAHAHAHAHA
        '''
        t = self.silu(self.embedding_fc(t))
        t = t.unsqueeze(-1).unsqueeze(-1)
        
        y = self.silu(self.norm(self.conv1(x)))
        y = y + t
        y = self.conv2(y)
        y = y + x # residual connection
        out = self.silu(self.norm(y))
        
        if self.downsample:
            out = self.downsample(out) # nonlinearity here or no?
        return out
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 1000, apply_dropout: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.apply_dropout = apply_dropout

        pos_encoding = torch.zeros(max_len, embedding_dim) # [1000, 256]
        position = torch.arange(start=0, end=max_len).unsqueeze(1) # [1000, 1]
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(name='pos_encoding', tensor=pos_encoding, persistent=False)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        positional_encoding = self.pos_encoding[t].squeeze(1)
        if self.apply_dropout:
            return self.dropout(positional_encoding)
        return positional_encoding