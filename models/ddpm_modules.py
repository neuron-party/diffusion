import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            dropout: float = 0.0,
            max_len: int = 1000,
            apply_dropout: bool = True,
    ):
        """Section 3.5 of attention is all you need paper.
        Extended slicing method is used to fill even and odd position of sin, cos with increment of 2.
        Ex, `[sin, cos, sin, cos, sin, cos]` for `embedding_dim = 6`.
        `max_len` is equivalent to number of noise steps or patches. `embedding_dim` must same as image
        embedding dimension of the model.
        Args:
            embedding_dim: `d_model` in given positional encoding formula.
            dropout: Dropout amount.
            max_len: Number of embeddings to generate. Here, equivalent to total noise steps.
        """
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
        """Get precalculated positional embedding at timestep t. Outputs same as video implementation
        code but embeddings are in [sin, cos, sin, cos] format instead of [sin, sin, cos, cos] in that code.
        Also batch dimension is added to final output.
        """
        positional_encoding = self.pos_encoding[t].squeeze(1)
        if self.apply_dropout:
            return self.dropout(positional_encoding)
        return positional_encoding


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, *args):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x
    

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x, *args):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, pad=2)
        return x
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=False, embedding_dim=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.conv_shortcut = conv_shortcut
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-6, affine=True)
        
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        
        self.embedding_linear = nn.Linear(embedding_dim, self.out_channels)
        
        if self.in_channels != self.out_channels:
            if self.conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)
        
        self.nonlinearity = nn.SiLU()
        
    def forward(self, x, t):
        t = self.nonlinearity(t)
        t = self.embedding_linear(t)
        t = t.unsqueeze(-1).unsqueeze(-1)
        
        y = x
        y = self.norm1(y)
        y = self.nonlinearity(y)
        y = self.conv1(y)
        y = y + t
        y = self.norm2(y)
        y = self.nonlinearity(y)
        y = self.conv2(y)
        
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.conv_shortcut else self.nin_shortcut(x)
        
        x = x + y
        return x
    
    
class AttentionModule(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        self.q_projection = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0)
        self.k_projection = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0)
        self.v_projection = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0)
        self.final_projection = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0)
        
        self.norm = nn.GroupNorm(num_groups=32, num_channels=num_channels)
        self.attention = nn.MultiheadAttention(num_heads=1, embed_dim=num_channels, batch_first=True)
    
    def forward(self, x, *args):
        b, c, h, w = x.shape
        y = x
        y = self.norm(y)
        q, k, v = self.q_projection(y), self.k_projection(y), self.v_projection(y)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w).permute(0, 2, 1)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        y, _ = self.attention(query=q, key=k, value=v)
        y = y.permute(0, 2, 1).reshape(b, c, h, w)
        y = self.final_projection(y)
        y = y + x
        return y