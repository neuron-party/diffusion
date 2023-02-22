# made my own smaller unet thats still pretty good
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from peep_modules import *


class Peep_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.positional_embedding = PositionalEmbedding(embedding_dim=256)
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # downsampling
        self.residual_1 = ResidualBlock(in_channels=64, out_channels=128, downsample=True) # [b, 64, h, w] -> [b, 128, h/2, w/2] 64
        self.residual_2 = ResidualBlock(in_channels=128, out_channels=256, downsample=True) # [b, 128, h/2, w/2] -> [b, 256, h/4, w/4] 32
        self.residual_3 = ResidualBlock(in_channels=256, out_channels=512, downsample=True) # [b, 256, h/4, w/4] -> [b, 512, h/8, w/8] 16
        self.attention_1 = AttentionModule(num_channels=512)
        self.residual_4 = ResidualBlock(in_channels=512, out_channels=1024, downsample=True) # [b, 512, h/8, w/8] -> [b, 1024, h/16, w/16]
        self.attention_2 = AttentionModule(num_channels=1024)
        
        # middle
        self.residual_5 = ResidualBlock(in_channels=1024, out_channels=1024)
        self.attention_3 = AttentionModule(num_channels=1024)
        self.residual_6 = ResidualBlock(in_channels=1024, out_channels=1024)
        
        # upsample
        self.upsample_1 = nn.PixelShuffle(upscale_factor=2)
        self.residual_7 = ResidualBlock(in_channels=512, out_channels=512)
        self.attention_4 = AttentionModule(num_channels=512)
        self.upsample_2 = nn.PixelShuffle(upscale_factor=2)
        self.residual_8 = ResidualBlock(in_channels=256, out_channels=256)
        self.upsample_3 = nn.PixelShuffle(upscale_factor=2)
        self.residual_9 = ResidualBlock(in_channels=128, out_channels=128)
        self.upsample_4 = nn.PixelShuffle(upscale_factor=2)
        self.residual_10 = ResidualBlock(in_channels=64, out_channels=64)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        )
        
    def forward(self, x, t):
        '''
        alternate design choices:
            apply convolution and attention on the catted tensors during the upsampling stage rather than after the upsample layer
            i.e cat  -> residual block -> attention -> upsample
            
            apply convolution and attention before the concatenation step
            i.e residual block -> attention -> cat -> upsample
            
            switch order of attention and convolution on the upsampling phase for symmetry?
        '''
        t = self.positional_embedding(t)
        
        x1 = self.initial_conv(x) # [64, 128, 128]
        x2 = self.residual_1(x1, t)  # [128, 64, 64]
        x3 = self.residual_2(x2, t)  # [256, 32, 32]
        x4 = self.residual_3(x3, t)  # [512, 16, 16]
        x4 = self.attention_1(x4) # [512, 16, 16]
        x5 = self.residual_4(x4, t)  # [1024, 8, 8]
        x5 = self.attention_2(x5) # [1024, 8, 8]
        
        # middle
        x = self.residual_5(x5, t)
        x = self.attention_3(x)
        x = self.residual_6(x, t) 
        
        # upsample
        x = torch.cat([x5, x], dim=1)  # [2048, 8, 8]
        x = self.upsample_1(x)         # [512, 16, 16]
        x = self.residual_7(x, t)      # [512, 16, 16]
        x = self.attention_4(x)        # [512, 16, 16]
        x = torch.cat([x4, x], dim=1)  # [1024, 16, 16]
        x = self.upsample_2(x)         # [256, 32, 32]
        x = self.residual_8(x, t)      # [256, 32, 32]
        x = torch.cat([x3, x], dim=1)  # [512, 32, 32]
        x = self.upsample_3(x)         # [128, 64, 64]
        x = self.residual_9(x, t)         # [128, 64, 64]
        x = torch.cat([x2, x], dim=1)  # [256, 64, 64]
        x = self.upsample_4(x)         # [64, 128, 128]
        x = self.residual_10(x, t)        # [64, 128, 128]
        x = torch.cat([x1, x], dim=1)  # [128, 128, 128]
        
        # final convs
        x = self.final_conv(x)
        return x