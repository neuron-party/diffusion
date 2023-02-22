# model used by the authors of ddpm paper
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ddpm_modules import *


class DDPM_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.positional_embedding = PositionalEmbedding(embedding_dim=128)
        self.embedding_linear = nn.Sequential(
            nn.Linear(in_features=128, out_features=512),
            nn.Linear(in_features=512, out_features=512)
        )

        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.down1 = nn.ModuleList([
            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),
            Downsample(in_channels=128)
        ]) # [[b, 128, 128, 128], [b, 128, 128, 128], [b, 128, 64, 64]]

        self.down2 = nn.ModuleList([
            ResidualBlock(in_channels=128, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256),
            Downsample(in_channels=256)
        ]) # [[b, 256, 64, 64], [256, 64, 64], [256, 32, 32]]

        self.down3 = nn.ModuleList([
            ResidualBlock(in_channels=256, out_channels=512),
            ResidualBlock(in_channels=512, out_channels=512),
            Downsample(in_channels=512)
        ]) # [[b, 512, 32, 32], [b, 512, 32, 32], [b, 512, 16, 16]]

        self.down4 = nn.ModuleList([
            ResidualBlock(in_channels=512, out_channels=1024),
            ResidualBlock(in_channels=1024, out_channels=1024)
        ]) # [[b, 1024, 16, 16], [b, 1024, 16, 16]]

        self.middle_blocks = nn.ModuleList([
            ResidualBlock(in_channels=1024, out_channels=1024),
            AttentionModule(num_channels=1024),
            ResidualBlock(in_channels=1024, out_channels=1024)
        ])

        self.up1 = nn.ModuleList([
            ResidualBlock(in_channels=2048, out_channels=1024),
            ResidualBlock(in_channels=2048, out_channels=1024),
            ResidualBlock(in_channels=1536, out_channels=1024),
            Upsample(in_channels=1024)
        ])

        self.up2 = nn.ModuleList([
            ResidualBlock(in_channels=1536, out_channels=512),
            ResidualBlock(in_channels=1024, out_channels=512),
            ResidualBlock(in_channels=768, out_channels=512),
            Upsample(in_channels=512)
        ])

        self.up3 = nn.ModuleList([
            ResidualBlock(in_channels=768, out_channels=256),
            ResidualBlock(in_channels=512, out_channels=256),
            ResidualBlock(in_channels=384, out_channels=256),
            Upsample(in_channels=256)
        ])

        self.up4 = nn.ModuleList([
            ResidualBlock(in_channels=384, out_channels=128),
            ResidualBlock(in_channels=256, out_channels=128),
            ResidualBlock(in_channels=256, out_channels=128)
        ])

        self.down_blocks = [self.down1, self.down2, self.down3, self.down4]
        self.up_blocks = [self.up1, self.up2, self.up3, self.up4]

    def forward(self, x, t):
        t = self.positional_embedding(t)
        t = self.embedding_linear(t)

        feature_maps = []
        x = self.initial_conv(x)
        feature_maps.append(x)

        for down_block in self.down_blocks:
            for layer in down_block:
                x = layer(x, t)
                feature_maps.append(x)

        assert len(feature_maps) == 12
        # return feature_maps
        for layer in self.middle_blocks:
            x = layer(x, t)

        for up_block in self.up_blocks:
            for i, layer in enumerate(up_block):
                if i < 3: # 3 residual blocks, last layer in upsample if exists
                    x_skip = feature_maps.pop()
                    x = torch.cat([x_skip, x], dim=1)
                x = layer(x, t)

        x = self.final_conv(x)
        return x