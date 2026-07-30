import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.blocks import *

    
class RefinementBlock(nn.Module):
    def __init__(self, embedding_dim, dim, z_dim=128, rank=None):
        super().__init__()
        a = embedding_dim
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        # self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        # self.norm1 = nn.LayerNorm(8, dim)
        self.norm1 = nn.LayerNorm([dim, a, a])
        self.attn = CrossAttention(dim, z_dim, rank=rank)
        # self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        # self.norm2 = nn.GroupNorm(8, dim)
        self.norm2 = nn.LayerNorm([dim, a, a])

    def forward(self, x, z):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)

        x = self.attn(x, z)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.gelu(x)

        return x
    

class LowRankRefiner(nn.Module):
    def __init__(self, embedding_dim, in_channels=3, base_dim=64, z_dim=128, rank=None):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.GELU()
        )

        self.refine1 = RefinementBlock(embedding_dim, base_dim, z_dim, rank)
        self.refine2 = RefinementBlock(embedding_dim, base_dim, z_dim, rank)
        self.refine3 = RefinementBlock(embedding_dim, base_dim, z_dim, rank)

        self.decoder = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, in_channels, 3, padding=1)
        )

    def forward(self, x_tilde, z):
        x = self.encoder(x_tilde)

        x = self.refine1(x, z)
        x = self.refine2(x, z)
        x = self.refine3(x, z)

        x = self.decoder(x)

        return x #x_tilde + x # residual refinement
