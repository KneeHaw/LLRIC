import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankLinear(nn.Module):
    """
    Implements W = A @ B^T
    where rank << min(in_dim, out_dim)
    """
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # x: (..., in_dim)
        W = self.A @ self.B.T  # (in_dim, out_dim)
        return x @ W + self.bias  # (..., out_dim)
    
class CrossAttention(nn.Module):
    def __init__(self, dim, z_dim=128, num_heads=4, rank=None):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        if rank is None:
            # Regular Projections
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
        else:
            # Low-rank projections
            self.q_proj = LowRankLinear(dim, dim, rank)
            self.k_proj = LowRankLinear(dim, dim, rank)
            self.v_proj = LowRankLinear(dim, dim, rank)
            self.out_proj = LowRankLinear(dim, dim, rank)
        
        self.n_token = 8
        # Expand z into tokens
        self.z_mlp = nn.Sequential(
            nn.Linear(z_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * self.n_token)
        )

    def forward(self, x, z):
        """
        x: (B, C, H, W)
        z: (B, z_dim)
        """

        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # (B, N, C)

        # Expand z into tokens
        z_tokens = self.z_mlp(z)  # (B, 4*dim)
        z_tokens = z_tokens.view(B, self.n_token, C)  # (B, T=4, C)

        # Project
        Q = self.q_proj(x_flat)
        K = self.k_proj(z_tokens)
        V = self.v_proj(z_tokens)

        # Reshape heads
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, self.n_token, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, self.n_token, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, V)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)

        # Restore spatial
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return out  # residual refinement
    
# 3-1-3 conv series
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, expand_ratio, kernel_size):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size, padding=(kernel_size-1)//2, groups=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, kernel_size, padding=(kernel_size-1)//2, groups=1, bias=False)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))
    
    
class SigmoidMask(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
    
        self.mask_layer = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x, T):
        mask = F.sigmoid(self.mask_layer(x) / T)
        return x * mask