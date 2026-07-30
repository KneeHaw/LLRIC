import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
from src.models.decoder import LowRankRefiner
from src.patch_utils import *


# Input = (B, N, h//16, w//16)
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, expand_ratio, kernel_size):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size, padding=(kernel_size-1)//2, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, kernel_size, padding=(kernel_size-1)//2, groups=1, bias=False)
        )

    def forward(self, x):
        return x + self.block(x)
    
class LRVQ(nn.Module):
    def __init__(self, rank, img_size, M, N, embedding_dim, n_embed, emb_ps=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.split_size = img_size // embedding_dim
        self.enc_dim = embedding_dim * 3 * (2*rank)
        self.rank = rank
        self.im_size = img_size
        # patch_dim = n_hid * embedding_dim ** 2
        
        # Patch transformer
        self.emb = emb_ps
        # self.embed_patches = EmbedPatches(self.split_size, embedding_dim, 3 * embedding_dim ** 2, n_hid)
        self.proj_patches = nn.Sequential(
            nn.Linear(N*16*3//2, N*12),
            nn.LayerNorm(N*12),
            nn.Linear(N*12, N*8),
            nn.LayerNorm(N*8),
        )
        # self.entropy_trans = Transformer(M, 128, num_layers=2)
        self.transformer = Transformer(N*8, self.enc_dim, num_layers=2)
        self.patch_trans = LowRankRefiner(embedding_dim)
        ##### Vector quantization
        self.quantizer = VQVAEQuantize(n_embed, embedding_dim)
        
        # self.smooth_conv = nn.Sequential(*[ResidualBlock(3, 3, 5)] * 3)
                
                
    def forward(self, x):
        # print(x.shape)
        z_emb = self.proj_patches(x)
        # print(z_emb.shape)
        z_r = self.transformer(z_emb)
        
        # print(z_r.shape)
        # exit()
        ##### Vector quantization
        s_x = z_r.shape[1]
        z_q = rearrange(z_r, "b s (enc_emb emb) -> b (s enc_emb) emb", s = s_x, 
                        emb = self.embedding_dim, enc_emb = self.enc_dim // self.embedding_dim)
        z_q, latent_loss, ind = self.quantizer(z_q)  # Map vectors to their nearest size
      
        ##### Low-Rank Reconstruction
        z_q = rearrange(z_q, "b (s inc rank) emb -> (b s) inc rank emb",
                        rank = 2*self.rank, inc = 3, emb = self.embedding_dim)
        x_hat = reconstruct(z_q, 1, self.rank, 3)

        # Refinement
        # z_u = self.entropy_trans(z_emb).permute(1, 0, 2)
        # z_u = z_u.reshape(-1, z_u.shape[2])
        # x_hat = self.patch_trans(x_hat, z_u)

        # Spatial placement
        x_hat = torch.cat(torch.chunk(x_hat, self.im_size // self.embedding_dim, 0), -2)
        x_hat = torch.cat(torch.chunk(x_hat, self.im_size // self.embedding_dim, 0), -1)
        
        x_hat = x_hat.clamp(0.0, 1.0) 
        # x_hat = x_hat / torch.max(x_hat)
        return x_hat, latent_loss, ind
    
class PatchTrans(nn.Module):
    def __init__(self, args, split_size, input_channels, embedding_dim=1):
        super().__init__()
        self.args = args
        self.in_c = input_channels
        self.embedding_dim=embedding_dim
        patch_dim = input_channels * embedding_dim ** 2
        
        self.embed_patches = EmbedPatches(split_size, embedding_dim, patch_dim, args.n_hid)
        
        ##### Encoding transformer
        self.enc_dim = embedding_dim * args.iterations * (2*args.rank) * input_channels
        self.rank_transformer = Transformer(2*args.n_hid, self.enc_dim, num_layers=1 if args.n_hid == 64 else 2)   # h/w to 4, 4
        
        ##### Vector quantization
        
    def forward(self, x):
        ##### Patch embeddings
        z = self.embed_patches(x)  
        
        ##### Encoding transformer
        z_r = self.rank_transformer(z)  # -> (B, num_patches, enc_dim)
        
        return z_r, z
    
class Transformer(nn.Module):
    """
    A general Transformer model designed to process data that's already positionally embedded.
    """
    def __init__(self,
                 input_channels: int,
                 transformer_output_channels: int, # Output channels from the transformer block
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 2048, # Dimension of the feedforward network model in TransformerEncoderLayer
                 dropout: float = 0.1):      # Dropout value
        """
        Args:
            input_channels (int): The number of channels in the input tensor
            num_heads (int): The number of attention heads in the Transformer's multi-head attention mechanism.
                             `input_channels` must be divisible by `num_heads`.
            num_layers (int): The number of Transformer Encoder layers to stack.
            transformer_output_channels (int): The desired number of channels in the output tensor
                                               *after* the Transformer Encoder.
            dim_feedforward (int): The dimension of the feedforward network model in the
                                   TransformerEncoderLayer. Defaults to 2048.
            dropout (float): The dropout value for the TransformerEncoderLayer. Defaults to 0.1.
        """
        super().__init__()

        # d_model is the dimension of the input features for the Transformer.
        self.d_model = input_channels
        self.transformer_output_channels = transformer_output_channels

        # Validate that d_model is divisible by num_heads, a requirement for multi-head attention.
        if self.d_model % num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({num_heads})")

        #    nn.TransformerEncoderLayer represents a single block of self-attention
        #    and feed-forward networks.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # PyTorch's Transformer modules typically expect (sequence_length, batch_size, d_model)
                              # by default. We will handle the permutation in the forward pass.
        )
        # nn.TransformerEncoder stacks multiple encoder_layer instances.
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Identity() # Default to no-op if channels are the same
        if self.d_model != transformer_output_channels:
            self.output_projection = nn.Linear(self.d_model, self.transformer_output_channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the SpatialTransformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).
                              Example: (batch_size, 512, 8, 8).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, height, width, transformer_output_channels).
                          Example: (batch_size, 8, 8, 512) or (batch_size, 8, 8, 256).
        """
        # Permute (B, Seq_Len, d_model) -> (Seq_Len, B, d_model)
        # x = x.permute(1, 0, 2)

        # Pass through the Transformer Encoder.
        transformer_output = self.transformer_encoder(x)

        # Apply output projection if transformer_output_channels is different
        # Output dim is (Seq_Len, B, d_model)
        projected_output = self.output_projection(transformer_output)

        return projected_output
    
    
class EmbedPatches(nn.Module):
    def __init__(self, split_size, embedding_dim, patch_dim, n_hid):
        super().__init__()
        print(f"Patches = {split_size} ({embedding_dim}, {embedding_dim})")
        self.to_patch_embedding = nn.Sequential(
            # ,
            # nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, n_hid),   
            # nn.LayerNorm(2*n_hid),
        )

        self.pos_embedding = self.posemb_sincos_2d(
            h = split_size,
            w = split_size,
            dim = n_hid,
        ) 
        
    @staticmethod
    def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)

    @staticmethod
    def add_b_to_a(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Adds elements of B to A such that each element of B corresponds
        to a block of A elements (e.g., 4×4 mapping when sqrt(len(A)/len(B)) = 4).

        Both A and B are assumed to be 1D tensors.

        Example:
            A: length 4096  -> 64×64
            B: length 256   -> 16×16
        """
        # Infer the 2D sizes
        size_a = int(A.numel() ** 0.5)
        size_b = int(B.numel() ** 0.5)

        # print(A.size(), B.size())
        assert size_a % size_b == 0, "A's size must be evenly divisible by B's size"

        block = size_a // size_b  # how many A elements per B element along one dim

        # Reshape into 2D
        A2 = A.view(size_a, size_a)
        B2 = B.view(size_b, size_b)

        # Expand B to A's size via nearest neighbor upsampling
        B_up = B2.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)

        # Add and flatten back
        A_out = A2 + B_up
        return A_out.flatten()

    def forward(self, x):
        # print(x.shape)
        patches = self.to_patch_embedding(x)
        pos_patches = patches + self.pos_embedding.to(patches.device)
        return pos_patches
    
    
class VQVAEQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, n_embed, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.kld_scale = 10.0
        self.embed = nn.Embedding(n_embed, embedding_dim)
        init.kaiming_uniform_(self.embed.weight)
        self.test = True

    def forward(self, z):
        B, C, D = z.size() 
        assert D == self.embedding_dim, print("D vs emb dim: ", D, self.embedding_dim)
        z_flat = z.reshape(-1, D)
        
        # if random.random() < 0.1:
        #     if self.test:
        #         torch.save(self.embed.weight, "./temp_data/svd16_codebook.pt")
        #         self.test = False
        #     append_tensor_to_npy(z_flat, "./temp_data/svd16_vectors.npy")

        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        _, ind = (-dist).max(1)
        ind = ind.view(B, C)

        # vector quantization cost that trains the embedding vectors
        z_q = self.embed_code(ind) # (B, H, W, C)
        commitment_cost = 0.25
        diff = commitment_cost * (z_q.detach() - z).pow(2).mean() + (z_q - z.detach()).pow(2).mean()
        diff *= self.kld_scale

        z_q = z + (z_q - z).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        return z_q, diff, ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)
    
    
def reconstruct(x: Tensor, iterations: int, rank: int, in_c: int) -> Tensor:
    # x = torch.stack(x.chunk(in_c, 2), 2)
    if iterations == 1:
        left_tens = x[:, :, :rank, :].transpose(-1, -2)
        right_tens = x[:, :, rank:, :]
        lr_approx = left_tens @ right_tens
        return lr_approx
    else:
        raise NotImplementedError("Fix rank reconstruct with iter")
    x_iters = x.split(1, 0)
    output = []
    # iter_tens size -> (Batch, Chann, 2*rank, img_H/W)
    for iter_tens in x_iters:
        iter_tens = iter_tens.squeeze()
        # if len(iter_tens.size()) < 4:
        #     iter_tens = iter_tens.unsqueeze(1)
        left_tens = iter_tens[:rank, :].transpose(0, 1)
        right_tens = iter_tens[rank:, :]
        
        print(x.shape, iter_tens.shape, left_tens.shape, right_tens.shape)
        exit()
        lr_approx = left_tens @ right_tens
        output.append(lr_approx)
    output = torch.stack(output)
    return output