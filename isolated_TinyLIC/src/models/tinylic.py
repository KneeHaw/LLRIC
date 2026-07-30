import math
import numpy as np

import torch
import torch.nn as nn

from src.ans import BufferedRansEncoder, RansDecoder
from src.entropy_models import EntropyBottleneck, GaussianConditional
from src.layers import ResViTBlock, MultistageMaskedConv2d
from timm.layers import trunc_normal_
from src.models.llric_block import LRVQ
from src.patch_utils import *
from einops.layers.torch import Rearrange
import time
from .utils import conv, deconv, update_registered_buffers, quantize_ste, \
    Demultiplexer, Multiplexer, Demultiplexerv2, Multiplexerv2

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class TinyLIC(nn.Module):
    r"""Lossy image compression framework from Ming Lu and Zhan Ma
    "High-Efficiency Lossy Image Coding Through Adaptive Neighborhood Information Aggregation"

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=128, M=320, args=None):
        super().__init__()

        depths = [2, 2, 6, 2, 2, 2]
        num_heads = [8, 12, 16, 20, 12, 12]
        kernel_size = 7
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        self.num_iters = 4
        self.gamma = self.gamma_func(mode="cosine")
        self.M = M
        self.args = args
        
        self.timers = {"fwd_total": 0.0,
                       "bwd_total": 0.0,
                       "encode": 0.0,
                       "entropy": 0.0,
                       "decode": 0.0,
                       "llric": 0.0}

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        ########################################
        ##### Create g_a latent encoder
        ########################################
        g_a_dims = [3, N, N*3//2, N*2, M]
        g_a_kernels = [5, 3, 3, 3]
        g_a_strides = [2, 1, 1, 1]
        self.g_a = nn.ModuleList([None] * 8)
        for i in range(3+1):
            self.g_a[i*2] = conv(g_a_dims[i], g_a_dims[i+1], 
                                 kernel_size=g_a_kernels[i], stride=g_a_strides[i])
            self.g_a[i*2 + 1] = ResViTBlock(dim=g_a_dims[i+1],
                                depth=depths[i],
                                num_heads=num_heads[i],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i+1])],
                                norm_layer=norm_layer,
            )

        ########################################
        ##### Create h_a hyperprior encoder
        ########################################
        h_a_dims = [M, N*3//2, N*3//2]
        h_a_kernels = [3, 3]
        h_a_strides = [2, 2]
        self.h_a = nn.ModuleList([None] * 4)
        for i in range(2):
            self.h_a[i*2] = conv(h_a_dims[i], h_a_dims[i+1], 
                                 kernel_size=h_a_kernels[i], stride=h_a_strides[i])
            self.h_a[i*2 + 1] = ResViTBlock(dim=h_a_dims[i+1],
                                depth=depths[i+4],
                                num_heads=num_heads[i+4],
                                kernel_size=kernel_size//2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:i+4]):sum(depths[:i+5])],
                                norm_layer=norm_layer,
            )
            
            
        depths = depths[::-1]
        num_heads = num_heads[::-1]

        ########################################
        ##### Create h_s hyperprior decoder
        ########################################
        h_s_dims = [N*3//2, N*3//2, M*2]
        h_s_kernels = [3, 3]
        h_s_strides = [2, 2]
        self.h_s = nn.ModuleList([None] * 4)
        for i in range(2):
            self.h_s[i*2] = ResViTBlock(dim=h_s_dims[i],
                                depth=depths[i],
                                num_heads=num_heads[i],
                                kernel_size=kernel_size//2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i+1])],
                                norm_layer=norm_layer,
            )
            self.h_s[i*2 + 1] = deconv(h_s_dims[i], h_s_dims[i+1], 
                                       kernel_size=h_s_kernels[i], stride=h_s_strides[i])

        ########################################
        ##### Create g_s decoder
        ########################################
        g_s_dims = [M, N*2, N*3//2, N, 3]
        g_s_kernels = [3, 3, 3, 5]
        g_s_strides = [1, 1, 1, 2]
        self.g_s = nn.ModuleList([None] * 8)
        for i in range(4):
            # print(f"Creating g_s block {i} with dim {g_s_dims[i]} and {depths[i]} layers, and {num_heads[i+2]} heads.")
            self.g_s[i*2] = ResViTBlock(dim=g_s_dims[i],
                                depth=depths[i+2],
                                num_heads=num_heads[i+2],
                                kernel_size=kernel_size//2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:i+2]):sum(depths[:i+3])],
                                norm_layer=norm_layer,
            )
            self.g_s[i*2 + 1] = deconv(g_s_dims[i], g_s_dims[i+1], 
                                       kernel_size=g_s_kernels[i], stride=g_s_strides[i])

        self.register_buffer("lambda_weight", torch.tensor(1.0))
        
        self.entropy_bottleneck = EntropyBottleneck(N*3//2)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.sc_transforms = nn.ModuleList([None] * 5)
        kernel_sizes = [3, 5, 5, 5, 5]
        paddings = [1, 2, 2, 2, 2]
        masks = ['A', 'B', 'C', 'B', 'B']
        for i in range(5):
            add_dim = M if i < 3 else np.ceil(self.gamma((i - 2)/self.num_iters) * M)
            sub_dim = np.ceil(self.gamma((1 if i < 3 else i-1)/self.num_iters)*M)
            int_dim = int(add_dim - sub_dim)
            self.sc_transforms[i] = MultistageMaskedConv2d(
                    int_dim, int_dim*2, 
                    kernel_size=kernel_sizes[i], padding=paddings[i], stride=1, mask_type=masks[i]
            )

        self.entropy_parameters = nn.ModuleList([None] * 4)
        for i in range(4):
            sub_dim = np.ceil(self.gamma(min(i+1, 3)/self.num_iters)*M)
            sub_dim = np.ceil(self.gamma(0)*M) - sub_dim if i == 3 else sub_dim
            add_dim = (M if (i == 0 or i == 3) else np.ceil(self.gamma((i)/self.num_iters)*M))
            int_dim = int(add_dim - sub_dim)
            scalar = [12, 10, 8, 6] if i < 3 else [6, 6, 6, 6]
            self.entropy_parameters[i] = nn.Sequential(
                conv(M*2 + int_dim * scalar[0]//3, int_dim*scalar[1]//3, 1, 1),
                nn.GELU(),
                conv(int_dim*scalar[1]//3, int_dim*scalar[2]//3, 1, 1),
                nn.GELU(),
                conv(int_dim*scalar[2]//3, int_dim*scalar[3]//3, 1, 1),
        )
            
        # exit()
            
        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M*2 + int(M - np.ceil(self.gamma(i/self.num_iters)*M)), 
                int(np.ceil(self.gamma(i/self.num_iters)*M)-np.ceil(self.gamma((i+1)/self.num_iters)*M))
                if i != self.num_iters-1 else int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(i/self.num_iters)*M))),
                stride=1, kernel_size=5),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(i/self.num_iters)*M)-np.ceil(self.gamma((i+1)/self.num_iters)*M))
                if i != self.num_iters-1 else int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(i/self.num_iters)*M))), 
                int(np.ceil(self.gamma(i/self.num_iters)*M)-np.ceil(self.gamma((i+1)/self.num_iters)*M))
                if i != self.num_iters-1 else int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(i/self.num_iters)*M))), 
                stride=1, kernel_size=5),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(i/self.num_iters)*M)-np.ceil(self.gamma((i+1)/self.num_iters)*M))
                if i != self.num_iters-1 else int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(i/self.num_iters)*M))), 
                int(np.ceil(self.gamma(i/self.num_iters)*M)-np.ceil(self.gamma((i+1)/self.num_iters)*M))*2
                if i != self.num_iters-1 else int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(i/self.num_iters)*M)))*2, 
                stride=1, kernel_size=3),
            ) for i in range(self.num_iters)
        )

        self.image_size = 64
        self.patch_size = 32
        self.patch_tensor_llric = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = self.patch_size, p2 = self.patch_size)
        self.patch_tensor_tiny = Rearrange("b c (h p1) (w p2) -> (b h w) c p1 p2", p1 = self.patch_size, p2 = self.patch_size)
        self.recons_tensor_tiny = Rearrange("(b h w) c p1 p2 -> b c (h p1) (w p2)", h=self.image_size//self.patch_size, w=self.image_size//self.patch_size, p1 = self.patch_size, p2 = self.patch_size)
        self.llric_blk = LRVQ(rank=4, img_size=self.image_size, M=M, N=N, embedding_dim=self.patch_size, n_embed=8192, emb_ps=False)
        self.lr_weight = nn.Parameter(torch.ones(1))

        self.apply(self._init_weights)  
        
    def update_bwd_t(self, elapsed_t):
        self.timers["bwd_total"] += elapsed_t
        
    def print_timers(self):
        fwd_t = self.timers["fwd_total"]
        bwd_t = self.timers["bwd_total"]
        
        frac_times = {}
        for key, value in self.timers.items():
            if key in ["fwd_total", "bwd_total"]: continue
            frac_times[key] = value / fwd_t
        sorted_dict = dict(sorted(frac_times.items(), key=lambda item: item[1], reverse=True))
        
        print("----- Relative Timing of Model Components -----")
        print(f"fwd_total ➡ {fwd_t/(fwd_t + bwd_t):.4f}")
        for key, value in sorted_dict.items(): print(f"\t↪  {key:<10} ➡ {value:.4f}")
        print(f"bwd_total ➡ {bwd_t/(fwd_t + bwd_t):.4f}")

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r**2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError  

    def g_a_fwd(self, x, lrvq_context=None):
        y = self.g_a[0](x)
        for i, layer in enumerate(self.g_a[1:]):
            if i == 0 and lrvq_context is not None:
                y = self.lrvq_context_fuse(torch.cat([x, lrvq_context], dim=1))
            y = layer(y)
        return y

    def g_s_fwd(self, x):
        y = x
        for i, layer in enumerate(self.g_s):
            try:
                y = layer(y)
            except Exception as e:
                print(f"Error occurred while processing g_s layer {i}: {e}")
                exit()
        return y

    def h_a_fwd(self, x):
        y = x
        for i, layer in enumerate(self.h_a):
            try:
                y = layer(y)
            except Exception as e:
                print(f"Error occurred while processing h_a layer {i}: {e}")
                exit()
        return y

    def h_s_fwd(self, x):
        y = x
        for i, layer in enumerate(self.h_s):
            try:
                y = layer(y)
            except Exception as e:
                print(f"Error occurred while processing h_s layer {i}: {e}")
                exit()
        return y

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck module(s)."""
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward(self, x):
        start_t = time.perf_counter_ns()
        
        patches_tiny = self.patch_tensor_tiny(x)
        y = self.g_a_fwd(patches_tiny) #, lrvq_context)
        z = self.h_a_fwd(y)
        # patches_llric = self.patch_tensor_llric(x)

        llric_t = time.perf_counter_ns()
        lr_recons, latent_loss, _ = self.llric_blk(z.view(z.size(0), -1))
        
        # exit()
        self.timers["llric"] += llric_t - start_t

        
        exit()
        encode_t = time.perf_counter_ns()
        self.timers["encode"] += encode_t - llric_t
        
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset
        
        params = self.h_s_fwd(z_hat)
        
        slice_list = []
        N = y.shape[1]
        for t in range(1, self.num_iters+1):
            if t == self.num_iters:
                n = 0
            else:
                n = np.ceil(self.gamma(t/(self.num_iters)) * y.shape[1])

            slice_list.append(int(N-n))
            N = n

        y_slices = y.split(tuple(slice_list), 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):

            if slice_index == 0:
                support_slices = torch.cat([params] + y_hat_slices, dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                sc_params_1 = torch.zeros_like(cc_params)
                sc_params = sc_params_1
                gaussian_params = self.entropy_parameters[0](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_0 = y_hat_slice.clone()
                y_0[:, :, 0::2, 1::2] = 0
                y_0[:, :, 1::2, :] = 0
                sc_params_2 = self.sc_transforms[0](y_0)
                sc_params_2[:, :, 0::2, :] = 0
                sc_params_2[:, :, 1::2, 0::2] = 0
                sc_params = sc_params_1 + sc_params_2
                gaussian_params = self.entropy_parameters[0](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_1 = y_hat_slice.clone()
                y_1[:, :, 0::2, 1::2] = 0
                y_1[:, :, 1::2, 0::2] = 0
                sc_params_3 = self.sc_transforms[1](y_1)
                sc_params_3[:, :, 0::2, 0::2] = 0
                sc_params_3[:, :, 1::2, :] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3
                gaussian_params = self.entropy_parameters[0](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_2 = y_hat_slice.clone()
                y_2[:, :, 1::2, 0::2] = 0
                sc_params_4 = self.sc_transforms[2](y_2)
                sc_params_4[:, :, 0::2, :] = 0
                sc_params_4[:, :, 1::2, 1::2] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3 + sc_params_4
                gaussian_params = self.entropy_parameters[0](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat
                y_hat_slices.append(y_hat_slice)

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat, means=means_hat)
                y_likelihood.append(y_slice_likelihood)

            elif slice_index == 1:
                support_slices = torch.cat([params] + y_hat_slices, dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                sc_params = torch.zeros_like(cc_params)
                gaussian_params = self.entropy_parameters[1](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_half = y_hat_slice.clone()
                y_half[:, :, 0::2, 0::2] = 0
                y_half[:, :, 1::2, 1::2] = 0

                sc_params = self.sc_transforms[3](y_half)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0
                
                gaussian_params = self.entropy_parameters[1](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat
                y_hat_slices.append(y_hat_slice)

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat, means=means_hat)
                y_likelihood.append(y_slice_likelihood)

            elif slice_index == 2:
                support_slices = torch.cat([params] + y_hat_slices, dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                sc_params = torch.zeros_like(cc_params)
                gaussian_params = self.entropy_parameters[2](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_half = y_hat_slice.clone()
                y_half[:, :, 0::2, 1::2] = 0
                y_half[:, :, 1::2, 0::2] = 0

                sc_params = self.sc_transforms[4](y_half)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0

                gaussian_params = self.entropy_parameters[2](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat
                y_hat_slices.append(y_hat_slice)

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat, means=means_hat)
                y_likelihood.append(y_slice_likelihood)

            else:
                support_slices = torch.cat([params] + y_hat_slices, dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                gaussian_params = self.entropy_parameters[3](
                    torch.cat((params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat
                y_hat_slices.append(y_hat_slice)

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat, means=means_hat)
                y_likelihood.append(y_slice_likelihood)

        entropy_t = time.perf_counter_ns()
        self.timers["entropy"] += entropy_t - encode_t

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s_fwd(y_hat)

        decode_t = time.perf_counter_ns()
        self.timers["decode"] += decode_t - entropy_t

        # print(x_hat.shape, lr_recons.shape)
        x_hat = self.recons_tensor_tiny(x_hat)
        output = (x_hat * (1-self.lr_weight) + lr_recons*self.lr_weight).clamp(0.0, 1.0)

        self.timers["fwd_total"] += time.perf_counter_ns() - start_t
        return {
            "x_hat": output,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "vq_loss": latent_loss,
            "sampled_lr": None,
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a[0].weight"].size(0)
        M = state_dict["g_a[6].weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        slice_list = []
        N = y.shape[1]
        for t in range(1, self.num_iters+1):
            if t == self.num_iters:
                n = 0
            else:
                n = np.ceil(self.gamma(t/(self.num_iters)) * y.shape[1])

            slice_list.append(int(N-n))
            N = n

        y_slices = y.split(tuple(slice_list), 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            
            if slice_index == 0:
                cc_params = self.cc_transforms[slice_index](params)

                y_slice_0, y_slice_1, y_slice_2, y_slice_3 = Demultiplexerv2(y_slice)
                
                sc_params_1 = torch.zeros(y_slice.shape[0], y_slice.shape[1]*2, y_slice.shape[2], y_slice.shape[3]).to(z_hat.device)
                sc_params = sc_params_1
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                scales_hat_0, _, _, _ = Demultiplexerv2(scales_hat)
                means_hat_0, _, _, _ = Demultiplexerv2(means_hat)
                index_0 = self.gaussian_conditional.build_indexes(scales_hat_0)
                y_q_slice_0 = self.gaussian_conditional.quantize(y_slice_0, "symbols", means_hat_0)
                y_hat_slice_0 = y_q_slice_0 + means_hat_0

                symbols_list.extend(y_q_slice_0.reshape(-1).tolist())
                indexes_list.extend(index_0.reshape(-1).tolist())

                y_hat_slice = Multiplexerv2(y_hat_slice_0, torch.zeros_like(y_hat_slice_0), 
                                            torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))
                sc_params_2 = self.sc_transform_1(y_hat_slice)
                sc_params_2[:, :, 0::2, :] = 0
                sc_params_2[:, :, 1::2, 0::2] = 0
                sc_params = sc_params_1 + sc_params_2
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_1, _, _ = Demultiplexerv2(scales_hat)
                _, means_hat_1, _, _ = Demultiplexerv2(means_hat)
                index_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
                y_q_slice_1 = self.gaussian_conditional.quantize(y_slice_1, "symbols", means_hat_1)
                y_hat_slice_1 = y_q_slice_1 + means_hat_1

                symbols_list.extend(y_q_slice_1.reshape(-1).tolist())
                indexes_list.extend(index_1.reshape(-1).tolist())

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                            torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))
                sc_params_3 = self.sc_transform_2(y_hat_slice)
                sc_params_3[:, :, 0::2, 0::2] = 0
                sc_params_3[:, :, 1::2, :] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, _, scales_hat_2, _ = Demultiplexerv2(scales_hat)
                _, _, means_hat_2, _ = Demultiplexerv2(means_hat)
                index_2 = self.gaussian_conditional.build_indexes(scales_hat_2)
                y_q_slice_2 = self.gaussian_conditional.quantize(y_slice_2, "symbols", means_hat_2)
                y_hat_slice_2 = y_q_slice_2 + means_hat_2

                symbols_list.extend(y_q_slice_2.reshape(-1).tolist())
                indexes_list.extend(index_2.reshape(-1).tolist())

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                            y_hat_slice_2, torch.zeros_like(y_hat_slice_0))
                sc_params_4 = self.sc_transform_3(y_hat_slice)
                sc_params_4[:, :, 0::2, :] = 0
                sc_params_4[:, :, 1::2, 1::2] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3 + sc_params_4
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, _, _, scales_hat_3 = Demultiplexerv2(scales_hat)
                _, _, _, means_hat_3 = Demultiplexerv2(means_hat)
                index_3 = self.gaussian_conditional.build_indexes(scales_hat_3)
                y_q_slice_3 = self.gaussian_conditional.quantize(y_slice_3, "symbols", means_hat_3)
                y_hat_slice_3 = y_q_slice_3 + means_hat_3

                symbols_list.extend(y_q_slice_3.reshape(-1).tolist())
                indexes_list.extend(index_3.reshape(-1).tolist())

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, y_hat_slice_2, y_hat_slice_3)
                y_hat_slices.append(y_hat_slice)

            elif slice_index == 1:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                y_slice_anchor, y_slice_non_anchor = Demultiplexer(y_slice)

                zero_sc_params = torch.zeros(y_slice.shape[0], y_slice.shape[1]*2, y_slice.shape[2], y_slice.shape[3]).to(z_hat.device)
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, zero_sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                scales_hat_anchor, _ = Demultiplexer(scales_hat)
                means_hat_anchor, _ = Demultiplexer(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                y_q_slice_anchor = self.gaussian_conditional.quantize(y_slice_anchor, "symbols", means_hat_anchor)
                y_hat_slice_anchor = y_q_slice_anchor + means_hat_anchor

                symbols_list.extend(y_q_slice_anchor.reshape(-1).tolist())
                indexes_list.extend(index_anchor.reshape(-1).tolist())

                y_hat_slice = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
                sc_params = self.sc_transform_4(y_hat_slice)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0
                
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_non_anchor = Demultiplexer(scales_hat)
                _, means_hat_non_anchor = Demultiplexer(means_hat)
                index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
                y_q_slice_non_anchor = self.gaussian_conditional.quantize(y_slice_non_anchor, "symbols", means_hat_non_anchor)
                y_hat_slice_non_anchor = y_q_slice_non_anchor + means_hat_non_anchor

                symbols_list.extend(y_q_slice_non_anchor.reshape(-1).tolist())
                indexes_list.extend(index_non_anchor.reshape(-1).tolist())

                y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
                y_hat_slices.append(y_hat_slice)

            elif slice_index == 2:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                y_slice_non_anchor, y_slice_anchor = Demultiplexer(y_slice)

                zero_sc_params = torch.zeros(y_slice.shape[0], y_slice.shape[1]*2, y_slice.shape[2], y_slice.shape[3]).to(z_hat.device)
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, zero_sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_anchor = Demultiplexer(scales_hat)
                _, means_hat_anchor = Demultiplexer(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                y_q_slice_anchor = self.gaussian_conditional.quantize(y_slice_anchor, "symbols", means_hat_anchor)
                y_hat_slice_anchor = y_q_slice_anchor + means_hat_anchor

                symbols_list.extend(y_q_slice_anchor.reshape(-1).tolist())
                indexes_list.extend(index_anchor.reshape(-1).tolist())

                y_hat_slice = Multiplexer(torch.zeros_like(y_hat_slice_anchor), y_hat_slice_anchor)
                sc_params = self.sc_transform_5(y_hat_slice)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0

                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                scales_hat_non_anchor, _ = Demultiplexer(scales_hat)
                means_hat_non_anchor, _ = Demultiplexer(means_hat)
                index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
                y_q_slice_non_anchor = self.gaussian_conditional.quantize(y_slice_non_anchor, "symbols", means_hat_non_anchor)
                y_hat_slice_non_anchor = y_q_slice_non_anchor + means_hat_non_anchor

                symbols_list.extend(y_q_slice_non_anchor.reshape(-1).tolist())
                indexes_list.extend(index_non_anchor.reshape(-1).tolist())

                y_hat_slice = Multiplexer(y_hat_slice_non_anchor, y_hat_slice_anchor)
                y_hat_slices.append(y_hat_slice)

            else:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                gaussian_params = self.entropy_parameters_4(
                    torch.cat((params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                index = self.gaussian_conditional.build_indexes(scales_hat)
                y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", means_hat)
                y_hat_slice = y_q_slice + means_hat

                symbols_list.extend(y_q_slice.reshape(-1).tolist())
                indexes_list.extend(index.reshape(-1).tolist())

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], 
                "shape": z.size()[-2:],
                "y": y}

    def decompress(self, strings, shape, y=None):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        params = self.h_s(z_hat)

        slice_list = []
        N = self.M
        for t in range(1, self.num_iters+1):
            if t == self.num_iters:
                n = 0
            else:
                n = np.ceil(self.gamma(t/(self.num_iters)) * self.M)

            slice_list.append(int(N-n))
            N = n

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_iters):

            if slice_index == 0:
                cc_params = self.cc_transforms[slice_index](params)

                sc_params_1 = torch.zeros(z_hat.shape[0], slice_list[slice_index]*2, z_hat.shape[2]*4, z_hat.shape[3]*4).to(z_hat.device)
                sc_params = sc_params_1
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                scales_hat_0, _, _, _ = Demultiplexerv2(scales_hat)
                means_hat_0, _, _, _ = Demultiplexerv2(means_hat)
                index_0 = self.gaussian_conditional.build_indexes(scales_hat_0)
                rv = decoder.decode_stream(index_0.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_0 = self.gaussian_conditional.dequantize(rv, means_hat_0)

                y_hat_slice = Multiplexerv2(y_hat_slice_0, torch.zeros_like(y_hat_slice_0), 
                                            torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))
                sc_params_2 = self.sc_transform_1(y_hat_slice)
                sc_params_2[:, :, 0::2, :] = 0
                sc_params_2[:, :, 1::2, 0::2] = 0
                sc_params = sc_params_1 + sc_params_2
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_1, _, _ = Demultiplexerv2(scales_hat)
                _, means_hat_1, _, _ = Demultiplexerv2(means_hat)
                index_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
                rv = decoder.decode_stream(index_1.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_1 = self.gaussian_conditional.dequantize(rv, means_hat_1)

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                            torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))
                sc_params_3 = self.sc_transform_2(y_hat_slice)
                sc_params_3[:, :, 0::2, 0::2] = 0
                sc_params_3[:, :, 1::2, :] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, _, scales_hat_2, _ = Demultiplexerv2(scales_hat)
                _, _, means_hat_2, _ = Demultiplexerv2(means_hat)
                index_2 = self.gaussian_conditional.build_indexes(scales_hat_2)
                rv = decoder.decode_stream(index_2.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_2 = self.gaussian_conditional.dequantize(rv, means_hat_2)

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                            y_hat_slice_2, torch.zeros_like(y_hat_slice_0))
                sc_params_4 = self.sc_transform_3(y_hat_slice)
                sc_params_4[:, :, 0::2, :] = 0
                sc_params_4[:, :, 1::2, 1::2] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3 + sc_params_4
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, _, _, scales_hat_3 = Demultiplexerv2(scales_hat)
                _, _, _, means_hat_3 = Demultiplexerv2(means_hat)
                index_3 = self.gaussian_conditional.build_indexes(scales_hat_3)
                rv = decoder.decode_stream(index_3.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_3 = self.gaussian_conditional.dequantize(rv, means_hat_3)

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, y_hat_slice_2, y_hat_slice_3)
                y_hat_slices.append(y_hat_slice)

            elif slice_index == 1:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                zero_sc_params = torch.zeros(z_hat.shape[0], slice_list[slice_index]*2, z_hat.shape[2]*4, z_hat.shape[3]*4).to(z_hat.device)
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, zero_sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                scales_hat_anchor, _ = Demultiplexer(scales_hat)
                means_hat_anchor, _ = Demultiplexer(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                rv = decoder.decode_stream(index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

                y_hat_slice = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
                sc_params = self.sc_transform_4(y_hat_slice)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0
                
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                
                _, scales_hat_non_anchor = Demultiplexer(scales_hat)
                _, means_hat_non_anchor = Demultiplexer(means_hat)
                index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
                rv = decoder.decode_stream(index_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_non_anchor = self.gaussian_conditional.dequantize(rv, means_hat_non_anchor)
                
                y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
                y_hat_slices.append(y_hat_slice)

            elif slice_index == 2:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                zero_sc_params = torch.zeros(z_hat.shape[0], slice_list[slice_index]*2, z_hat.shape[2]*4, z_hat.shape[3]*4).to(z_hat.device)
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, zero_sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_anchor = Demultiplexer(scales_hat)
                _, means_hat_anchor = Demultiplexer(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                rv = decoder.decode_stream(index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

                y_hat_slice = Multiplexer(torch.zeros_like(y_hat_slice_anchor), y_hat_slice_anchor)
                sc_params = self.sc_transform_5(y_hat_slice)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0
                
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                
                scales_hat_non_anchor, _ = Demultiplexer(scales_hat)
                means_hat_non_anchor, _ = Demultiplexer(means_hat)
                index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
                rv = decoder.decode_stream(index_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_non_anchor = self.gaussian_conditional.dequantize(rv, means_hat_non_anchor)

                y_hat_slice = Multiplexer(y_hat_slice_non_anchor, y_hat_slice_anchor)
                y_hat_slices.append(y_hat_slice)

            else:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                gaussian_params = self.entropy_parameters_4(
                    torch.cat((params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                
                index = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*4, z_hat.shape[3]*4)
                y_hat_slice = self.gaussian_conditional.dequantize(rv, means_hat)
                
                y_hat_slices.append(y_hat_slice)
        
        
        
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        is_llric = True #TODO: fix self.args.model_name.find('llric') > -1
        y_tilde = None
        if is_llric and y is not None:
            y_tilde, latent_loss, ind  = self.llric_blk(y)
            output = x_hat + y_tilde
            return {"x_hat": output, "sampled_lr": y_tilde}
        return {"x_hat": x_hat, "sampled_lr": y_tilde}