import math
import numpy as np

import torch
import torch.nn as nn

from src.ans import BufferedRansEncoder, RansDecoder
from src.entropy_models import EntropyBottleneck, GaussianConditional
from src.layers import ResViTBlock, MultistageMaskedConv2d
from timm.layers import trunc_normal_
from src.models.llric_block import LRVQ

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

        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = ResViTBlock(dim=N,
                                depth=depths[0],
                                num_heads=num_heads[0],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                                norm_layer=norm_layer,
        )
        self.g_a2 = conv(N, N*3//2, kernel_size=3, stride=2)
        self.g_a3 = ResViTBlock(dim=N*3//2,
                        depth=depths[1],
                        num_heads=num_heads[1],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
        )
        self.g_a4 = conv(N*3//2, N*2, kernel_size=3, stride=1)
        self.g_a5 = ResViTBlock(dim=N*2,
                        depth=depths[2],
                        num_heads=num_heads[2],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
        )
        self.g_a6 = conv(N*2, M, kernel_size=3, stride=1)
        self.g_a7 = ResViTBlock(dim=M,
                        depth=depths[3],
                        num_heads=num_heads[3],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
        )

        self.h_a0 = conv(M, N*3//2, kernel_size=3, stride=2)
        self.h_a1 = ResViTBlock(dim=N*3//2,
                         depth=depths[4],
                         num_heads=num_heads[4],
                         kernel_size=kernel_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
        )
        self.h_a2 = conv(N*3//2, N*3//2, kernel_size=3, stride=2)
        self.h_a3 = ResViTBlock(dim=N*3//2,
                         depth=depths[5],
                         num_heads=num_heads[5],
                         kernel_size=kernel_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = ResViTBlock(dim=N*3//2,
                         depth=depths[0],
                         num_heads=num_heads[0],
                         kernel_size=kernel_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
        )
        self.h_s1 = deconv(N*3//2, N*3//2, kernel_size=3, stride=2)
        self.h_s2 = ResViTBlock(dim=N*3//2,
                         depth=depths[1],
                         num_heads=num_heads[1],
                         kernel_size=kernel_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
        )
        self.h_s3 = deconv(N*3//2, M*2, kernel_size=3, stride=2)
        
        self.g_s0 = ResViTBlock(dim=M,
                        depth=depths[2],
                        num_heads=num_heads[2],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
        )
        self.g_s1 = deconv(M, N*2, kernel_size=3, stride=1)
        self.g_s2 = ResViTBlock(dim=N*2,
                        depth=depths[3],
                        num_heads=num_heads[3],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
        )
        self.g_s3 = deconv(N*2, N*3//2, kernel_size=3, stride=1)
        self.g_s4 = ResViTBlock(dim=N*3//2,
                        depth=depths[4],
                        num_heads=num_heads[4],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
        )
        self.g_s5 = deconv(N*3//2, N, kernel_size=3, stride=2)
        self.g_s6 = ResViTBlock(dim=N,
                        depth=depths[5],
                        num_heads=num_heads[5],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)

        # model_name = "llric_test1"
        # if model_name.find('llric') > -1:
        #     # self.llric_blk = LRVQ(rank=2, split_size=16, n_hid=M, embedding_dim=16, n_embed=8192)
        #     self.llric_blk = LRVQ(rank=4, img_size=64, n_hid=512, embedding_dim=16, n_embed=8192, emb_ps=True)

        #     # Learned gain applied to the residual before it enters g_a.
        #     # Initialized to 1 so training starts from the identity; the network
        #     # can shrink or amplify the residual signal into a range the LIC
        #     # entropy model handles well.
        #     self.residual_gain = nn.Parameter(torch.ones(1))

        #     # Lightweight projection that maps the low-rank reconstruction
        #     # (y_tilde, 3-channel RGB) down to N feature channels at stride 2
        #     # so it can be concatenated with the g_a0 output as side-channel
        #     # context.  This lets the encoder "know" what the LRVQ already
        #     # reconstructed and focus its capacity on the true residual.
        #     self.lrvq_context_proj = nn.Sequential(
        #         conv(3, N // 2, kernel_size=3, stride=2),
        #         nn.GELU(),
        #         conv(N // 2, N, kernel_size=3, stride=1),
        #     )

        #     # 1x1 fusion conv that merges the g_a0 features (N ch) with the
        #     # projected LRVQ context (N ch) back down to N channels before
        #     # continuing through the rest of g_a.
        #     self.lrvq_context_fuse = conv(N * 2, N, kernel_size=1, stride=1)

        # Curriculum λ weight: multiplied into the distortion term of the loss
        # in the training loop.  Start high (pure distortion) and anneal toward
        # 1.0 so the entropy model has a chance to learn a useful signal before
        # rate pressure turns "transmit nothing" into a local minimum.
        self.register_buffer("lambda_weight", torch.tensor(1.0))
        
        self.entropy_bottleneck = EntropyBottleneck(N*3//2)
        self.gaussian_conditional = GaussianConditional(None)

        self.sc_transform_1 = MultistageMaskedConv2d(
                int(M - np.ceil(self.gamma(1/self.num_iters)*M)), int(M - np.ceil(self.gamma(1/self.num_iters)*M))*2, 
                kernel_size=3, padding=1, stride=1, mask_type='A'
        )
        self.sc_transform_2 = MultistageMaskedConv2d(
                int(M - np.ceil(self.gamma(1/self.num_iters)*M)), int(M - np.ceil(self.gamma(1/self.num_iters)*M))*2, 
                kernel_size=5, padding=2, stride=1, mask_type='B'
        )
        self.sc_transform_3 = MultistageMaskedConv2d(
                int(M - np.ceil(self.gamma(1/self.num_iters)*M)), int(M - np.ceil(self.gamma(1/self.num_iters)*M))*2, 
                kernel_size=5, padding=2, stride=1, mask_type='C'
        )
        self.sc_transform_4 = MultistageMaskedConv2d(
                int(np.ceil(self.gamma(1/self.num_iters)*M)-np.ceil(self.gamma(2/self.num_iters)*M)), 
                int(np.ceil(self.gamma(1/self.num_iters)*M)-np.ceil(self.gamma(2/self.num_iters)*M))*2,
                kernel_size=5, padding=2, stride=1, mask_type='B'
        )
        self.sc_transform_5 = MultistageMaskedConv2d(
                int(np.ceil(self.gamma(2/self.num_iters)*M)-np.ceil(self.gamma(3/self.num_iters)*M)), 
                int(np.ceil(self.gamma(2/self.num_iters)*M)-np.ceil(self.gamma(3/self.num_iters)*M))*2,
                kernel_size=5, padding=2, stride=1, mask_type='B'
        )

        self.entropy_parameters_1 = nn.Sequential(
                conv(M*2 + int(M - np.ceil(self.gamma(1/self.num_iters)*M))*12//3, 
                int(M - np.ceil(self.gamma(1/self.num_iters)*M))*10//3, 1, 1),
                nn.GELU(),
                conv(int(M - np.ceil(self.gamma(1/self.num_iters)*M))*10//3, 
                int(M - np.ceil(self.gamma(1/self.num_iters)*M))*8//3, 1, 1),
                nn.GELU(),
                conv(int(M - np.ceil(self.gamma(1/self.num_iters)*M))*8//3, 
                int(M - np.ceil(self.gamma(1/self.num_iters)*M))*6//3, 1, 1),
        )
        self.entropy_parameters_2 = nn.Sequential(
                conv(M*2 + int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*12//3, 
                int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*10//3, 1, 1),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*10//3, 
                int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*8//3, 1, 1),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*8//3, 
                int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*6//3, 1, 1),
        ) 
        self.entropy_parameters_3 = nn.Sequential(
                conv(M*2 + int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*12//3, 
                int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*10//3, 1, 1),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*10//3, 
                int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*8//3, 1, 1),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*8//3, 
                int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*6//3, 1, 1),
        ) 
        self.entropy_parameters_4 = nn.Sequential(
                conv(M*2 + int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 
                int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 1, 1),
                nn.GELU(),
                conv(int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 
                int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 1, 1),
                nn.GELU(),
                conv(int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 
                int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 1, 1),
        ) 
        
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

        self.apply(self._init_weights)  
        
    def update_bwd_t(self, elapsed_t):
        self.timers["bwd_total"] += elapsed_t
        
    def print_timers(self):
        fwd_t = self.timers["fwd_total"]
        bwd_t = self.timers["bwd_total"]
        total_t = fwd_t + bwd_t
        
        frac_times = {}
        for key, value in self.timers.items():
            if key in ["fwd_total", "bwd_total"]: continue
            frac_times[key] = value / fwd_t
            
        sorted_dict = dict(sorted(frac_times.items(), key=lambda item: item[1], reverse=True))
        
        print("----- Relative Timing of Model Components -----")
        print(f"fwd_total ➡ {fwd_t/total_t:.4f}")
        for key, value in sorted_dict.items(): print(f"\t↪  {key:<10} ➡ {value:.4f}")
        print(f"bwd_total ➡ {bwd_t/total_t:.4f}")

            
    def set_lambda_weight(self, w: float):
        """Set the distortion curriculum weight (call from training loop).

        During early training, pass a large value (e.g. 10–50) so the LIC is
        forced to encode something useful before rate pressure kicks in.  Anneal
        toward 1.0 over the first N epochs.  Example schedule:

            for epoch in range(total_epochs):
                w = max(1.0, 20.0 * (1 - epoch / warmup_epochs))
                model.set_lambda_weight(w)
        """
        self.lambda_weight.fill_(w)

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

    def g_a(self, x, lrvq_context=None):
        """Analysis transform.

        Args:
            x: Input image (or scaled residual when LRVQ is active).
            lrvq_context: Optional projected LRVQ reconstruction features.
                When provided they are concatenated with the g_a0 output and
                fused via a 1x1 conv so the encoder has explicit side-channel
                knowledge of what the low-rank stage already captured.
        """
        x = self.g_a0(x)
        if lrvq_context is not None:
            # Fuse LRVQ context: gives the encoder a reference frame so it can
            # focus on encoding the true residual rather than re-encoding the
            # already-reconstructed signal.
            x = self.lrvq_context_fuse(torch.cat([x, lrvq_context], dim=1))
        x = self.g_a1(x)
        x = self.g_a2(x)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x)
        x = self.g_a6(x)
        x = self.g_a7(x)
        return x

    def g_s(self, x):
        x = self.g_s0(x)
        x = self.g_s1(x)
        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s4(x)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x)
        return x

    def h_a(self, x):
        x = self.h_a0(x)
        x = self.h_a1(x)
        x = self.h_a2(x)
        x = self.h_a3(x)
        return x

    def h_s(self, x):
        x = self.h_s0(x)
        x = self.h_s1(x)
        x = self.h_s2(x)
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
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
        ########################################
        ##### Low-rank first-stage (LLRIC)
        ########################################
        # is_llric = self.args.model_name.find('llric') > -1
        # y_tilde = 0
        lrvq_context = None
        latent_loss = torch.tensor([0.0], device=x.device)

        # if is_llric:
        #     with torch.no_grad():
        #         # LLRIC is frozen; detach so no gradient flows back into it.
        #         y_tilde, latent_loss, ind = self.llric_blk(x)
        #         y_tilde = y_tilde.detach()

        #     # --- Correct residual ---
        #     # The LIC sees and compresses (x - y_tilde), i.e. only what the
        #     # low-rank stage failed to reconstruct.  A learned gain normalizes
        #     # this residual into a numerically comfortable range; it is
        #     # initialized to 1 and adapts during training.
        #     residual = (x - y_tilde) * self.residual_gain

        #     # --- Side-channel context ---
        #     # Project y_tilde to N feature channels at the same spatial
        #     # resolution as g_a0's output so the encoder has an explicit
        #     # reference frame and can focus on the true residual content.
        #     lrvq_context = self.lrvq_context_proj(y_tilde)
        # else:
        residual = x

        llric_t = time.perf_counter_ns()
        self.timers["llric"] += llric_t - start_t

        y = self.g_a(residual, lrvq_context)
        z = self.h_a(y)
        encode_t = time.perf_counter_ns()
        self.timers["encode"] += encode_t - llric_t
        
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset
        
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
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):

            if slice_index == 0:
                support_slices = torch.cat([params] + y_hat_slices, dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                sc_params_1 = torch.zeros_like(cc_params)
                sc_params = sc_params_1
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_0 = y_hat_slice.clone()
                y_0[:, :, 0::2, 1::2] = 0
                y_0[:, :, 1::2, :] = 0
                sc_params_2 = self.sc_transform_1(y_0)
                sc_params_2[:, :, 0::2, :] = 0
                sc_params_2[:, :, 1::2, 0::2] = 0
                sc_params = sc_params_1 + sc_params_2
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_1 = y_hat_slice.clone()
                y_1[:, :, 0::2, 1::2] = 0
                y_1[:, :, 1::2, 0::2] = 0
                sc_params_3 = self.sc_transform_2(y_1)
                sc_params_3[:, :, 0::2, 0::2] = 0
                sc_params_3[:, :, 1::2, :] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_2 = y_hat_slice.clone()
                y_2[:, :, 1::2, 0::2] = 0
                sc_params_4 = self.sc_transform_3(y_2)
                sc_params_4[:, :, 0::2, :] = 0
                sc_params_4[:, :, 1::2, 1::2] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3 + sc_params_4
                gaussian_params = self.entropy_parameters_1(
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
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_half = y_hat_slice.clone()
                y_half[:, :, 0::2, 0::2] = 0
                y_half[:, :, 1::2, 1::2] = 0

                sc_params = self.sc_transform_4(y_half)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0
                
                gaussian_params = self.entropy_parameters_2(
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
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_half = y_hat_slice.clone()
                y_half[:, :, 0::2, 1::2] = 0
                y_half[:, :, 1::2, 0::2] = 0

                sc_params = self.sc_transform_5(y_half)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0

                gaussian_params = self.entropy_parameters_3(
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

                gaussian_params = self.entropy_parameters_4(
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
        x_hat = self.g_s(y_hat)

        decode_t = time.perf_counter_ns()
        self.timers["decode"] += decode_t - entropy_t

        ########################################
        ##### Recombine residual + low-rank
        ########################################
        # if is_llric:
        #     # Undo the learned gain: x_hat is the compressed residual in
        #     # gain-scaled space; divide back before adding y_tilde.
        #     output = (x_hat / self.residual_gain) + y_tilde
        # else:
        output = x_hat
        # y_tilde, latent_loss, ind = self.llric_blk(x)
        # # output = y_tilde
        # output = ((y_tilde ** 2 + output ** 2) / 2) ** 0.5
        output = output.clamp(0.0, 1.0)

        self.timers["fwd_total"] += time.perf_counter_ns() - start_t
        return {
            "x_hat": output,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "vq_loss": latent_loss,
            "sampled_lr": None,
            # Expose for training loop: multiply the distortion term of your
            # R-D loss by this value.  Use set_lambda_weight() to schedule it.
            "lambda_weight": self.lambda_weight,
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
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
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
                
                print(torch.cat((params, sc_params, cc_params), dim=1).shape)
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

                print(torch.cat((params, sc_params, cc_params), dim=1).shape)
                exit()
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