# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys
# print(sys.path)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

print("CWD:", os.getcwd())
print("FILE DIR:", os.path.dirname(__file__))
print("sys.path[0]:", sys.path[0])
import time

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

from models.image import image_models as pretrained_models
from models.pretrained import load_pretrained as load_state_dict
from models.image import model_architectures as architectures
import matplotlib.pyplot as plt
import numpy as np
import src

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

class IterTimer:
    def __init__(self, total):
        self.total = total
        self.accum = 0

    def __enter__(self):
        self.start = time.time()
        print(f"Entering timer with {self.total} iterations")
        self.print()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        build_str = "".join(['#'] * 10)
        build_str = '[' + build_str + ']' + f"   Iter {self.accum}/{self.total}"
        print(build_str)
        print(f"Execution took {self.end - self.start} seconds\n\n")

    def update(self, add_line=None):
        self.accum += 1
        self.print(add_line)

    def print(self, add_line=None):
        LINE_UP = "\033[A"
        frac = int((self.accum / self.total) * 100)
        hashes = frac // 10
        slashes = (frac % 10) // 5
        build_str = "".join(['#'] * hashes + ['/']*slashes + ['-']*(10-slashes-hashes))
        build_str = '\r[' + build_str + ']' + f"   Iter {self.accum}/{self.total}   "
        if add_line is not None:
            format_addition = '\n' + add_line + "          " + LINE_UP
        else:
            format_addition = ''
        print(build_str, end=format_addition)
        
        
def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)

num_img = 1
saved_images = []
saved_lr_images = []
image_names = []
image_lr_names = []
imgs_are_saved = False

@torch.no_grad()
def inference(model, x):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 256  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], out_enc['y'])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    if len(saved_images) < num_img * 3:
        x_np = x.squeeze().cpu().detach().numpy()
        saved_images.append(np.transpose(x_np, (1, 2, 0)))
        img_np = out_dec["x_hat"].squeeze().cpu().detach().numpy()
        saved_images.append(np.transpose(img_np, (1, 2, 0)))
        img_np = out_dec["sampled_lr"].squeeze().cpu().detach().numpy()
        saved_images.append(np.transpose(img_np, (1, 2, 0)))
        im_size = 256
        embedding_dim = 16
        num_embeddings = 4096
        rank = 2
        input_channels = 3
        num_patches = (im_size // embedding_dim) ** 2
        bits_per_vec = math.log2(num_embeddings)
        num_vecs_per_patch = 2 * rank * input_channels
        exp_bits = int(bits_per_vec * num_vecs_per_patch * num_patches)
        bpp = exp_bits / (im_size ** 2)
        print(out_dec["sampled_lr"].shape)
        lr_psnr = psnr(x, out_dec["sampled_lr"])
        image_lr_names.append(f"{bpp:.3f} | {lr_psnr: .3f}")
        
    stats = {
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }
    _psnr = stats["psnr"]
    image_names.append(f"{bpp:.3f} | {_psnr: .3f}")
    
    
    
    
    return stats


@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x)
    elapsed_time = time.time() - start

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )
    
    

    return {
        "psnr": psnr(x, out_net["x_hat"].clamp_(0, 1)),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()


def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    state_dict = load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return architectures[arch].from_state_dict(state_dict).eval()


def eval_model(model, filepaths, entropy_estimation=False, half=False):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    global imgs_are_saved
    with IterTimer(len(filepaths)) as timer:
        for f in filepaths:
            x = read_image(f).to(device)
            crop = transforms.RandomCrop(256)
            x = crop(x)
            if not entropy_estimation:
                if half:
                    model = model.half()
                    x = x.half()
                rv = inference(model, x)
            else:
                rv = inference_entropy_estimation(model, x)
            for k, v in rv.items():
                metrics[k] += v
            
            if (len(saved_images) >= num_img * 3) and not imgs_are_saved:
                fig, axes = plt.subplots(num_img, 3, dpi=450, figsize=(4, 4))  # 8 rows, 2 columns
                for i, ax in enumerate(axes.flat):
                    if i % 3 == 1:
                        ax.set_title(image_names[i // 3], fontsize=8)
                    if i % 3 == 2:
                        ax.set_title(image_lr_names[i // 3], fontsize=8)
                    ax.imshow(saved_images[i], cmap='gray' if saved_images[i].ndim == 2 else None)
                    ax.axis('off')  # Hide axis ticks and labels
                plt.savefig(os.path.join("base_100ep.png"))
                imgs_are_saved = True
                # exit()
            
            
            timer.update()
        
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser(
        add_help=False,
    )

    # Common options.
    parent_parser.add_argument("dataset", type=str, help="dataset path")
    parent_parser.add_argument("--model-name", type=str, default='', help="model name")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=pretrained_models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=src.available_entropy_coders(),
        default=src.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    
    parent_parser.add_argument(
        "--save-lr-intermediate",
        action="store_true",
        help="save image to ./lr_intermediate.png",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="+",
        type=int,
        default=(1,),
    )

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )

    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    if not args.source:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    src.set_entropy_coder(args.entropy_coder)

    if args.source == "pretrained":
        runs = sorted(args.qualities)
        opts = (args.architecture, args.metric)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d}"
    elif args.source == "checkpoint":
        runs = args.paths
        opts = (args.architecture,)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s}"

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()
        model = load_func(*opts, run)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")

        if args.source == "checkpoint":
            model.update(force=True)

        metrics = eval_model(model, filepaths, args.entropy_estimation, args.half)
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": args.architecture,
        "description": f"Inference ({description})",
        "results": results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
    

# python -m ./src/utils/eval_model checkpoint /kneehaw/datasets/flicker/ -a tinylic -p ./checkpoints/tinylic/3/base_100ep_bestL.tar --cuda