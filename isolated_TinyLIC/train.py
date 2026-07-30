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

import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import ImageFolder
from src.models.image import image_models
from src.data_loaders import get_loaders

import matplotlib.pyplot as plt
import numpy as np


##################################################
##### Monkey Patching
##################################################
MACS_FILE = "./saved/MACS.csv"
with open (MACS_FILE, 'w') as f:
    f.write("Layer,MACS\n")

_original_init = nn.Linear.__init__

def patched_init(self, in_features, out_features, *args, **kwargs):
    # print(f"Linear: {in_features} -> {out_features}")
    # print(f">> MACS: {in_features * out_features}")
    with open (MACS_FILE, 'a') as f:
        f.write(f"Linear,{in_features * out_features}\n")
    _original_init(self, in_features, out_features, *args, **kwargs)

nn.Linear.__init__ = patched_init

# _original_init = nn.Conv2d.__init__

# def patched_init(self, in_features, out_features, kernel_size, stride, *args, **kwargs):
#     print(f"Conv2d: {in_features} -> {out_features}, kernel_size: {kernel_size}, stride: {stride}")
#     print(f"MACS: {in_features * out_features * kernel_size[0] * kernel_size[1]}")
#     _original_init(self, in_features, out_features, kernel_size, stride, *args, **kwargs)

# nn.Conv2d.__init__ = patched_init
##################################################


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["vq_loss"] = output["vq_loss"].sum()
        # print(out["vq_loss"].shape)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"] + out["vq_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'./saved/checkpoints/{args.model_name}'# /{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    imgs_dir = f'./saved/images/{args.model_name}'# /{args.quality_level}/'
    os.makedirs(imgs_dir, exist_ok=True)

    return base_dir, imgs_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    # assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        # exit("STOPPING AFTER FIRST BATCH: LINE 205 train.py")

        bwd_start_t = time.perf_counter_ns()
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        model.update_bwd_t(time.perf_counter_ns() - bwd_start_t)

        # if i*len(d) % 5000 == 0:
        iter = i // (int(len(train_dataloader) / 20))
        # print(mod, i, len(train_dataloader))
        if i % (int(len(train_dataloader) / 20)) == 0:
            logging.info(
                f'[{iter}/{20}] | '
                # f'[{i*len(d)}/{len(train_dataloader.dataset)}] '
                # f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'Loss: {out_criterion["loss"].item():.3f} | '
                f'MSE loss: {out_criterion["mse_loss"].item():.5f} | '
                f'Bpp loss: {out_criterion["bpp_loss"].item():.4f} | '
                f'Lat loss: {out_criterion["vq_loss"].item():.5f} | '
                f"Aux loss: {aux_loss.item():.2f}"
            )
            model.print_timers()
            
    sample_result(train_dataloader, model, add_str=f"_epoch{epoch}")

imgs_dir = None
def sample_result(valid_loader, vae_model, images_dir='.', run_name="test", add_str=""):
    """
    Take a sample of four images from the dataset for fidelity observations
    """
    global imgs_dir
    vae_model.eval()
    images = []
    labels = []
    num_samples = 4
    idxs = [i for i in range(num_samples)]
    
    with torch.no_grad():
        for i, item in enumerate(valid_loader):
            print(i)
            target = item
            if i >= num_samples:
                break        
            target = target.cuda()
            input_var = torch.autograd.Variable(item.cuda())
            
            # Forward pass
            output = vae_model(input_var)["x_hat"]
            out_np = np.transpose(output.cpu().detach().numpy(), (0, 2, 3, 1))  # Change to (B, H, W, C)
            target_np = np.transpose(target.cpu().detach().numpy(), (0, 2, 3, 1))  # Change to (B, H, W, C)))
            
            select_target = target_np[0, :, :, :]
            select_out = out_np[0, :, :, :]
            images.append(select_target)
            images.append(select_out)
            err_mse = np.mean(np.square(select_out - select_target))
            err_mae = np.mean(np.abs(select_out - select_target))
            err_psnr = 10 * np.log10(1 / err_mse)
            labels.append(f"{err_mae:.3f} | {err_mse:.3f} | {err_psnr: .3f}")
             
        fig, axes = plt.subplots(num_samples, 2, figsize=(6, 14))  # 8 rows, 2 columns

        for i, ax in enumerate(axes.flat):
            if i % 2 == 1:
                ax.set_title(labels[i // 2])
            ax.imshow(images[i], cmap='gray' if images[i].ndim == 2 else None)
            ax.axis('off')  # Hide axis ticks and labels

        plt.tight_layout()
        plt.show()
        fname = f"{run_name}{add_str}.png"
        print("Saving figure...", os.path.join(imgs_dir, fname))
        plt.savefig(os.path.join(imgs_dir, fname))
        plt.close()


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    lat_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            # lat_loss.update(out_criterion["vq_loss"])

    logging.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.3f} | "
        f"MSE loss: {mse_loss.avg:.5f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        # f"Lat loss: {lat_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f} |\n"
        
    )
    
    sample_result(test_dataloader, model)
    
    return loss.avg


def save_checkpoint(state, is_best, base_dir, model_name, filename=None):
    if filename is None:
        filename = model_name + "_checkpoint.pth.tar"
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+f"{model_name}_checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m", "--model",
        default="tinylic",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-root", 
        type=str,
        default="/home/kneehaw/datasets/",
        help="Root of dataset(s)",   
    )
    parser.add_argument(
        "-d", "--dataset", 
        type=str,
        default="CELEBA",
        help="Training dataset",   
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=100,  # paper = 400
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr", "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n", "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality-level",
        type=int,
        default=3,
        help="Quality level (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=48, help="Per-device batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=48,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(64, 64),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda", 
        default=True, 
        action="store_true", 
        help="Use cuda")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="1,2,3",
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        '--model-name', 
        default="", 
        type=str,
        help='Result dir name', 
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir, _imgs_dir = init(args)
    global imgs_dir 
    imgs_dir = _imgs_dir

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count() if args.cuda and torch.cuda.is_available() else 1
    print("Device: ", device, " | Count: ", device_count)
    
    train_dataloader, val_dataloader, test_dataloader, im_size = get_loaders(args=args, 
                                                                             worker_init_fn=None, 
                                                                             dev_count=device_count, 
                                                                             device=device)

    net = image_models[args.model](quality=int(args.quality_level), args=args)
    net = net.to(device)
        
    if args.cuda and torch.cuda.device_count() > 1:
        print("Distributing model parallel...")
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # if all([args.model_name.find('llric') > -1, args.model_name.find('frozen') > -1]):
        # logging.info("Loading LLRIC backbone")
        # if all([args.model_name.find('llric') > -1]):
        #     source_dict = torch.load("/home/kneehaw/python_projects/LLRIC/isolated_TinyLIC/saved/checkpoints/tinylic/3/celeba_llriconly1_checkpoint_best_loss.pth.tar", weights_only=True)["state_dict"]
        #     target_dict = net.state_dict()
        #     filtered_dict = target_dict
        #     # for k, v in target_dict.items():
        #     #     if k.find('llric') == -1: continue
        #     #     print(k)
                
        #     for k, v in source_dict.items():
        #         if k.find('llric') == -1: continue
        #         if (k in target_dict.keys()): #.replace('module.llric', 'llric') 
        #             # print(k)
        #             filtered_dict[k] = v
        #     # exit()
        #     # print(filtered_dict.keys())
        #     # exit()
        #     net.load_state_dict(filtered_dict, strict=False)
        #     # net.llric_blk.requires_grad_(False)
        
        logging.info("Testing loaded model...")
        sample_result(test_dataloader, net)
        # loss = test_epoch(0, test_dataloader, net, criterion)
        exit()
        
        
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logging.info('======Current epoch %s ======'%epoch)
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                base_dir,
                args.model_name
            )


if __name__ == "__main__":
    main(sys.argv[1:])
