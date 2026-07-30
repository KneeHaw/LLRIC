import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import ImageFolder
from src.models.image import image_models

import torch
import torch.nn as nn
from src.models.utils import conv, deconv, update_registered_buffers, quantize_ste, \
    Demultiplexer, Multiplexer, Demultiplexerv2, Multiplexerv2
from src.layers import ResViTBlock, MultistageMaskedConv2d
from timm.layers import trunc_normal_
from src.models.llric_block import LRVQ
from src.data_loaders import get_loaders


class LR_frozen(nn.Module):
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
        self.g_a4 = conv(N*3//2, N*2, kernel_size=3, stride=2)
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
        self.g_a6 = conv(N*2, M, kernel_size=3, stride=2)
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
        
        self.llric_blk = LRVQ(rank=4, img_size=64, n_hid=512, embedding_dim=16, n_embed=8192, emb_ps=True)
        
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
        
    def g_a(self, x):
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x)
        x = self.g_a6(x)
        x = self.g_a7(x)
        return x
        
    def forward(self, x):
        # print(x.shape)
        # y = self.g_a(x)
        # print(y.shape)
        # exit()
        y_tilde, latent_loss, ind  = self.llric_blk(x)
        
        return {
            "x_hat": y_tilde,
            "likelihoods": {"y": None, "z": None},
            "vq_loss": latent_loss,
            "sampled_lr": y_tilde,
        }
           

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

        out["bpp_loss"] = 0.0 
        # sum(
        #     (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        #     for likelihoods in output["likelihoods"].values()
        # )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["vq_loss"] = output["vq_loss"].sum()
        # print(out["vq_loss"].shape)
        out["loss"] = out["mse_loss"] + out["bpp_loss"] + out["vq_loss"]
        # self.lmbda * 255**2 * 
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
    base_dir = f'./checkpoints/{args.model}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    return base_dir


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
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    # aux_optimizer = optim.Adam(
    #     (params_dict[n] for n in sorted(aux_parameters)),
    #     lr=args.aux_learning_rate,
    # )
    return optimizer, None


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        # aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # aux_loss = model.aux_loss()
        # aux_loss.backward()
        # aux_optimizer.step()

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
                f'Bpp loss: {out_criterion["bpp_loss"]:.4f} | '
                f'Lat loss: {out_criterion["vq_loss"].item():.5f} | '
                # f"Aux loss: {aux_loss.item():.2f}"
            )
            # model.print_timers()


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    # aux_loss = AverageMeter()
    lat_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            # print(torch.min(d), torch.max(d), torch.mean(d), torch.std(d))
            # exit()
            # aux_loss.update(model.aux_loss())
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
        # f"Aux loss: {aux_loss.avg:.2f} |\n"
        
    )

    return loss.avg


def sample_result(valid_loader, vae_model, images_dir, run_name):
    """
    Take a sample of four images from the dataset for fidelity observations
    """
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
        print("Saving figure...", os.path.join(images_dir, f"{run_name}.png"))
        plt.savefig(os.path.join(images_dir, f"{run_name}.png"))
        plt.close()
        

def save_checkpoint(state, is_best, base_dir, model_name, filename=None):
    print("Saving Checkpoint... ", end="")
    # exit()
    if filename is None:
        filename = model_name + "_checkpoint.pth.tar"
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+f"{model_name}_checkpoint_best_loss.pth.tar")
    print("Done!")


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
        "--batch-size", type=int, default=256, help="Per-device batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        help="Test batch size (default: %(default)s)",
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
        default=1e-4,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-4,
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
        default="0,1,2,3",
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
        default="frozenLRVQ_test2", 
        type=str,
        help='Result dir name', 
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    
    parser.add_argument("--checkpoint", 
                        default=None, 
                        type=str, 
                        help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

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
    print("Lambda: ", args.lmbda)
    
    train_dataloader, val_dataloader, test_dataloader, im_size = get_loaders(args=args, 
                                                                             worker_init_fn=None, 
                                                                             dev_count=device_count, 
                                                                             device=device)

    net = LR_frozen(args=args)
    net = net.to(device)
    
    if args.cuda and torch.cuda.device_count() > 1:
        print("Distributing model parallel...")
        net = CustomDataParallel(net)

    optimizer, _ = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        sample_result(test_dataloader, net, '.', 'test')
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
            _,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        sample_result(test_dataloader, net, '.', 'test')
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save and is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                base_dir,
                args.model_name
            )


if __name__ == "__main__":
    main(sys.argv[1:])