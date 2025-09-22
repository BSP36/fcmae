import os
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from typing import Sequence, Union

from modules.patch_utils import patch_wise_dernormalize
from utils.visualization import frames2mp4


def fix_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def eval_fcmae(
    model: torch.nn.Module,
    data_loaders: Sequence[DataLoader],
    device: Union[str, torch.device],
    output_dir: str,
    norm_pix_loss: bool,
    std: Sequence[float] = (0.229, 0.224, 0.225),
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    seed: int = 42,
    num_viz: int = 10**8,
):
    """
    Runs inference for the FCMAE model on the test dataset and saves visualizations.

    Args:
        model (torch.nn.Module): FCMAE model instance.
        data_loaders (Sequence[DataLoader]): DataLoaders to evaluate.
        device (str or torch.device): Device to run inference on.
        output_dir (str): Directory to save results.
        norm_pix_loss (bool): Whether to denormalize pixel loss.
        std (Sequence[float], optional): Per-channel standard deviation for normalization. Default is ImageNet std.
        mean (Sequence[float], optional): Per-channel mean for normalization. Default is ImageNet mean.
        seed (int, optional): Random seed for random mask. Default is 42.
        num_viz (int, optional): The number of images to visualize. Default is 10**8.
    """
    model = model.to(device).eval()
    std_tensor = torch.tensor(std, dtype=torch.float32)[None, :, None, None].to(device)
    mean_tensor = torch.tensor(mean, dtype=torch.float32)[None, :, None, None].to(device)

    for key, data_loader in data_loaders.items():
        fix_seed(seed)
        total_loss = 0.0
        for idx, (images, _) in tqdm(enumerate(data_loader), desc=f"Testing {key} dataset"):
            viz_tmp = os.path.join(output_dir, f"results/{key}/images_aug")
            os.makedirs(viz_tmp, exist_ok=True)
            images = images.to(device)
            # Forward pass with fixed mask ratio
            loss, pred, mask = model(images, mask_ratio=0.6)
            N, C, H, W = images.shape
            total_loss += loss.item() * N

            if idx * N >= num_viz:
                continue

            # Visualization
            _, D, h, w = pred.shape
            assert H // h == W // w, "Patch size mismatch"
            
            # Patch size
            patch_size = H // h

            # Optionally denormalize predictions
            if norm_pix_loss:
                pred = patch_wise_dernormalize(pred, images)

            # Upsample mask to image size
            mask = mask.reshape(N, 1, h, w)
            mask = mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)

            # Reshape predictions to image size
            pred = pred.reshape(N, D, h * w).permute(0, 2, 1)
            pred = pred.reshape(N, h, w, patch_size, patch_size, C)
            pred = pred.permute(0, 5, 1, 3, 2, 4).reshape(N, C, H, W)

            # Combine masked input and prediction
            input_masked = images * (~mask).float()
            pred = pred * mask.float() + input_masked

            # Concatenate masked input, prediction, and original image for visualization
            # Output shape: (N, C, H, W*3)
            im_vis = torch.cat([input_masked, pred, images], dim=-1)
            im_vis = im_vis * std_tensor + mean_tensor
            im_vis = torch.clamp(im_vis, min=0.0, max=1.0)
            im_vis = im_vis.cpu()

            # Save images for each sample in the batch
            for i in range(N):
                if idx * N + i < num_viz:
                    save_image(im_vis[i], os.path.join(viz_tmp, f"img{idx * N + i:04d}.png"))

        # Generate video from saved images
        image_paths = sorted(Path(viz_tmp).glob("*.png"))
        if len(image_paths) > 0:
            frames2mp4(image_paths, os.path.join(output_dir, f"results/{key}/viz_aug.mp4"), fps=4)
        # Optionally remove temporary images directory
        # import shutil
        # shutil.rmtree(viz_tmp)
        print(f"{key} loss: {total_loss / len(data_loader.dataset):.4f}")

if __name__ == '__main__':
    import argparse
    from configs.args import load_config
    from modules.fcmae import FCMAE
    from dataloader.stl10 import get_stl10_dataloaders, simple_transform

    parser = argparse.ArgumentParser('FCMAE test script', add_help=False)
    parser.add_argument('--name', type=str, help='Experiment name')
    args = parser.parse_args()
    config_path = os.path.join("./experiments", args.name, "config.yaml")
    args = load_config(config_path)

    # Compute patch size from model config
    patch_size = args.stem_stride * 2 ** (len(args.dims) - 1)
    model = FCMAE(
        num_colors=3,
        stem_stride=args.stem_stride,
        depths=args.depths,
        dims=args.dims,
        patch_size=patch_size,
        dec_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        norm_pix_loss=args.norm_pix_loss,
    )

    # Load model checkpoint
    checkpoint_path = os.path.join(args.output, "checkpoints", "fcmae_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Prepare data loaders
    data_loaders = {}
    for key in ["unlabeled", "train", "test"]:
        data_loaders[key] = get_stl10_dataloaders(
            datatype=key,
            data_root="./datasets/stl10",
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False
        )
        data_loaders[key].dataset.transform = simple_transform((96, 96))

    # Run inference
    eval_fcmae(
        model,
        data_loaders,
        args.device,
        args.output,
        args.norm_pix_loss,
        num_viz=1000,
    )