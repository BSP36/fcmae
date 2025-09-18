import os
import torch
from typing import Tuple
from torchvision.utils import save_image
from tqdm import tqdm
from models.patch_utils import patch_wise_dernormalize


@torch.no_grad()
def test_fcmae(model, test_loader, device, output_dir, norm_pix_loss) -> Tuple[float, float]:
    viz_dir = os.path.join(output_dir, "test_viz")
    os.makedirs(viz_dir, exist_ok=True)
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(device) 
    for idx, (images, _) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        loss, pred, mask = model(images, mask_ratio=0.6)  # Using a fixed mask ratio for testing
        total_loss += loss.item()

        if norm_pix_loss:
            pred = patch_wise_dernormalize(pred, images)
        # visualization
        N, C, H, W = images.shape
        _, D, h, w = pred.shape
        assert H // h == W // w
        p = H // h  # patch size
        mask = mask.reshape(N, 1, h, w).repeat_interleave(p, axis=2).repeat_interleave(p, axis=3)
        pred = pred.reshape(N, D, h * w).permute(0, 2, 1).reshape(N, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(N, C, H, W)

        # combine images
        input_masked = images * (~mask).float()

        pred = pred * mask.float() + input_masked
        
        im_vis = torch.cat([images, input_masked, pred], dim=-1)  # (N, C, H, W*3)
        im_vis = im_vis * std + mean
        im_vis = torch.clip(im_vis, min=0.0, max=1.0)
        im_vis = im_vis.cpu()

        for i in range(N):
            save_image(im_vis[i], os.path.join(viz_dir, f"img{idx * N + i:04d}.png"))

    avg_loss = total_loss / len(test_loader)
    
    return avg_loss


if __name__ == '__main__':
    import argparse
    from configs.args import load_config
    from models.fcmae import FCMAE
    from dataloader.stl10 import get_stl10_dataloaders

    parser = argparse.ArgumentParser('FCMAE test script', add_help=False)
    parser.add_argument('--name', type=str, help='experiment name')
    args = parser.parse_args()
    config_path = os.path.join("./experiments", args.name, "config.yaml")
    args = load_config(config_path)
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
    checkpoint_path = os.path.join(args.output, "checkpoints", "fcmae_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    device = args.device


    test_loader = get_stl10_dataloaders(datatype="test", data_root="./datasets/stl10", batch_size=1, num_workers=0)

    test_fcmae(model, test_loader, args.device, args.output, args.norm_pix_loss)