import os
import torch
import torch.utils.tensorboard as tensorboard
from tqdm import tqdm

def train_fcmae(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int,
    mask_ratio: float,
    save_interval: int,
    output_dir: str,
    writer: tensorboard.SummaryWriter,
) -> None:
    """
    Train the FCMAE model using masked autoencoding.

    Args:
        model (torch.nn.Module): The FCMAE model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run training on (CPU or CUDA).
        num_epochs (int): Number of training epochs.
        mask_ratio (float): Ratio of patches to mask for MAE.
        save_interval (int): Interval (in epochs) to save checkpoints.
        output_dir (str): Directory to save model checkpoints.
        writer (tensorboard.SummaryWriter): TensorBoard writer for logging.

    Returns:
        None
    """
    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, _ in pbar:  # Labels are unused for unsupervised MAE
            images = images.to(device)
            # Forward pass with masking
            loss, _, _ = model(images, mask_ratio=mask_ratio)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            # Log batch loss and learning rate
            writer.add_scalar('train/batch_loss', batch_loss, global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            pbar.set_postfix({'batch_loss': batch_loss})
            global_step += 1

        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)

        # Save best model checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_path = os.path.join(output_dir, "checkpoints/fcmae_best.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                },
                save_path
            )
            print(f"Best model saved with loss: {best_loss:.4f}")

        # Periodically save model checkpoint
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(output_dir, f"checkpoints/fcmae_epoch{epoch}.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                },
                save_path
            )


if __name__ == '__main__':
    from configs.args import parse_args
    from modules.fcmae import FCMAE
    from utils.custom_optimizer import warmup_cosine_scheduler
    from dataloader.stl10 import get_stl10_dataloaders, simple_transform

    args = parse_args()
    patch_size = args.stem_stride * 2 ** (len(args.dims) - 1)

    model = FCMAE(
        num_colors=3,
        stem_stride=args.stem_stride,
        depths=args.depths,
        dims=args.dims,
        decoder_depth=1,
        dec_dim=args.decoder_embed_dim,
        patch_size=patch_size,
        norm_pix_loss=args.norm_pix_loss,
    )
    device = args.device
    model.to(device)
    image_size = (96, 96)
    from torchinfo import summary
    input_data = torch.randn(1, 3, 96, 96).to(device)
    summary(model, input_data=input_data)

    # Unsupervised set (no labels)
    unlabelded_loader = get_stl10_dataloaders(
        datatype="unlabeled",  
        batch_size=args.batch_size,
        num_workers=0, 
        data_root='./datasets/stl10',
        shuffle=True,
    )
    unlabelded_loader.dataset.transform = simple_transform(image_size)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.05)
    scheduler = warmup_cosine_scheduler(
        optimizer, 
        warmup_steps=args.warmup_epochs * len(unlabelded_loader),
        total_steps=args.epochs * len(unlabelded_loader),
        min_lr_ratio=0.1
    )
    writer = tensorboard.SummaryWriter(log_dir=os.path.join(args.output, "runs"))

    train_fcmae(
        model=model,
        data_loader=unlabelded_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        mask_ratio=args.mask_ratio,
        save_interval=args.save_interval,
        output_dir=args.output,
        writer=writer,
    )

    writer.close()