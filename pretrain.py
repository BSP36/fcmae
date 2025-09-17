import os
import torch
import torch.utils.tensorboard as tensorboard
from tqdm import tqdm

def train_fcmae(
    model,
    data_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    writer,
):
    
    model.train()
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in pbar:
            images = images.to(device)
            loss, _, _ = model(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            writer.add_scalar('train/batch_loss', batch_loss, global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            pbar.set_postfix({'batch_loss': batch_loss})
            global_step += 1

        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)

    

if __name__ == '__main__':
    from configs.args import parse_args
    from models.fcmae import FCMAE
    from dataloader.stl10 import get_stl10_dataloaders, simple_transform
    # [2, 2, 6, 2], dims=[40, 80, 160, 320]
    args = parse_args()
    patch_size = args.stem_stride * 2 ** (len(args.dims) - 1)
    print(patch_size)

    model = FCMAE(
        num_colors=3,
        stem_stride=args.stem_stride,
        depths=args.depths,
        dims=args.dims,
        decoder_depth=1,
        decoder_embed_dim=args.decoder_embed_dim,
        patch_size=patch_size,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
    )
    # device = args.device
    device = "mps"
    model.to(device)
    image_size = (96, 96)
    # print(model)
    from torchinfo import summary
    input_data = torch.randn(1, 3, 96, 96).to(device)
    summary(model, input_data=input_data)

    # Unsupervised set (no labels)
    unlabelded_loader = get_stl10_dataloaders(
        datatype="unlabeled",  
        batch_size=args.batch_size,
        num_workers=1, 
        data_root='./datasets/stl10'
    )
    print(len(unlabelded_loader.dataset))
    transform = simple_transform(image_size)
    unlabelded_loader.dataset.transform = transform

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    writer = tensorboard.SummaryWriter(log_dir=os.path.join(args.output, "logs"))

    train_fcmae(
        model=model,
        data_loader=unlabelded_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        writer=writer,
    )

    writer.close()