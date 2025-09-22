import os
import numpy as np
import torch
import torch.utils.tensorboard as tensorboard
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils.metrics import get_metrics, viz_conf_mat
from typing import Sequence


def train_classifer(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    classes: Sequence[str],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int,
    save_interval: int,
    output_dir: str,
    writer: tensorboard.SummaryWriter,
    best_metrics_target: str = "all_acc",
):
    """Trains a deep learning classifier model

    Args:
        model (torch.nn.Module): Classifer model
        criterion (torch.nn.Module): Loss function
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        classes (Sequence[str]): Classification labels
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run training on (CPU or CUDA).
        num_epochs (int): Number of training epochs.
        save_interval (int): Interval (in epochs) to save checkpoints.
        output_dir (str): Directory to save model checkpoints.
        writer (tensorboard.SummaryWriter): TensorBoard writer for logging.
        best_metrics_target (str, optional): Metric name used to determine the best model checkpoint. Defaults to "all_acc".
    """

    global_step = 0
    best_metrics = float('inf') if "loss" in best_metrics_target else -float('inf')
    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        conf_mat = np.zeros([len(classes), len(classes)])
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs} | Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            epoch_train_loss += batch_loss
            # pred should have a shape (N, num_classes, .....)
            pred = torch.argmax(logits, dim=-1).cpu().numpy().flatten()
            gt = labels.cpu().numpy().flatten()
            conf_mat = conf_mat + confusion_matrix(y_true=gt, y_pred=pred, labels=np.arange(len(classes)))

            writer.add_scalar('train/batch_loss', batch_loss, global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            pbar.set_postfix({'batch_loss': batch_loss})
            global_step += 1

        # Output (train)
        epoch_train_loss = epoch_train_loss / len(train_loader)
        writer.add_scalar('train/epoch_loss', epoch_train_loss, epoch)
        metrics = get_metrics(conf_mat, classes)
        for key, value in metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        # viz_conf_mat(conf_mat, classes, output_path=os.path.join(output_dir, f"results/train_cm_epoch{epoch}.png"))

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        conf_mat = np.zeros([len(classes), len(classes)])
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs} |   Val]"):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                epoch_val_loss += loss.item() * logits.shape[0]
                # pred should have a shape (N, num_classes, .....)
                pred = torch.argmax(logits, dim=1).cpu().numpy().flatten()
                gt = labels.cpu().numpy().flatten()
                conf_mat = conf_mat + confusion_matrix(y_true=gt, y_pred=pred, labels=np.arange(len(classes)))

        # Output (val)
        epoch_val_loss /= len(val_loader.dataset)
        writer.add_scalar(f'val/epoch_loss', epoch_val_loss, epoch)
        metrics = get_metrics(conf_mat, classes)
        for key, value in metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
        viz_conf_mat(conf_mat, classes, output_path=os.path.join(output_dir, f"results/val_cm_epoch{epoch}.png"))

        # Checkpoints
        if metrics[best_metrics_target] > best_metrics:
            best_metrics = metrics[best_metrics_target]
            save_path = os.path.join(output_dir, "checkpoints/ft_best.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    f"best_{best_metrics_target}": best_metrics,
                },
                save_path
            )
            print(f"Best model saved with loss: {best_metrics:.4f}")

        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(output_dir, f"checkpoints/ft_epoch{epoch}.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    f"best_{best_metrics_target}": best_metrics,
                },
                save_path
            )
        

    

if __name__ == '__main__':
    from configs.args import parse_args_ft
    from modules.convnextv2 import ConvNeXtV2
    from modules.task import Classifier
    from utils.custom_optimizer import build_optimizer, warmup_cosine_scheduler
    from dataloader.stl10 import get_stl10_dataloaders, simple_transform
    args = parse_args_ft()

    # Dataloaders
    image_size = (96, 96)
    train_loader = get_stl10_dataloaders(
        datatype="train",  
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        data_root='./datasets/stl10',
        shuffle=True,
    )
    classes = train_loader.dataset.classes
    train_loader.dataset.transform = simple_transform(image_size)
    val_loader = get_stl10_dataloaders(
        datatype="test",  
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        data_root='./datasets/stl10',
        shuffle=False
    )

    # Model
    backbone = ConvNeXtV2(
        in_chans=3,
        stem_stride=args.stem_stride,
        depths=args.depths,
        dims=args.dims,
    )
    if args.ckpt:
        print(f"Use a pretrained model: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        encoder_dict = {k.replace("encoder.", ""):v for k, v in checkpoint['model_state_dict'].items() if "encoder." in k}
        backbone.load_state_dict(encoder_dict)
    model = Classifier(num_classes=len(classes), backbone=backbone)
    device = args.device
    model.to(device)
    
    # Others
    optimizer = build_optimizer(
        model,
        base_lr=args.base_lr,
        bb_mult=args.bb_mult,
        weight_decay=args.weight_decay
    )
    scheduler = warmup_cosine_scheduler(
        optimizer, 
        warmup_steps=args.warmup_epochs * len(train_loader),
        total_steps=args.epochs * len(train_loader),
        min_lr_ratio=0.1
    )
    criterion = torch.nn.CrossEntropyLoss()
    writer = tensorboard.SummaryWriter(log_dir=os.path.join(args.output, "runs"))

    train_classifer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        classes=classes,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_interval=args.save_interval,
        output_dir=args.output,
        writer=writer,
    )

    writer.close()