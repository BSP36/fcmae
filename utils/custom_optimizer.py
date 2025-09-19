import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from typing import Iterable, Tuple


def split_decay_no_decay(named_params: Iterable[Tuple[str, nn.Parameter]]) -> Tuple[list, list]:
    """
    Splits model parameters into two groups: those to apply weight decay to, and those not to.
    This is typically used in optimizer setup to exclude normalization layers and bias terms
    from weight decay.

    Args:
        named_params (Iterable[Tuple[str, nn.Parameter]]): Iterable of (name, parameter) pairs.

    Returns:
        Tuple[list, list]: A tuple containing two lists:
            - decay: parameters to apply weight decay to
            - no_decay: parameters to exclude from weight decay
    """
    decay, no_decay = [], []
    for name, param in named_params:
        if not param.requires_grad:
            continue  # Skip frozen parameters
        # Exclude bias and normalization parameters from weight decay
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_optimizer(
    model: nn.Module,
    base_lr: float,
    bb_mult: float = 0.1,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.999),
    backbone_attr: str = "backbone",
) -> torch.optim.Optimizer:
    """
    Builds an AdamW optimizer with separate parameter groups for backbone and non-backbone layers,
    and for parameters with/without weight decay.

    Args:
        model (nn.Module): The model containing parameters to optimize.
        base_lr (float): Base learning rate for non-backbone parameters.
        bb_mult (float, optional): Multiplier for backbone learning rate. Defaults to 0.1.
        weight_decay (float, optional): Weight decay for applicable parameters. Defaults to 0.05.
        betas (Tuple[float, float], optional): AdamW betas. Defaults to (0.9, 0.999).
        backbone_attr (str, optional): Attribute name for the backbone module. Defaults to "backbone".

    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer.
    """
    assert hasattr(model, backbone_attr), f"Model has no attribute '{backbone_attr}'"
    backbone = getattr(model, backbone_attr)

    # Collect parameter IDs for backbone
    bb_ids = {id(p) for p in backbone.parameters()}
    bb_named, other_named = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in bb_ids:
            bb_named.append((n, p))
        else:
            other_named.append((n, p))

    # Split parameters into decay/no_decay groups
    bb_decay, bb_no_decay = split_decay_no_decay(bb_named)
    other_decay, other_no_decay = split_decay_no_decay(other_named)

    # Define parameter groups for optimizer
    param_groups = [
        {"params": other_decay, "lr": base_lr, "weight_decay": weight_decay},
        {"params": other_no_decay, "lr": base_lr, "weight_decay": 0.0},
        {"params": bb_decay, "lr": base_lr * bb_mult, "weight_decay": weight_decay},
        {"params": bb_no_decay, "lr": base_lr * bb_mult, "weight_decay": 0.0},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=betas)

    # Sanity check: ensure all trainable parameters are included and no extras
    all_trainable = {id(p) for _, p in model.named_parameters() if p.requires_grad}
    grouped = set()
    for g in optimizer.param_groups:
        for p in g["params"]:
            grouped.add(id(p))
    missing = all_trainable - grouped
    extra = grouped - all_trainable
    assert len(missing) == 0, f"Missing parameters: {len(missing)}"
    assert len(extra) == 0, f"Extra parameters: {len(extra)}"

    return optimizer


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """
    Creates a learning rate scheduler with a linear warmup followed by cosine decay.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled.
        warmup_steps (int): Number of steps for linear warmup.
        total_steps (int): Total number of training steps.
        min_lr_ratio (float, optional): Minimum learning rate as a ratio of the initial LR. Defaults to 0.05.

    Returns:
        LambdaLR: PyTorch learning rate scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
