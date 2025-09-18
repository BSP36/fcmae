import torch
import torch.nn as nn

def patchify(image: torch.Tensor, P: int) -> torch.Tensor:
    """images to patches

    Args:
        image (torch.Tensor): input images (N, C, H, W)
        P (int): patch size

    Returns:
        torch.Tensor: patches (N, L, P*P*C)
    """
    N, C, H, W = image.shape
    L = (W // P) * (H // P)
    x = image.reshape((N, C, H // P, P, W // P, P))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape((N, L, P * P * C))
    return x

def patch_wise_normalize(patch: torch.Tensor, eps: float=1e-6):
    """Normalize patches by subtracting the mean and dividing by the standard deviation.

    Args:
        patch (torch.Tensor): Input patches with shape (N, L, D), where N is the batch size,
        eps (float, optional): A small value to prevent division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Normalized patches with the same shape as the input (N, L, D).
    """
    mean = patch.mean(dim=-1, keepdim=True)
    var = patch.var(dim=-1, keepdim=True)
    return (patch - mean) / (var + eps) ** 0.5


def patch_wise_dernormalize(pred: torch.Tensor, image: torch.Tensor):
    """Denormalize patch-wise predictions using the mean and variance of the corresponding image patches.

    Args:
        pred (torch.Tensor): Predicted tensor of shape (N, D, h, w).
        image (torch.Tensor): Original image tensor of shape (N, C, H, W).

    Returns:
        torch.Tensor: Denormalized predictions with shape (N, D, h, w).
    """
    N, D, h, w = pred.shape
    pred = pred.reshape(N, D, h * w)  # (N, D, L)
    pred = torch.einsum('ndl->nld', pred)  # (N, L, D)
    # normalization
    target = patchify(image, image.shape[-1] // w)  # (N, L, D)
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)

    pred = pred * var ** 0.5 + mean
    pred = torch.einsum('nld->ndl', pred)
    return pred.reshape(N, D, h, w)
