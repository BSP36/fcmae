import torch
import torch.nn as nn
from typing import List, Tuple
from timm.models.layers import trunc_normal_

from .convnextv2 import Block, ConvNeXtV2Sparse
from .patch_utils import patch_wise_normalize, patchify


class FCMAE(nn.Module):
    """
    Full Convolutional Masked AutoEncoder (FCMAE) module.

    Args:
        num_colors (int): Number of input image channels (e.g., 3 for RGB).
        stem_stride (int): Stride for the initial convolutional stem.
        depths (List[int]): Number of blocks at each stage of the encoder.
        dims (List[int]): Feature dimensions at each encoder stage.
        decoder_depth (int): Number of blocks in the decoder.
        dec_dim (int): Feature dimension in the decoder.
        patch_size (int): Size of each image patch (patch_size x patch_size).
        norm_pix_loss (bool, optional): If True, normalize pixel values when computing loss. Defaults to False.
    """
    def __init__(
        self,
        num_colors: int,
        stem_stride: int,
        depths: List[int],
        dims: List[int],
        decoder_depth: int,
        dec_dim: int,
        patch_size: int,
        norm_pix_loss: bool=False
    ):
        super().__init__()
        # sanity check
        assert len(depths) == len(dims)
        assert stem_stride * 2 ** (len(depths) - 1) == patch_size

        # configs
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss

        # Encoder
        self.encoder = ConvNeXtV2Sparse(
            in_chans=num_colors,
            stem_stride=stem_stride,
            depths=depths,
            dims=dims
        )
        # Mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, dec_dim, 1, 1))
        # Decoder
        self.proj = nn.Conv2d(in_channels=dims[-1],  out_channels=dec_dim, kernel_size=1)
        decoder = [Block(dec_dim) for _ in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv2d(
            in_channels=dec_dim,
            out_channels=patch_size * patch_size * num_colors,
            kernel_size=1
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)
    
    def gen_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generate a random binary mask for input images.

        Args:
            x (torch.Tensor): Input images of shape (N, C, H, W).
            mask_ratio (float): Ratio of patches to mask (between 0 and 1).

        Returns:
            torch.Tensor: Boolean mask of shape (N, L), where 0 indicates a kept patch and 1 indicates a masked patch.
        """
        N, _, H, W = x.shape
        L = (H // self.patch_size) * (W // self.patch_size)
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask.bool()
    
    
    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a random mask and extract encoder features.

        Args:
            x (torch.Tensor): Input images of shape (N, C, H, W).
            mask_ratio (float): Ratio of patches to mask (between 0 and 1).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the encoder feature map (N, D, h, w) and the mask (N, L).
        """
        # generate random masks
        mask = self.gen_random_mask(x, mask_ratio) # [N, L], 0 is keep, 1 is remove
        # encoding
        N, _, H, W = x.shape
        p = self.patch_size
        assert H % p == W % p == 0, f"Invalid image size: {H}x{W}"
        # upsampling
        mask_upsampled = mask.reshape(N, 1, H // p, W // p).\
                    repeat_interleave(p, axis=2).repeat_interleave(p, axis=3)
        x = self.encoder(x, mask_upsampled)

        return x, mask

    def forward_decoder(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Estimate the removed patches from the feature map.

        Args:
            x (torch.Tensor): Feature map of shape (N, emb_dim, h=H//p, w=W//p).
            mask (torch.Tensor): Binary mask of shape (N, L), where 0 indicates a kept patch and 1 indicates a masked patch.

        Returns:
            torch.Tensor: Predicted patches of shape (N, D=p*p*C, h, w).
        """
        x = self.proj(x)
        # append mask token
        n, _, h, w = x.shape
        mask = mask.reshape(n, 1, h, w).type_as(x)
        mask_token = self.mask_token.repeat(n, 1, h, w)
        x = x * (1.0 - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the reconstruction loss for FCMAE.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).
            pred (torch.Tensor): Predicted patches of shape (N, D, h, w).
            mask (torch.Tensor): Binary mask of shape (N, L), where 0 indicates a kept patch and 1 indicates a masked patch.

        Returns:
            torch.Tensor: Mean squared error (MSE) loss computed over masked (removed) patches.
        """
        N, D, h, w = pred.shape
        pred = pred.reshape(N, D, h * w) # (N, D, L)
        pred = torch.einsum('ndl->nld', pred) # (N, L, D)

        target = patchify(imgs, self.patch_size) # (N, L, D)
        if self.norm_pix_loss:
            target = patch_wise_normalize(target)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (N, L)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self, 
        imgs: torch.Tensor, 
        mask_ratio: float = 0.6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for FCMAE.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).
            mask_ratio (float, optional): Ratio of patches to mask. Defaults to 0.6.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - loss (torch.Tensor): Reconstruction loss over masked patches.
                - pred (torch.Tensor): Predicted patches of shape (N, D, h, w).
                - mask (torch.Tensor): Binary mask of shape (N, L).
        """
        features, mask = self.forward_encoder(imgs.clone(), mask_ratio)
        # print("feature", features.max().item(), features.min().item())
        pred = self.forward_decoder(features, mask)
        # print(imgs.max().item(), imgs.min().item(), pred.max().item(), pred.min().item())
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask