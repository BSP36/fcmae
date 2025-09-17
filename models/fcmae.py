import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from timm.models.layers import trunc_normal_
from .convnextv2 import Block, ConvNeXtV2Sparse

class FCMAE(nn.Module):
    """ Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone
    """
    def __init__(
        self,
        num_colors: int,
        stem_stride: int,
        depths: List[int],
        dims: List[int],
        decoder_depth: int,
        decoder_embed_dim: int,
        patch_size: int,
        mask_ratio: float,
        norm_pix_loss: bool=False
    ):
        super().__init__()
        assert len(depths) == len(dims)
        # assert image_shape[0] % patch_size == image_shape[1] % patch_size == 0
        assert stem_stride * 2 ** (len(depths) - 1) == patch_size

        # configs
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # Encoder
        self.encoder = ConvNeXtV2Sparse(
            in_chans=num_colors,
            stem_stride=stem_stride,
            depths=depths,
            dims=dims
        )
        # Decoder
        self.proj = nn.Conv2d(in_channels=dims[-1],  out_channels=decoder_embed_dim, kernel_size=1)
        # Mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(decoder_embed_dim) for _ in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size * patch_size * num_colors,
            kernel_size=1
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)
    
    def patchify(self, imgs):
        N, C, H, W = imgs.shape
        p = self.patch_size

        h = H // p
        w = W // p
        x = imgs.reshape((N, C, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape((N, h * w, p * p * C))
        return x

    # def unpatchify(self, x):
    #     N, L, D = x.shape
    #     p = self.patch_size
    #     H, W = self.img_size
    #     C = D // (p * p) # color
    #     h = H // p
    #     w = W // p
    #     assert L == h * w
        
    #     x = x.reshape((N, h, w, p, p, C))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape((N, C, h * p, w * p))
    #     return imgs

    def gen_random_mask(self, x, mask_ratio):
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
    
    def forward_encoder(self, imgs, mask_ratio):
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio) # [N, L], 0 is keep, 1 is remove
        # encoding
        N, _, H, W = imgs.shape
        p = self.patch_size
        assert H % p == W % p == 0, f"Invalid image size: {H}x{W}"
        # upsampling
        mask_upsampled = mask.reshape(N, 1, H // p, W // p).\
                    repeat_interleave(p, axis=2).repeat_interleave(p, axis=3)
        x = self.encoder(imgs, mask_upsampled)

        return x, mask

    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        n, _, h, w = x.shape
        mask = mask.reshape(n, 1, h, w).type_as(x)
        mask_token = self.mask_token.repeat(n, 1, h, w)
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, D=p*p*C, H//p, W//p]
        mask: [N, L], 0 is keep, 1 is remove
        """
        N, D, _, _ = pred.shape
        pred = pred.reshape(N, D, -1) # (N, D, L)
        pred = torch.einsum('ndl->nld', pred) # (N, L, D)

        target = self.patchify(imgs) # (N, L, D)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, labels=None, mask_ratio=0.6):
        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask