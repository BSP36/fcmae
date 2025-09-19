import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from .layer_utils import LayerNorm, GRN
from typing import List

class Block(nn.Module):
    """ConvNeXtV2 Block (as in the original implementation)

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default is 0.0.
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels.
        depths (tuple(int)): Number of blocks at each stage.
        dims (int): Feature dimension at each stage.
        drop_path_rate (float): Stochastic depth rate. default: 0.0
    """
    def __init__(
        self,
        in_chans: int, 
        stem_stride: int,
        depths: List[int],
        dims: List[int], 
        drop_path_rate: float=0.0,
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims
        # downsample_layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=stem_stride, stride=stem_stride),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(len(depths) - 1):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # stages
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(depths)):
            stage = nn.ModuleList([Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    def update_head(self, head: nn.Module):
        self.head = head

    def forward(self, x):
        # feature extraction
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            for layer in self.stages[i]:
                x = layer(x)
        return x

class ConvNeXtV2Sparse(ConvNeXtV2):
    """
    ConvNeXtV2 for FCMAE.

    Args:
        Same as ConvNeXtV2, plus:
        mask (torch.Tensor): Binary mask indicating valid input regions (1 = masked, 0 = valid).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def downsample_mask(self, mask, target_size):
        """
        Downsample the mask to match the spatial size of the feature map.

        Args:
            mask (torch.Tensor): Input mask of shape (N, 1, H, W).
            target_size (int): Target spatial size after downsampling.

        Returns:
            torch.Tensor: Downsampled mask.
        """
        stride = mask.shape[-1] // target_size
        mask = mask[:, :, ::stride, ::stride]
        return mask

    def forward(self, x, mask):
        """
        Forward pass with sparse masking.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            mask (torch.Tensor): Binary mask of shape (N, 1, H, W).

        Returns:
            torch.Tensor: Output tensor after applying sparse masking.
        """
        assert x.shape[-1] == mask.shape[-1], "Input and mask spatial dimensions must match."
        assert len(x.shape) == len(mask.shape) == 4, "Input and mask must be 4D tensors."
        x = x * (~mask).float()  # Zero out masked regions

        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            mask = self.downsample_mask(mask, target_size=x.shape[-1])
            x = x * (~mask).float()
            for layer in self.stages[i]:
                x = layer(x) * (~mask).float()
        return x