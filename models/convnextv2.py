# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN
# from utils import LayerNorm, GRN

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
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
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """
    def __init__(
        self,
        in_chans=3, 
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.,
    ):
        super().__init__()
        self.depths = depths
        # downsample_layers
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
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
        # head
        self.head = nn.Identity() # for FCMAE

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
        # head
        x = self.head(x)
        return x

class ConvNeXtV2Sparse(ConvNeXtV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # @torch.no_grad()
    def downsample_mask(self, mask, target_size):
        stride = mask.shape[-1] // target_size
        mask = mask[:, :, ::stride, ::stride]
        return mask
    
    def forward(self, x, mask):
        assert x.shape[-1] == mask.shape[-1]
        assert len(x.shape) == len(mask.shape) == 4
        x = x * (~mask).float()
        # feature extraction
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            mask = self.downsample_mask(mask, target_size=x.shape[-1])
            x = x * (~mask).float()
            for layer in self.stages[i]:
                x = layer(x) * (~mask).float()
                
        return x


class ClassificationHead(nn.Module):
    """ ConvNeXtV2 Classification Head.
    
    Args:
        num_classes (int): Number of classes. Default: 1000
        in_dim (int): Input dimension. Default: 768
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, num_classes, in_dim, head_init_scale=1.):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.head = nn.Linear(in_dim, num_classes)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def forward(self, x):
        return self.head(self.norm(x.mean([-2, -1])))

def convnextv2_atto(**kwargs):
    head = ClassificationHead(num_classes=1000, in_dim=320, head_init_scale=1.)
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    model.update_head(head)
    return model

def convnextv2_femto(**kwargs):
    head = ClassificationHead(num_classes=1000, in_dim=384, head_init_scale=1.)
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    model.update_head(head)
    return model

def convnext_pico(**kwargs):
    head = ClassificationHead(num_classes=1000, in_dim=512, head_init_scale=1.)
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    model.update_head(head)
    return model

def convnextv2_nano(**kwargs):
    head = ClassificationHead(num_classes=1000, in_dim=640, head_init_scale=1.)
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    model.update_head(head)
    return model

def convnextv2_tiny(**kwargs):
    head = ClassificationHead(num_classes=1000, in_dim=768, head_init_scale=1.)
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    model.update_head(head)
    return model

def convnextv2_base(**kwargs):
    head = ClassificationHead(num_classes=1000, in_dim=1024, head_init_scale=1.)
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model.update_head(head)
    return model

def convnextv2_large(**kwargs):
    head = ClassificationHead(num_classes=1000, in_dim=1536, head_init_scale=1.)
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model.update_head(head)
    return model

def convnextv2_huge(**kwargs):
    head = ClassificationHead(num_classes=1000, in_dim=2816, head_init_scale=1.)
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    model.update_head(head)
    return model



if __name__ == "__main__":
    model = ConvNeXtV2Sparse(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320])
    from torchinfo import summary
    input_data = torch.randn(1, 3, 224, 224)
    mask = (torch.randn(1, 1, 224, 224) > 0.0)
    summary(model, input_data=(input_data, mask))