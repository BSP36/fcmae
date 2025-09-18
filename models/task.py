import torch.nn as nn

class Classifier(nn.Module):
    """
    ConvNeXtV2 Classification Module.

    Args:
        num_classes (int): Number of output classes
        backbone (nn.Module): Feature extractor module.
        head_init_scale (float): Scaling factor for classifier weights and biases. Default: 1.0.
    """
    def __init__(self, num_classes: int, backbone: nn.Module, head_init_scale: float = 1.0):
        super().__init__()
        self.backbone = backbone
        D = backbone.dims[-1]

        self.norm = nn.LayerNorm(D, eps=1e-6)
        self.head = nn.Linear(D, num_classes)
        # Scale classifier weights and biases
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def forward(self, x):
        # Extract features
        x = self.backbone(x) # (N, D, h ,w)
        # Global average pooling
        x = x.mean(dim=(-2, -1)) # (N, D)
        # Normalize and classify
        x = self.head(self.norm(x))
        return x
