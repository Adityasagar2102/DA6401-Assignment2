import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout          

class VGG11Classifier(nn.Module):

    def __init__(self, num_classes: int = 37, in_channels: int = 3,
                 dropout_p: float = 0.5):
        super(VGG11Classifier, self).__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # I am using CustomDropout.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
        Returns:
            logits: [B, num_classes]
        """
        features = self.encoder(x)
        pooled   = self.avgpool(features)
        return self.classifier(pooled)