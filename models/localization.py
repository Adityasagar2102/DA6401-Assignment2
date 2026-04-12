import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Localizer(nn.Module):

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11Localizer,self).__init__()

        self.encoder = VGG11Encoder(in_channels = in_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((7,7))

        self.regressor = nn.Sequential(
          nn.Linear(512*7*7, 4096),
          nn.ReLU(inplace=True),
          CustomDropout(p=dropout_p),

          nn.Linear(4096, 4096),
          nn.ReLU(inplace = True),
          CustomDropout(p=dropout_p),

          nn.Linear(4096,4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x, return_features = False)
        features = self.avgpool(features)

        features = torch.flatten(features,1)

        raw_coords = self.regressor(features)

        out = torch.sigmoid(raw_coords)* 224.0

        return out






