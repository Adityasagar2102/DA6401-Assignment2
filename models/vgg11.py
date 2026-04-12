from typing import Dict, Tuple, Union
import torch
import torch.nn as nn

# Import your custom dropout layer!
from .layers import CustomDropout

class VGG11Encoder(nn.Module):

    def __init__(self, in_channels: int = 3):
        super(VGG11Encoder, self).__init__()

        # Block 1: 1 Conv layer
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 1 Conv layer
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 2 Conv layers
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 2 Conv layers
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: 2 Conv layers
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

    
        # pass through block 1 and save the feature map before pooling 
        f1 = self.block1(x)
        x = self.pool1(f1)

        # Block2 
        f2 = self.block2(x)
        x = self.pool2(f2)

        # Block3 
        f3 = self.block3(x)
        x = self.pool3(f3)

        # Block 4
        f4 = self.block4(x)
        x = self.pool4(f4)

        # Block 5
        f5 = self.block5(x)
        bottleneck = self.pool5(f5)

        if return_features:
          feature_dict = {
            'f1': f1,  # 64 channels
            'f2': f2,  # 128 channels
            'f3': f3,  # 256 channels
            'f4': f4,  # 512 channels
            'f5': f5   # 512 channels
          }
          return bottleneck, feature_dict

        # If Task 1 (Classification) is calling this, just return the bottleneck
        return bottleneck



class VGG11(nn.Module):
    """Full VGG11 model for classification."""
    
    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super(VGG11, self).__init__()
        
        # Instantiate the encoder we just built
        self.features = VGG11Encoder(in_channels=3)
        
        # Hardcoded for 224x224 input as per TA instructions
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # The Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We don't need intermediate features for simple classification
        x = self.features(x, return_features=False)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x