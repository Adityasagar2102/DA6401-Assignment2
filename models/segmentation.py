import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two-conv decoder block: Conv→BN→ReLU→Conv→BN→ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super(VGG11UNet, self).__init__()

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Stage 1: bottleneck [B,512,7,7] → upsample → cat f5 → dec
        self.up1  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)   # →[B,512,14,14]
        self.dec1 = _dec_block(512 + 512, 512)                               # cat f5(512)

        # Stage 2: [B,512,14,14] → upsample → cat f4 → dec
        self.up2  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)   # →[B,512,28,28]
        self.dec2 = _dec_block(512 + 512, 512)                               # cat f4(512)

        # Stage 3: [B,512,28,28] → upsample → cat f3 → dec
        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # →[B,256,56,56]
        self.dec3 = _dec_block(256 + 256, 256)                               # cat f3(256)

        # Stage 4: [B,256,56,56] → upsample → cat f2 → dec
        self.up4  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # →[B,128,112,112]
        self.dec4 = _dec_block(128 + 128, 128)                               # cat f2(128)

        # Stage 5: [B,128,112,112] → upsample → cat f1 → dec
        self.up5  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    # →[B,64,224,224]
        self.dec5 = _dec_block(64 + 64, 64)                                  # cat f1(64)

        # Final pixel-wise classifier
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, features = self.encoder(x, return_features=True)
        
        d1 = self.up1(bottleneck)
        d1 = torch.cat([features['f5'], d1], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([features['f4'], d2], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([features['f3'], d3], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        d4 = torch.cat([features['f2'], d4], dim=1)
        d4 = self.dec4(d4)
        
        d5 = self.up5(d4)
        d5 = torch.cat([features['f1'], d5], dim=1)
        d5 = self.dec5(d5)
        
        return self.final_conv(d5)