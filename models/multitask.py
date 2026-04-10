import os
import gdown
import torch
import torch.nn as nn

from .vgg11 import VGG11
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
    
        super(MultiTaskPerceptionModel, self).__init__()

        # ── gdown downloads (paste your IDs here before submission) ──────────
        gdown.download(id='1zz9vQLE-3Q7xjfFetJMP-ST3BM39jnWu', output=classifier_path, quiet=False)
        gdown.download(id='12do8tf-FcNqh7lrhqBc2Q1OKEfUIEcTX', output=localizer_path, quiet=False)
        gdown.download(id='1Jmo5WHMDFYnLJjmfpa0Ot6KvGnZwvrTL', output=unet_path, quiet=False)
        # ─────────────────────────────────────────────────────────────────────

        # Instantiate individual models with their architectures
        cls_model = VGG11(num_classes=num_breeds, dropout_p=0.5)
        loc_model = VGG11Localizer(in_channels=in_channels)
        seg_model = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Load checkpoints if available
        if os.path.exists(classifier_path):
            state = torch.load(classifier_path, map_location="cpu")
            missing, unexpected = cls_model.load_state_dict(state, strict=False)

            print("[DEBUG] Classifier missing keys:", missing)
            print("[DEBUG] Classifier unexpected keys:", unexpected)
            print(f"[MultiTask] Loaded classifier from {classifier_path}")
        else:
            print(f"[MultiTask] WARNING: {classifier_path} not found — random weights.")

        if os.path.exists(localizer_path):
            loc_model.load_state_dict(
                torch.load(localizer_path, map_location="cpu"))
            print(f"[MultiTask] Loaded localizer from {localizer_path}")
        else:
            print(f"[MultiTask] WARNING: {localizer_path} not found — random weights.")

        if os.path.exists(unet_path):
            seg_model.load_state_dict(
                torch.load(unet_path, map_location="cpu"))
            print(f"[MultiTask] Loaded UNet from {unet_path}")
        else:
            print(f"[MultiTask] WARNING: {unet_path} not found — random weights.")

        # VGG11 stores its encoder as self.features (a VGG11Encoder)
        self.shared_encoder = VGG11().features
        self.shared_encoder.load_state_dict(cls_model.features.state_dict())          # VGG11Encoder

        # ── Classification head ───────────────────────────────────────────────
        self.avgpool          = cls_model.avgpool          # AdaptiveAvgPool2d(7,7)
        self.classifier_head  = cls_model.classifier       # Sequential(Linear→…→Linear)

        # ── Localization head ─────────────────────────────────────────────────
        # loc_model.regressor expects a flat [B, 512*7*7] input
        self.localizer_head   = loc_model.regressor

        # ── Segmentation decoder ──────────────────────────────────────────────
        self.up1       = seg_model.up1
        self.dec1      = seg_model.dec1
        self.up2       = seg_model.up2
        self.dec2      = seg_model.dec2
        self.up3       = seg_model.up3
        self.dec3      = seg_model.dec3
        self.up4       = seg_model.up4
        self.dec4      = seg_model.dec4
        self.up5       = seg_model.up5
        self.dec5      = seg_model.dec5
        self.final_conv = seg_model.final_conv

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, 224, 224] normalized input image
        Returns:
            dict with keys 'classification', 'localization', 'segmentation'
        """
        # ── Shared encoder (run ONCE) ─────────────────────────────────────────
        bottleneck, skip = self.shared_encoder(x, return_features=True)
        # bottleneck : [B, 512, 7, 7]
        # skip       : {f1:…, f2:…, f3:…, f4:…, f5:…}

        # ── Classification branch ─────────────────────────────────────────────
        cls_x    = self.avgpool(bottleneck)           # [B, 512, 7, 7]
        cls_flat = torch.flatten(cls_x, 1)            # [B, 512*7*7]
        cls_out  = self.classifier_head(cls_flat)     # [B, num_breeds]

        # ── Localization branch (reuses the same pooled features) ─────────────
        raw_coords = self.localizer_head(cls_flat)    # [B, 4]
        loc_out    = torch.sigmoid(raw_coords) * 224.0  # pixel space [0, 224]

        # ── Segmentation branch ───────────────────────────────────────────────
        d = self.up1(bottleneck)
        d = torch.cat([skip['f5'], d], dim=1)
        d = self.dec1(d)

        d = self.up2(d)
        d = torch.cat([skip['f4'], d], dim=1)
        d = self.dec2(d)

        d = self.up3(d)
        d = torch.cat([skip['f3'], d], dim=1)
        d = self.dec3(d)

        d = self.up4(d)
        d = torch.cat([skip['f2'], d], dim=1)
        d = self.dec4(d)

        d = self.up5(d)
        d = torch.cat([skip['f1'], d], dim=1)
        d = self.dec5(d)

        seg_out = self.final_conv(d)    # [B, seg_classes, 224, 224]

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }
