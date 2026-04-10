"""Oxford-IIIT Pet multi-task dataset loader."""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OxfordIIITPetDataset(Dataset):
    """
    Oxford-IIIT Pet multi-task dataset.

    Returns per sample:
        image : FloatTensor [3, 224, 224]  — normalized
        label : LongTensor  scalar         — breed index in [0, 36]
        bbox  : FloatTensor [4]            — [x_center, y_center, w, h] pixel coords
        mask  : LongTensor  [224, 224]     — trimap class in {0, 1, 2}
    """

    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir  = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir  = os.path.join(root_dir, "annotations", "trimaps")
        self.xmls_dir   = os.path.join(root_dir, "annotations", "xmls")

        # ── Collect valid samples (image + mask + xml must all exist) ─────────
        valid = []
        for fname in os.listdir(self.images_dir):
            if not fname.endswith(".jpg"):
                continue
            stem = os.path.splitext(fname)[0]
            if (os.path.exists(os.path.join(self.masks_dir, stem + ".png")) and
                    os.path.exists(os.path.join(self.xmls_dir,  stem + ".xml"))):
                valid.append(stem)

        self.filenames = sorted(valid)

        # ── Build breed → index mapping ───────────────────────────────────────
        self.classes = [
            "Abyssinian", "american_bulldog", "american_pit_bull_terrier",
            "basset_hound", "beagle", "Bengal", "Birman", "Bombay", "boxer",
            "British_Shorthair", "chihuahua", "Egyptian_Mau",
            "english_cocker_spaniel", "english_setter", "german_shorthaired",
            "great_pyrenees", "havanese", "japanese_chin", "keeshond",
            "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland",
            "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue",
            "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu",
            "Siamese", "Sphynx", "staffordshire_bull_terrier", "wheaten_terrier",
            "yorkshire_terrier"
        ]
        self.class_to_idx = {b: i for i, b in enumerate(self.classes)}

        # ── Albumentations pipeline ───────────────────────────────────────────
        self.transform = A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5), # Add flipping
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3), # Add slight rotations
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3), # Add lighting changes
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.1,
            ),
        )

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        stem = self.filenames[idx]

        # 1. Image
        img = cv2.imread(os.path.join(self.images_dir, stem + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # 2. Classification label
        breed_name  = "_".join(stem.split("_")[:-1])
        class_label = self.class_to_idx[breed_name]

        # 3. Segmentation mask — trimaps have values {1, 2, 3}; shift to {0, 1, 2}
        mask = cv2.imread(
            os.path.join(self.masks_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE
        )
        mask = np.clip(mask.astype(np.int32) - 1, 0, 2).astype(np.uint8)

        # 4. Bounding box — parse XML, clamp to image bounds
        xml_path = os.path.join(self.xmls_dir, stem + ".xml")
        tree = ET.parse(xml_path)
        bndbox = tree.getroot().find("object").find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        # Clamp to valid image bounds before passing to albumentations
        xmin = max(0.0, min(xmin, orig_w - 1))
        ymin = max(0.0, min(ymin, orig_h - 1))
        xmax = max(xmin + 1, min(xmax, orig_w))
        ymax = max(ymin + 1, min(ymax, orig_h))

        # 5. Apply transforms
        transformed = self.transform(
            image=img,
            masks=[mask],
            bboxes=[[xmin, ymin, xmax, ymax]],
            class_labels=[class_label],
        )

        image_tensor = transformed["image"]
        mask_tensor  = torch.tensor(
            transformed["masks"][0], dtype=torch.long)

        # 6. Convert bbox back from pascal_voc → (x_center, y_center, w, h)
        if len(transformed["bboxes"]) > 0:
            t_xmin, t_ymin, t_xmax, t_ymax = transformed["bboxes"][0]
        else:
            # Fallback: full image
            t_xmin, t_ymin, t_xmax, t_ymax = 0.0, 0.0, 224.0, 224.0

        w        = t_xmax - t_xmin
        h        = t_ymax - t_ymin
        x_center = t_xmin + w / 2.0
        y_center = t_ymin + h / 2.0

        bbox_tensor = torch.tensor(
            [x_center, y_center, w, h], dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": torch.tensor(class_label, dtype=torch.long),
            "bbox":  bbox_tensor,
            "mask":  mask_tensor,
        }
