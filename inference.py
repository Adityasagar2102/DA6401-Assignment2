"""Inference utilities — DA6401 Assignment 2.

Run from project root:
    python inference.py --image path/to/cat.jpg --save output.jpg
"""

import argparse
import os

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.multitask import MultiTaskPerceptionModel


# ── Preprocessing (must match training transform) ─────────────────────────────

_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Oxford-IIIT Pet breeds — alphabetically sorted to match dataset.class_to_idx
# (These are inferred from the dataset; verify with your actual dataset.classes)
BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]

# Trimap class → overlay color (RGB)
_MASK_COLORS = {
    0: np.array([0,   200,  50], dtype=np.uint8),   # Foreground → green
    1: np.array([50,   50, 200], dtype=np.uint8),   # Background → blue
    2: np.array([220, 200,   0], dtype=np.uint8),   # Boundary   → yellow
}


# ── Core functions ─────────────────────────────────────────────────────────────

def load_model(
    classifier_path: str = "checkpoints/classifier.pth",
    localizer_path:  str = "checkpoints/localizer.pth",
    unet_path:       str = "checkpoints/unet.pth",
    device: torch.device = None,
) -> MultiTaskPerceptionModel:

    """Instantiate and return MultiTaskPerceptionModel in eval mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskPerceptionModel(
        num_breeds=37, seg_classes=3,
        classifier_path=classifier_path,
        localizer_path=localizer_path,
        unet_path=unet_path,
    ).to(device)
    model.eval()
    return model


def preprocess(image_path: str) -> torch.Tensor:
    """Load an image and return a [1, 3, 224, 224] tensor."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = _TRANSFORM(image=img)["image"]
    return tensor.unsqueeze(0)   # [1, 3, 224, 224]


def run_inference(
    image_path: str,
    model: MultiTaskPerceptionModel = None,
    classifier_path: str = "checkpoints/classifier.pth",
    localizer_path:  str = "checkpoints/localizer.pth",
    unet_path:       str = "checkpoints/unet.pth",
    device: torch.device = None,
) -> dict:
    """
    Run full multi-task pipeline on one image.

    Returns:
        {
          'breed'      : str         — predicted breed name
          'breed_idx'  : int         — class index
          'confidence' : float       — softmax probability of predicted class
          'bbox'       : list[float] — [x_center, y_center, width, height] pixels
          'seg_mask'   : np.ndarray  — [224, 224] int array, values in {0,1,2}
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model(classifier_path, localizer_path, unet_path, device)

    tensor = preprocess(image_path).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    # Classification
    probs     = torch.softmax(outputs["classification"][0], dim=0)
    breed_idx = probs.argmax().item()
    confidence = probs[breed_idx].item()
    breed     = BREEDS[breed_idx] if breed_idx < len(BREEDS) else str(breed_idx)

    # Localization
    bbox = outputs["localization"][0].cpu().tolist()   # [x_c, y_c, w, h]

    # Segmentation
    seg_mask = outputs["segmentation"][0].argmax(dim=0).cpu().numpy()  # [224,224]

    return {
        "breed":      breed,
        "breed_idx":  breed_idx,
        "confidence": confidence,
        "bbox":       bbox,
        "seg_mask":   seg_mask,
    }


def visualize(
    image_path: str,
    result: dict,
    save_path: str = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay bounding box and segmentation mask on the original image.

    Args:
        image_path : path to original image
        result     : dict returned by run_inference()
        save_path  : if given, write the output image here
        alpha      : mask overlay transparency (0=no mask, 1=opaque)

    Returns:
        Annotated image as [H, W, 3] RGB numpy array.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    orig_h, orig_w = img.shape[:2]

    # ── Scale bbox from 224-space → original image space ─────────────────────
    sx, sy = orig_w / 224.0, orig_h / 224.0
    x_c, y_c, bw, bh = result["bbox"]
    x_c *= sx; y_c *= sy; bw *= sx; bh *= sy
    x1, y1 = int(x_c - bw / 2), int(y_c - bh / 2)
    x2, y2 = int(x_c + bw / 2), int(y_c + bh / 2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 50, 50), 2)
    label = f"{result['breed']}  {result['confidence']*100:.1f}%"
    cv2.putText(img, label, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 50, 50), 2,
                lineType=cv2.LINE_AA)

    # ── Overlay segmentation mask ─────────────────────────────────────────────
    mask_224 = result["seg_mask"].astype(np.uint8)
    mask_full = cv2.resize(mask_224, (orig_w, orig_h),
                           interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    for cls_id, color in _MASK_COLORS.items():
        region = mask_full == cls_id
        overlay[region] = (
            img[region] * (1 - alpha) + color * alpha
        ).astype(np.uint8)

    if save_path:
        out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, out)
        print(f"Saved visualization → {save_path}")

    return overlay


def print_result(result: dict) -> None:
    print(f"  Breed       : {result['breed']}  (idx={result['breed_idx']})")
    print(f"  Confidence  : {result['confidence']*100:.2f}%")
    bbox = [f"{v:.1f}" for v in result["bbox"]]
    print(f"  Bounding Box: [x_c={bbox[0]}, y_c={bbox[1]}, w={bbox[2]}, h={bbox[3]}]  (pixels)")
    print(f"  Seg Mask    : shape={result['seg_mask'].shape}, "
          f"classes={np.unique(result['seg_mask']).tolist()}")


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 — Inference")
    parser.add_argument("--image",            required=True, help="Input image path")
    parser.add_argument("--save",             default=None,  help="Save annotated output here")
    parser.add_argument("--classifier_path",  default="checkpoints/classifier.pth")
    parser.add_argument("--localizer_path",   default="checkpoints/localizer.pth")
    parser.add_argument("--unet_path",        default="checkpoints/unet.pth")
    args = parser.parse_args()

    print(f"\nRunning inference on: {args.image}")
    result = run_inference(
        args.image,
        classifier_path=args.classifier_path,
        localizer_path=args.localizer_path,
        unet_path=args.unet_path,
    )
    print_result(result)

    if args.save:
        visualize(args.image, result, save_path=args.save)
