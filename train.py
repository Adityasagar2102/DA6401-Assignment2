import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score as sk_f1
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.vgg11 import VGG11

import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3, eps: float = 1e-6) -> torch.Tensor:

    probs = torch.softmax(logits, dim=1)
    loss = 0.0
    for c in range(num_classes):
        p = probs[:, c, :, :]
        t = (targets == c).float()
        intersection = (p * t).sum(dim=(1, 2))
        cardinality  = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice = (2.0 * intersection + eps) / (cardinality + eps)
        loss += (1.0 - dice).mean()
    return loss / num_classes


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loaders(args):
    """80/20 train/val split, seed=42, with STRICTLY separated transforms."""
    dataset = OxfordIIITPetDataset(root_dir=args.data_dir)

    val_len   = int(len(dataset) * 0.2)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    # Clean copy for validation — NO augmentation, just resize + normalise
    val_dataset_clean = copy.deepcopy(dataset)
    val_dataset_clean.transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"],
                                min_visibility=0.1))
    val_ds.dataset = val_dataset_clean

    kw = dict(num_workers=args.num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **kw),
        dataset,
    )


def _load_pretrained_vgg11bn(model: VGG11) -> bool:
    try:
        import torchvision.models as tv

        # Use new-style weights API; fall back to old 'pretrained=True' for
        # older torchvision versions.
        try:
            tv_model = tv.vgg11_bn(weights=tv.VGG11_BN_Weights.DEFAULT)
        except AttributeError:
            tv_model = tv.vgg11_bn(pretrained=True)

        tv_sd = tv_model.features.state_dict()

        # Map: torchvision flat index → our named block submodule path
        # Each entry covers both Conv2d weights and BN parameters.
        mapping = {
            '0':  'block1.0',   # Conv2d(3 → 64)
            '1':  'block1.1',   # BatchNorm2d(64)
            '4':  'block2.0',   # Conv2d(64 → 128)
            '5':  'block2.1',   # BatchNorm2d(128)
            '8':  'block3.0',   # Conv2d(128 → 256)
            '9':  'block3.1',   # BatchNorm2d(256)
            '11': 'block3.3',   # Conv2d(256 → 256)
            '12': 'block3.4',   # BatchNorm2d(256)
            '15': 'block4.0',   # Conv2d(256 → 512)
            '16': 'block4.1',   # BatchNorm2d(512)
            '18': 'block4.3',   # Conv2d(512 → 512)
            '19': 'block4.4',   # BatchNorm2d(512)
            '22': 'block5.0',   # Conv2d(512 → 512)
            '23': 'block5.1',   # BatchNorm2d(512)
            '25': 'block5.3',   # Conv2d(512 → 512)
            '26': 'block5.4',   # BatchNorm2d(512)
        }

        new_sd = {}
        for tv_key, tv_val in tv_sd.items():
            idx = tv_key.split('.')[0]          # e.g. '0', '1', '4' …
            suffix = '.'.join(tv_key.split('.')[1:])  # e.g. 'weight', 'bias' …
            if idx in mapping:
                new_sd[f"{mapping[idx]}.{suffix}"] = tv_val

        missing, unexpected = model.features.load_state_dict(new_sd, strict=False)
        print(f"[Pretrained] VGG11-BN weights loaded successfully. "
              f"Missing={len(missing)}, Unexpected={len(unexpected)}")
        if missing:
            print(f"  Missing keys (expected only ReLU/pool — no parameters): {missing[:5]}")
        return True

    except Exception as e:
        print(f"[Pretrained] Could not load pretrained weights — training from scratch. Reason: {e}")
        return False


def _transfer_encoder(model, checkpoint_path: str,
                       src_prefix: str, dst_prefix: str) -> None:
    """Copy encoder weights from a saved checkpoint into a different model."""
    if not os.path.exists(checkpoint_path):
        print(f"  [transfer] {checkpoint_path} not found — skipping.")
        return
    state = torch.load(checkpoint_path, map_location="cpu")
    new_state = {
        k.replace(src_prefix, dst_prefix): v
        for k, v in state.items() if k.startswith(src_prefix)
    }
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"  [transfer] Loaded encoder from {checkpoint_path}. "
          f"Missing={len(missing)}, Unexpected={len(unexpected)}")


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation → [H,W,3] uint8 RGB."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().permute(1, 2, 0).numpy()
    return ((img * std + mean).clip(0, 1) * 255).astype(np.uint8)


def dice_score(preds: torch.Tensor, targets: torch.Tensor,
               num_classes: int = 3, eps: float = 1e-6) -> float:
    pred_labels = preds.argmax(dim=1)
    score = 0.0
    for c in range(num_classes):
        p = (pred_labels == c).float()
        t = (targets     == c).float()
        score += (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return (score / num_classes).item()


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds.argmax(dim=1) == targets).float().mean().item()


def _compute_iou_single(pred: np.ndarray, target: np.ndarray,
                         eps: float = 1e-6) -> float:
    """IoU for a single (x_c, y_c, w, h) box pair in pixel coords."""
    def corners(b):
        return b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    px1,py1,px2,py2 = corners(pred)
    tx1,ty1,tx2,ty2 = corners(target)
    iw = max(0.0, min(px2,tx2) - max(px1,tx1))
    ih = max(0.0, min(py2,ty2) - max(py1,ty1))
    inter = iw * ih
    union = pred[2]*pred[3] + target[2]*target[3] - inter
    return float(inter / (union + eps))


# ═══════════════════════════════════════════════════════════════════════════════
#  WandB visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def log_activation_distributions(model: VGG11, val_loader,
                                   device, run_name: str = "classifier") -> None:
    model.eval()
    activations = {}
    h1 = model.features.block2.register_forward_hook(
        lambda _,__,o: activations.update({"conv3_block2": o.detach().cpu()}))
    h2 = model.features.block5[3].register_forward_hook(
        lambda _,__,o: activations.update({"last_conv_block5": o.detach().cpu()}))
    batch = next(iter(val_loader))
    with torch.no_grad():
        model(batch["image"].to(device))
    h1.remove(); h2.remove()
    for name, acts in activations.items():
        flat = acts.flatten().numpy()
        wandb.log({
            f"{run_name}/{name}_histogram": wandb.Histogram(flat),
            f"{run_name}/{name}_mean":      float(flat.mean()),
            f"{run_name}/{name}_std":       float(flat.std()),
        })
    print("  [WandB 2.1] Logged activation distributions.")


def log_feature_maps(model: VGG11, val_loader, device,
                      n_filters: int = 16) -> None:
    model.eval()
    feat = {}
    h1 = model.features.block1.register_forward_hook(
        lambda _,__,o: feat.update({"first_conv_block1": o.detach().cpu()}))
    h2 = model.features.block5.register_forward_hook(
        lambda _,__,o: feat.update({"last_conv_block5":  o.detach().cpu()}))
    batch = next(iter(val_loader))
    with torch.no_grad():
        model(batch["image"][:1].to(device))
    h1.remove(); h2.remove()
    orig_img = _denorm(batch["image"][0])
    panels = {"feature_maps/input_image": wandb.Image(orig_img, caption="input")}
    for layer_name, fmaps in feat.items():
        maps   = fmaps[0, :n_filters]
        n_cols = 4
        n_rows = (len(maps) + n_cols - 1) // n_cols
        h, w   = maps.shape[1], maps.shape[2]
        grid   = np.zeros((n_rows * h, n_cols * w), dtype=np.uint8)
        for i, fm in enumerate(maps.numpy()):
            r, c  = divmod(i, n_cols)
            lo, hi = fm.min(), fm.max()
            norm  = ((fm - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = norm
        panels[f"feature_maps/{layer_name}"] = wandb.Image(
            grid, caption=f"{layer_name} — first {n_filters} filters")
    wandb.log(panels)
    print("  [WandB 2.4] Logged feature maps.")


def log_detection_table(model: VGG11Localizer, val_loader, device,
                         n_samples: int = 10) -> None:
    model.eval()
    table = wandb.Table(columns=[
        "image", "iou",
        "pred_xc", "pred_yc", "pred_w", "pred_h",
        "gt_xc",   "gt_yc",   "gt_w",   "gt_h",
    ])
    collected = 0
    with torch.no_grad():
        for batch in val_loader:
            if collected >= n_samples:
                break
            imgs   = batch["image"].to(device)
            bboxes = batch["bbox"]
            preds  = model(imgs).cpu()
            for i in range(min(imgs.size(0), n_samples - collected)):
                img_np = _denorm(batch["image"][i])
                pred_b = preds[i].numpy()
                gt_b   = bboxes[i].numpy()
                iou    = _compute_iou_single(pred_b, gt_b)
                vis = img_np.copy()
                for box, color in [(gt_b, (0,200,50)), (pred_b, (220,50,50))]:
                    xc,yc,bw,bh = box
                    x1,y1 = int(xc-bw/2), int(yc-bh/2)
                    x2,y2 = int(xc+bw/2), int(yc+bh/2)
                    cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                cv2.putText(vis, f"IoU={iou:.2f}", (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255,255,255), 1, cv2.LINE_AA)
                table.add_data(
                    wandb.Image(vis, caption=f"green=GT  red=Pred  IoU={iou:.3f}"),
                    round(iou, 4),
                    *[round(v, 2) for v in pred_b.tolist()],
                    *[round(v, 2) for v in gt_b.tolist()],
                )
                collected += 1
    wandb.log({"detection/bbox_table": table})
    print(f"  [WandB 2.5] Logged detection table ({collected} samples).")


_TRIMAP_PALETTE = np.array([
    [  0, 200,  50],
    [ 50,  50, 200],
    [220, 200,   0],
], dtype=np.uint8)


def log_seg_samples(model: VGG11UNet, val_loader, device,
                     n_samples: int = 5) -> None:
    model.eval()
    panels    = {}
    collected = 0
    with torch.no_grad():
        for batch in val_loader:
            if collected >= n_samples:
                break
            imgs  = batch["image"].to(device)
            masks = batch["mask"]
            preds = model(imgs).argmax(dim=1).cpu()
            for i in range(min(imgs.size(0), n_samples - collected)):
                orig   = _denorm(batch["image"][i])
                gt_col = _TRIMAP_PALETTE[masks[i].numpy().clip(0, 2)]
                pr_col = _TRIMAP_PALETTE[preds[i].numpy().clip(0, 2)]
                k = f"segmentation/sample_{collected:02d}"
                panels[f"{k}_original"]    = wandb.Image(orig,   caption="original")
                panels[f"{k}_gt_trimap"]   = wandb.Image(gt_col, caption="GT trimap")
                panels[f"{k}_pred_trimap"] = wandb.Image(pr_col, caption="predicted trimap")
                collected += 1
    wandb.log(panels)
    print(f"  [WandB 2.6] Logged {collected} segmentation samples.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Task 1 — Classification
# ═══════════════════════════════════════════════════════════════════════════════

def train_classifier(args, run_name: str = "classifier") -> None:
    device = _get_device()
    print(f"\n{'='*60}\n  Task 1: {run_name}  |  device={device}\n{'='*60}")

    train_loader, val_loader, _ = _make_loaders(args)

    model = VGG11(num_classes=37, dropout_p=args.dropout_p).to(device)

    _load_pretrained_vgg11bn(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    wandb.init(
        project=args.wandb_project, name=run_name, reinit=True,
        config=dict(task="classification", lr=args.lr, epochs=args.epochs,
                    batch_size=args.batch_size, dropout_p=args.dropout_p,
                    weight_decay=args.weight_decay),
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_f1 = 0.0

    for epoch in range(args.epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc   = (np.array(all_preds) == np.array(all_labels)).mean()
        train_f1    = sk_f1(all_labels, all_preds, average="macro", zero_division=0)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        v_preds, v_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item() * imgs.size(0)
                v_preds.extend(logits.argmax(1).cpu().numpy())
                v_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc   = (np.array(v_preds) == np.array(v_labels)).mean()
        val_f1    = sk_f1(v_labels, v_preds, average="macro", zero_division=0)
        scheduler.step()

        print(f"[{run_name}] Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_f1={train_f1:.4f}  "
              f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")

        wandb.log({
            "epoch":          epoch + 1,
            "cls/train_loss": train_loss,
            "cls/train_acc":  float(train_acc),
            "cls/train_f1":   train_f1,
            "cls/val_loss":   val_loss,
            "cls/val_acc":    float(val_acc),
            "cls/val_f1":     val_f1,
            "cls/lr":         optimizer.param_groups[0]["lr"],
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt = os.path.join(args.checkpoint_dir, "classifier.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  → saved {ckpt}  (val_f1={val_f1:.4f})")

    log_activation_distributions(model, val_loader, device, run_name=run_name)
    if run_name == "classifier":
        log_feature_maps(model, val_loader, device)

    wandb.finish()
    print(f"[{run_name}] Best val macro-F1: {best_val_f1:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Task 2 — Localization
# ═══════════════════════════════════════════════════════════════════════════════

def train_localizer(args) -> None:
    device = _get_device()
    print(f"\n{'='*60}\n  Task 2: Localization  |  device={device}\n{'='*60}")

    train_loader, val_loader, _ = _make_loaders(args)

    model = VGG11Localizer(in_channels=3, dropout_p=args.dropout_p).to(device)

    # Load the trained classifier encoder into the localizer encoder
    _transfer_encoder(
        model,
        os.path.join(args.checkpoint_dir, "classifier.pth"),
        src_prefix="features.",
        dst_prefix="encoder.",
    )

    for p in model.encoder.parameters():
        p.requires_grad = False
    print("  Encoder FROZEN — only regressor head will be trained.")

    # Only optimise the regressor parameters
    optimizer = torch.optim.Adam(
        model.regressor.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    # ─────────────────────────────────────────────────────────────────────────

    smooth_l1  = nn.SmoothL1Loss()
    iou_loss   = IoULoss(reduction="mean")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    wandb.init(
        project=args.wandb_project, name="localizer", reinit=True,
        config=dict(task="localization", lr=args.lr, epochs=args.epochs,
                    batch_size=args.batch_size, freeze_encoder=True),
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        # Keep encoder in eval mode so BN/Dropout behave correctly
        model.encoder.eval()
        t_mse = t_iou = t_tot = n = 0.0

        for batch in train_loader:
            imgs   = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)   # pixel space [0, 224]

            optimizer.zero_grad()
            pred = model(imgs)                  # pixel space [0, 224]

            # Normalise to [0,1] for the regression loss so both terms match scale
            pred_norm  = pred   / 224.0
            bbox_norm  = bboxes / 224.0
            m_loss = smooth_l1(pred_norm, bbox_norm)   # ~[0, 1] range
            i_loss = iou_loss(pred, bboxes)            # already [0, 1]

            # Now both terms are comparable — equal weighting works well
            loss = m_loss + 2.0 * i_loss
            loss.backward()
            optimizer.step()

            bs     = imgs.size(0)
            t_mse += m_loss.item() * bs
            t_iou += i_loss.item() * bs
            t_tot += loss.item()   * bs
            n     += bs

        t_mse /= n; t_iou /= n; t_tot /= n

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_mse = v_iou = v_tot = vn = 0.0

        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                pred   = model(imgs)
                pred_norm = pred   / 224.0
                bbox_norm = bboxes / 224.0
                m_loss = smooth_l1(pred_norm, bbox_norm)
                i_loss = iou_loss(pred, bboxes)
                bs     = imgs.size(0)
                v_mse += m_loss.item() * bs
                v_iou += i_loss.item() * bs
                v_tot += (m_loss + 2.0 * i_loss).item() * bs
                vn    += bs

        v_mse /= vn; v_iou /= vn; v_tot /= vn
        scheduler.step()

        print(f"[Loc] Epoch {epoch+1:3d}/{args.epochs}  "
              f"train={t_tot:.4f} (mse={t_mse:.4f} iou={t_iou:.4f})  "
              f"val={v_tot:.4f} (mse={v_mse:.4f} iou={v_iou:.4f})")

        wandb.log({
            "epoch":                epoch + 1,
            "loc/train_mse":        t_mse,
            "loc/train_iou_loss":   t_iou,
            "loc/train_total_loss": t_tot,
            "loc/val_mse":          v_mse,
            "loc/val_iou_loss":     v_iou,
            "loc/val_total_loss":   v_tot,
            "loc/lr":               optimizer.param_groups[0]["lr"],
        })

        if v_tot < best_val_loss:
            best_val_loss = v_tot
            ckpt = os.path.join(args.checkpoint_dir, "localizer.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  → saved {ckpt}  (val_loss={v_tot:.4f})")

    log_detection_table(model, val_loader, device, n_samples=10)
    wandb.finish()
    print(f"[Loc] Best val loss: {best_val_loss:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Task 3 — Segmentation
# ═══════════════════════════════════════════════════════════════════════════════

def train_segmentation(args) -> None:
    device = _get_device()
    print(f"\n{'='*60}\n  Task 3: Segmentation  |  device={device}\n{'='*60}")

    train_loader, val_loader, _ = _make_loaders(args)

    model = VGG11UNet(num_classes=3, in_channels=3).to(device)

    _transfer_encoder(
        model,
        os.path.join(args.checkpoint_dir, "classifier.pth"),
        src_prefix="features.",
        dst_prefix="encoder.",
    )

    # ── FIX #2 (same principle): freeze encoder so decoder learns from the
    # same fixed feature distribution that multitask.py will provide.
    for p in model.encoder.parameters():
        p.requires_grad = False
    print("  Encoder FROZEN — only decoder will be trained.")

    # Weighted CE: boundary class is hardest → give it the highest weight
    class_weights = torch.tensor([2.0, 1.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Only optimise decoder parameters
    decoder_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        decoder_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    wandb.init(
        project=args.wandb_project, name="segmentation", reinit=True,
        config=dict(task="segmentation", lr=args.lr, epochs=args.epochs,
                    batch_size=args.batch_size, freeze_encoder=True),
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_dice = 0.0

    for epoch in range(args.epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        model.encoder.eval()   # Keep encoder in eval mode (BN uses running stats)
        t_loss = n = 0.0

        for batch in train_loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits  = model(imgs)
            ce_loss = criterion(logits, masks)
            d_loss  = soft_dice_loss(logits, masks)
            loss    = ce_loss + d_loss
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            n      += imgs.size(0)

        t_loss /= n

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_loss = v_dice = v_pix = vn = 0.0

        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                bs     = imgs.size(0)
                v_loss += criterion(logits, masks).item() * bs
                v_dice += dice_score(logits, masks)       * bs
                v_pix  += pixel_accuracy(logits, masks)   * bs
                vn     += bs

        v_loss /= vn; v_dice /= vn; v_pix /= vn
        scheduler.step()

        print(f"[Seg] Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={t_loss:.4f}  "
              f"val_loss={v_loss:.4f}  val_dice={v_dice:.4f}  val_pix={v_pix:.4f}")

        wandb.log({
            "epoch":             epoch + 1,
            "seg/train_loss":    t_loss,
            "seg/val_loss":      v_loss,
            "seg/val_dice":      v_dice,
            "seg/val_pixel_acc": v_pix,
            "seg/lr":            optimizer.param_groups[0]["lr"],
        })

        if v_dice > best_val_dice:
            best_val_dice = v_dice
            ckpt = os.path.join(args.checkpoint_dir, "unet.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  → saved {ckpt}  (val_dice={v_dice:.4f})")

    log_seg_samples(model, val_loader, device, n_samples=5)
    wandb.finish()
    print(f"[Seg] Best val Dice: {best_val_dice:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Dropout sweep (report section 2.2)
# ═══════════════════════════════════════════════════════════════════════════════

def run_dropout_sweep(args) -> None:
    for p, name in [
        (0.0, "dropout_p0.0"),
        (0.2, "dropout_p0.2"),
        (0.5, "dropout_p0.5"),
    ]:
        args.dropout_p = p
        train_classifier(args, run_name=name)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 — Training")
    parser.add_argument(
        "--task",
        choices=["classify", "localize", "segment", "all", "dropout_sweep"],
        default="all",
    )
    parser.add_argument("--data_dir",       type=str,   default="data")
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--dropout_p",      type=float, default=0.5)
    parser.add_argument("--num_workers",    type=int,   default=2)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--wandb_project",  type=str,   default="da6401-assignment2")

    args = parser.parse_args()

    if args.task == "dropout_sweep":
        run_dropout_sweep(args)
    else:
        if args.task in ("classify", "all"):
            train_classifier(args)
        if args.task in ("localize", "all"):
            train_localizer(args)
        if args.task in ("segment", "all"):
            train_segmentation(args)


if __name__ == "__main__":
    main()