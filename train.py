"""Training entrypoint — DA6401 Assignment 2.

Usage (run from project root):
    python train.py --task all --data_dir data --epochs 30 --freeze_encoder
    python train.py --task classify  --epochs 30
    python train.py --task localize  --epochs 30 --freeze_encoder
    python train.py --task segment   --epochs 30 --freeze_encoder
    python train.py --task dropout_sweep --epochs 15   # for report section 2.2

WandB report sections covered automatically:
    2.1  Activation distributions (conv3) — logged after classifier training
    2.2  Three dropout curves via dropout_sweep task
    2.4  Feature maps from first & last conv layer
    2.5  Detection table — 10 val images, GT+pred boxes, IoU score per box
    2.6  Segmentation samples — 5 images: original / GT trimap / pred trimap
"""

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





def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Dice Loss to directly optimize the Gradescope metric."""
    probs = torch.softmax(logits, dim=1)
    loss = 0.0
    for c in range(num_classes):
        p = probs[:, c, :, :]
        t = (targets == c).float()
        intersection = (p * t).sum(dim=(1, 2))
        cardinality = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice = (2.0 * intersection + eps) / (cardinality + eps)
        loss += (1.0 - dice).mean()
    return loss / num_classes

# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loaders(args):
    """80/20 train/val split, seed=42."""
    dataset   = OxfordIIITPetDataset(root_dir=args.data_dir)
    val_len   = int(len(dataset) * 0.2)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )
    kw = dict(num_workers=args.num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **kw),
        dataset,
    )


def _transfer_encoder(model, checkpoint_path: str,
                       src_prefix: str, dst_prefix: str) -> None:
    """Copy encoder weights from one checkpoint into a different model."""
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
    """
    Report 2.1 — log activation histograms of the 3rd conv layer (block2)
    and last conv layer (block5) for one validation batch.
    """
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
    """
    Report 2.4 — visualise the first n_filters feature maps from the first
    conv block and last conv block for a single validation image.
    """
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
        maps   = fmaps[0, :n_filters]   # [n_filters, H, W]
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
    """
    Report 2.5 — interactive wandb.Table with n_samples val images.
    Each row shows: annotated image (GT=green, Pred=red), IoU, and all coords.
    """
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
    [  0, 200,  50],   # 0 → green  (foreground)
    [ 50,  50, 200],   # 1 → blue   (background)
    [220, 200,   0],   # 2 → yellow (boundary)
], dtype=np.uint8)


def log_seg_samples(model: VGG11UNet, val_loader, device,
                     n_samples: int = 5) -> None:
    """
    Report 2.6 — log n_samples val images showing:
    original image / GT trimap / predicted trimap.
    """
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
    """
    run_name is exposed so the dropout sweep can call this three times with
    different names and dropout probabilities (report section 2.2).
    """
    device = _get_device()
    print(f"\n{'='*60}\n  Task 1: {run_name}  |  device={device}\n{'='*60}")

    train_loader, val_loader, _ = _make_loaders(args)

    model     = VGG11(num_classes=37, dropout_p=args.dropout_p).to(device)
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

        # wandb.log → interactive line charts (not screenshots)
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

    # ── Post-training WandB extras ─────────────────────────────────────────────
    log_activation_distributions(model, val_loader, device, run_name=run_name)
    if run_name == "classifier":          # skip for dropout sweep sub-runs
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

    # Initialise encoder with trained classifier weights
    _transfer_encoder(
        model,
        os.path.join(args.checkpoint_dir, "classifier.pth"),
        src_prefix="features.",
        dst_prefix="encoder.",
    )

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("  Encoder FROZEN for first half of training.")

    mse_loss  = nn.SmoothL1Loss() # Much better for bounding box regression    
    iou_loss  = IoULoss(reduction="mean")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    wandb.init(
        project=args.wandb_project, name="localizer", reinit=True,
        config=dict(task="localization", lr=args.lr, epochs=args.epochs,
                    batch_size=args.batch_size, freeze_encoder=args.freeze_encoder),
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    unfreeze_done = False

    for epoch in range(args.epochs):
        if args.freeze_encoder and not unfreeze_done and epoch >= args.epochs // 2:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr * 0.1,
                weight_decay=args.weight_decay,
            )
            unfreeze_done = True
            print(f"  Epoch {epoch+1}: Encoder UNFROZEN (lr={args.lr*0.1:.2e})")

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_mse = t_iou = t_tot = n = 0.0

        for batch in train_loader:
            imgs   = batch["image"].to(device)
            # IMPORTANT: dataset returns bboxes in pixel space [0,224] — same
            # scale as model output (sigmoid * 224), so IoULoss is consistent.
            bboxes = batch["bbox"].to(device)

            optimizer.zero_grad()
            pred   = model(imgs)
            m_loss = mse_loss(pred, bboxes)
            i_loss = iou_loss(pred, bboxes)
            # Give IoU loss a higher weight so the model cares about perfect overlaps
            loss   = m_loss + (2.0 * i_loss)
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
                m_loss = mse_loss(pred, bboxes)
                i_loss = iou_loss(pred, bboxes)
                bs     = imgs.size(0)
                v_mse += m_loss.item() * bs
                v_iou += i_loss.item() * bs
                v_tot += (m_loss + i_loss).item() * bs
                vn    += bs

        v_mse /= vn; v_iou /= vn; v_tot /= vn
        scheduler.step()

        print(f"[Loc] Epoch {epoch+1:3d}/{args.epochs}  "
              f"train={t_tot:.4f} (mse={t_mse:.2f} iou={t_iou:.4f})  "
              f"val={v_tot:.4f} (mse={v_mse:.2f} iou={v_iou:.4f})")

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

    # ── 2.5: interactive detection table ──────────────────────────────────────
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

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("  Encoder FROZEN for first half of training.")

    # Class 0 (Pet): 2.0x weight, Class 1 (Bg): 1.0x weight, Class 2 (Boundary): 5.0x weight
    class_weights = torch.tensor([2.0, 1.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    wandb.init(
        project=args.wandb_project, name="segmentation", reinit=True,
        config=dict(task="segmentation", lr=args.lr, epochs=args.epochs,
                    batch_size=args.batch_size, freeze_encoder=args.freeze_encoder),
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_dice = 0.0
    unfreeze_done = False

    for epoch in range(args.epochs):
        if args.freeze_encoder and not unfreeze_done and epoch >= args.epochs // 2:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr * 0.1,
                weight_decay=args.weight_decay,
            )
            unfreeze_done = True
            print(f"  Epoch {epoch+1}: Encoder UNFROZEN (lr={args.lr*0.1:.2e})")

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss = n = 0.0

        for batch in train_loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            ce_loss = criterion(logits, masks)
            d_loss  = soft_dice_loss(logits, masks)
            loss    = ce_loss + d_loss # Combine them!
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

    # ── 2.6: segmentation visual samples ──────────────────────────────────────
    log_seg_samples(model, val_loader, device, n_samples=5)

    wandb.finish()
    print(f"[Seg] Best val Dice: {best_val_dice:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Dropout sweep helper  (report section 2.2)
# ═══════════════════════════════════════════════════════════════════════════════

def run_dropout_sweep(args) -> None:
    """
    Train the classifier 3× with different dropout rates so all three curves
    appear overlaid in WandB for section 2.2.
    """
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