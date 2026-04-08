import torch
import torch.nn as nn

class IoULoss(nn.Module):

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps

        if reduction not in ["none" , "mean" , "sum"]:
          raise ValueError(f"Invalid reduction type:{reduction}. Must be 'none','mean', or 'sum'.")
        self.reduction = reduction
        # TODO: validate reduction in {"none", "mean", "sum"}.

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        # TODO: implement IoU loss.

        # 1. Split the coordinates for easy calculation
        p_cx, p_cy, p_w, p_h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        t_cx, t_cy, t_w, t_h = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
        
        # 2. Convert (cw,cy,w,h) to (x_min,y_min,x_max,y_max)
        p_xmin = p_cx - (p_w / 2)
        p_ymin = p_cy - (p_h / 2)
        p_xmax = p_cx + (p_w / 2)
        p_ymax = p_cy + (p_h / 2)

        t_xmin = t_cx - (t_w / 2)
        t_ymin = t_cy - (t_h / 2)
        t_xmax = t_cx + (t_w / 2)
        t_ymax = t_cy + (t_h / 2)

        # 3. Calculate the coordinates of the Intersection box
        inter_xmin = torch.max(p_xmin, t_xmin)
        inter_ymin = torch.max(p_ymin, t_ymin)
        inter_xmax = torch.min(p_xmax, t_xmax)
        inter_ymax = torch.min(p_ymax, t_ymax)

        # 4. Calculate Intersection Area
        # CRITICAL: .clamp(min=0) prevents negative areas when boxes don't overlap at all
        inter_w = (inter_xmax - inter_xmin).clamp(min=0)
        inter_h = (inter_ymax - inter_ymin).clamp(min=0)
        inter_area = inter_w * inter_h

        # 5. Calculate Union Area
        pred_area = p_w * p_h
        target_area = t_w * t_h
        union_area = pred_area + target_area - inter_area

        # 6. Calculate IoU and the Loss
        # We add self.eps to the denominator to prevent division by zero
        iou = inter_area / (union_area + self.eps)
        
        # Loss is bounded [0, 1]. A perfect IoU of 1 gives a Loss of 0.
        loss = 1.0 - iou 

        # 7. Apply the requested reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else: # "none"
            return loss



