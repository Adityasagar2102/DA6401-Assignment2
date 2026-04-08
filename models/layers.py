import torch
import torch.nn as nn


class CustomDropout(nn.Module):

    def __init__(self, p: float = 0.5):
        super(CustomDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability must be between 0 and 1")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # At eval time or p=0: pass through unchanged
        if not self.training or self.p == 0:
            return x

        # p=1 means drop everything → return zeros (avoid division by zero)
        if self.p >= 1.0:
            return torch.zeros_like(x)

        # Inverted dropout: scale up by 1/(1-p) so the expected value is unchanged
        mask = (torch.rand(x.shape, device=x.device) > self.p).float()
        return (x * mask) / (1.0 - self.p)