import torch
import torch.nn as nn
import torch.nn.functional as F

from eosar.config import Config


class FocalDiceLoss(nn.Module):
    """Combined focal loss and dice loss for imbalanced binary masks with validity masking."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.gamma = cfg.focal_gamma
        self.focal_w = cfg.focal_w
        self.dice_w = cfg.dice_w
        self.register_buffer("pos_weight", torch.tensor([cfg.pos_weight]))

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Focal loss with optional validity masking."""
        bce = F.binary_cross_entropy_with_logits(
            pred,
            target,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        prob = torch.sigmoid(pred)
        pt = torch.where(target == 1, prob, 1 - prob)
        loss = bce * (1 - pt) ** self.gamma
        
        if valid is not None:
            loss = loss * valid
            return loss.sum() / valid.sum().clamp_min(1.0)
        
        return loss.mean()

    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
        smooth: float = 1.0,
    ) -> torch.Tensor:
        """Dice loss with optional validity masking."""
        prob = torch.sigmoid(pred)
        
        if valid is not None:
            prob = prob * valid
            target = target * valid
        
        prob = prob.reshape(-1)
        target = target.reshape(-1)
        inter = (prob * target).sum()
        return 1 - (2 * inter + smooth) / (prob.sum() + target.sum() + smooth)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional validity mask.
        
        Args:
            pred: Logits tensor (B, 1, H, W)
            target: Binary target tensor (B, 1, H, W)
            valid: Optional validity mask (B, 1, H, W) where 1.0 = valid, 0.0 = invalid
        
        Returns:
            Combined focal + dice loss
        """
        fl = self.focal_loss(pred, target, valid)
        dl = self.dice_loss(pred, target, valid)
        return self.focal_w * fl + self.dice_w * dl
