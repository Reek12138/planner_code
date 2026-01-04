# losses/bok.py
import torch
import torch.nn.functional as F
from typing import Dict, Any
from .base import LossModule

class BestOfKHuber(LossModule):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        traj_pred = pred["traj_pred"]                          # [B,K,H,2]
        gt = batch["gt_traj"].to(traj_pred.device).unsqueeze(1)  # [B,1,H,2]
        B, K, H, _ = traj_pred.shape

        # Huber per step -> 平均成序列损失
        hub = F.huber_loss(traj_pred, gt.expand(B, K, H, 2), delta=self.delta, reduction="none")  # [B,K,H,2]
        hub = hub.mean(dim=(-1, -2))                         # [B,K] 先 (H,2) 聚合

        loss = hub.min(dim=1).values.mean()                  # Best-of-K
        return {"loss": loss, "huber_minK": loss}
