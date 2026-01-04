# losses/ade.py
import torch
from typing import Dict, Any
from .base import LossModule

class ADELoss(LossModule):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ["mean", "sum"]
        self.reduction = reduction

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        期望 pred:
          - traj_pred: [B, K, H, 2]  (K可为1)
        期望 batch:
          - gt_traj  : [B, H, 2]
        """
        traj_pred = pred["traj_pred"]            # [B,K,H,2]
        gt = batch["gt_traj"].to(traj_pred.device)  # [B,H,2]
        B, K, H, _ = traj_pred.shape

        # L2 逐点距离
        diff = traj_pred - gt.unsqueeze(1).expand(B, K, H, 2)      # [B,K,H,2]
        dist = torch.sqrt((diff ** 2).sum(dim=-1))                 # [B,K,H]
        ade_k = dist.mean(dim=-1)                                  # [B,K]

        # 如果 K>1，取 minADE；否则就是 ADE
        if K > 1:
            ade = ade_k.min(dim=1).values                          # [B]
        else:
            ade = ade_k.squeeze(1)                                 # [B]

        loss = ade.mean() if self.reduction == "mean" else ade.sum()
        return {"loss": loss, "ade": ade.mean()}
