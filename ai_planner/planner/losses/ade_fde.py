# losses/ade_fde.py
import torch
from typing import Dict, Any
from .base import LossModule

class ADEFDELoss(LossModule):
    def __init__(self, w_ade: float = 1.0, w_fde: float = 1.0):
        super().__init__()
        self.w_ade = w_ade
        self.w_fde = w_fde

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        traj_pred = pred["traj_pred"]      # [B,K,H,2]
        gt = batch["gt_traj"].to(traj_pred.device)  # [B,H,2]
        B, K, H, _ = traj_pred.shape

        diff = traj_pred - gt.unsqueeze(1).expand(B, K, H, 2)
        dist = torch.sqrt((diff ** 2).sum(dim=-1))            # [B,K,H]
        ade_k = dist.mean(dim=-1)                             # [B,K]
        fde_k = dist[..., -1]                                 # [B,K] 末点距离

        # min-of-K
        ade = ade_k.min(dim=1).values
        fde = fde_k.min(dim=1).values

        loss = (self.w_ade * ade + self.w_fde * fde).mean()
        return {"loss": loss, "ade": ade.mean(), "fde": fde.mean()}
