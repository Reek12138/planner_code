# losses/nll_gaussian.py
import torch
from typing import Dict, Any
from .base import LossModule

class GaussianNLL(LossModule):
    def __init__(self, eps: float = 1e-6, min_of_k: bool = True):
        super().__init__()
        self.eps = eps
        self.min_of_k = min_of_k

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        期望 pred 还包含：
          - traj_mu:  [B,K,H,2]
          - traj_logvar: [B,K,H,2]  逐维对数方差
        """
        mu = pred["traj_mu"]
        logvar = pred["traj_logvar"]
        gt = batch["gt_traj"].to(mu.device).unsqueeze(1)  # [B,1,H,2]

        inv_var = torch.exp(-logvar)
        nll = 0.5 * ( (gt - mu)**2 * inv_var + logvar )   # [B,K,H,2]
        nll = nll.sum(dim=-1).mean(dim=-1)                # [B,K] -> sum over 2, mean over H

        loss = nll.min(dim=1).values.mean() if self.min_of_k else nll.mean()
        return {"loss": loss, "nll": nll.mean()}
