# models/modeling/trajectory_model.py
import torch
import torch.nn as nn
from typing import Dict, Any, List
from hydra.utils import instantiate

class TrajectoryModel(nn.Module):
    def __init__(self, encoder, decoder, T_fixed: int = None, loss: dict = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.T_fixed = T_fixed
        # loss 通过 Hydra 传 dict（含 _target_），可热插拔
        self.loss_fn = instantiate(loss) if loss is not None else None

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        seqs: List[List] = batch["seqs"]
        feats, frame_mask, coords_uv = self.encoder(seqs, T_fixed=self.T_fixed)  # [B,T,N,D], [B,T], [N,2]

        traj_pred, score = self.decoder(
            image_feats=feats, coords_uv=coords_uv, frame_mask=frame_mask
        )  # [B,K,H,2], [B,K] or None

        pred = {"traj_pred": traj_pred, "score": score}

        if self.loss_fn is None or "gt_traj" not in batch:
            return pred

        losses = self.loss_fn(pred, batch)   # 必须返回 {"loss": ... , ...}
        return {**pred, **losses}
