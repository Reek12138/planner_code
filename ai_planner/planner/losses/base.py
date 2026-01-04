# losses/base.py
import torch
import torch.nn as nn
from typing import Dict, Any

class LossModule(nn.Module):
    """
    统一约定：
      forward(pred: Dict, batch: Dict) -> Dict[str, torch.Tensor]
    返回字段必须包含 "loss"（标量），其他键是可选指标（会被 Trainer 记录到日志）。
    """
    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
