from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np

import torch
import torch.nn as nn
from .encoder import Feature_Encoder
from .decoder import TRAJ_Decoder

# ============================================================
# 1) PolicyNet —— 占位：请自行实现
#    约定:
#      - 若 policy_output="controls": forward(obs) -> [B, T, 2] 或 [B, 2] in [-1,1]
#      - 若 policy_output="traj_rel":  forward(obs) -> [B, T, 2/3]（自车系增量）
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int = 2):
        super().__init__()
        # TODO: 自行实现
        self.encoder = Feature_Encoder()
        self.decoder = TRAJ_Decoder()
        
        pass

    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.tanh(self.mlp(x))

        feature = self.encoder(x) #B T D

        traj = self.decoder(feature)

        # TODO:后处理

        return traj

        # raise NotImplementedError("Please implement PolicyNet.forward()")