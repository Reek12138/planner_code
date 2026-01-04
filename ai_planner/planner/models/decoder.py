from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SWiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.silu(x) * F.sigmoid(x)
    
class Score_head(nn.Module):
    def __init__(self, input_dim: int, n_points: int, hidden_dim: int):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Linear(input_dim*n_points, hidden_dim),
            SWiGLU(),
            nn.Linear(hidden_dim, 1),
            SWiGLU()
        )
    def forward(self, x):
        return self.score_head(x)
    

class Anchor_Free_Decoder(nn.Module):
    def __init__(self, input_dim: int, 
                 n_traj: int = 5, n_points: int = 6, 
                 hidden_dim: int = 512, use_residual: bool = True):
        super(Anchor_Free_Decoder).__init__()
        self.n_traj = n_traj
        self.n_points = n_points
        self.query_embedding = nn.Embedding(num_embeddings=n_points, embedding_dim=n_traj * input_dim)  #[n_traj, n_points * input_dim]
        self.temporal_attn = nn.MultiheadAttention(embed_dim=n_traj * input_dim, num_heads=n_traj, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(n_points*input_dim, n_points*hidden_dim),
            SWiGLU(),
            nn.Linear(n_points*hidden_dim, n_points*input_dim),
            SWiGLU()
        )
        self.score_head = Score_head(input_dim=input_dim, n_points=n_points, hidden_dim=hidden_dim)

        self.use_residual = use_residual
        self.traj_output_head = nn.Linear(input_dim, 2)
    
    def forward(self, feature):
        B, T, D = feature.shape

        query = self.query_embedding.weight.unsqueeze(0).repeat(B, 1, 1) #[B, n_points, n_traj * input_dim]

        feature_ = feature.unsqueeze(2).repeat(1, 1, self.n_traj, 1) #[B, T, n_traj, D]
        feature_ = feature_.view(B, T, self.n_traj*D) #[B, T, n_traj*D]

        attn_output, _ = self.temporal_attn(query, feature_, feature_) #[B, n_points, n_traj*D]

        attn_output = attn_output.contiguous().view(B, self.n_points, self.n_traj, D)  # After view

        traj_features = attn_output.transpose(1, 2).contiguous().view(B, self.n_traj, self.n_points*D)  #[B, n_traj, n_points*input_dim]
        ffn_output = self.ffn(traj_features).view(B, self.n_traj, self.n_points, D)

        if self.use_residual:
            ffn_output = ffn_output + attn_output.transpose(1, 2)

        traj_points = self.traj_output_head(ffn_output) #[B, n_traj, n_points, 2]

        traj_score = self.score_head(ffn_output.view(B, self.n_traj, self.n_points*D))
        traj_probs = torch.softmax(traj_score, dim=-1) #[B, n_traj, score_prob]

        return traj_points, ffn_output, traj_probs
    

class Anchor_Based_Decoder(nn.Module):
    def __init__(self, input_dim: int,
                 n_traj: int = 5, n_points: int = 6,
                 hidden_dim: int = 512, use_residual: bool = True):
        super(Anchor_Based_Decoder).__init__()

        self.temporal_attn = nn.MultiheadAttention(embed_dim=n_traj * input_dim, num_heads=n_traj, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(n_points*input_dim, n_points*hidden_dim),
            SWiGLU(),
            nn.Linear(n_points*hidden_dim, n_points*input_dim),
            SWiGLU()
        )

        self.use_residual = use_residual
        self.traj_output_head = nn.Linear(input_dim, 2)
        self.score_head = Score_head(input_dim=input_dim, n_points=n_points, hidden_dim=hidden_dim)
    
        self.n_traj = n_traj
        self.n_points = n_points
    
    def forward(self, traj_feature, his_feature):
        B, n_traj, n_points, input_dim = traj_feature.shape
        _, T, D = his_feature.shape
        
        assert self.n_traj == n_traj and self.n_points == n_points and \
            input_dim == D, "第二段解码维度不匹配"
        
        his_feature_ = his_feature.unsqueeze(2).repeat(1, 1, self.n_traj, 1) #[B, T, n_traj, D]
        his_feature_ = his_feature_.view(B, T, self.n_traj*D) #[B, T, n_traj*D]

        traj_feature_ = traj_feature.transpose(1, 2).contiguous().view(B, n_points, n_traj*input_dim) #[B, n_points, n_traj*input_dim]

        attn_output, _ = self.temporal_attn(traj_feature_, his_feature_, his_feature_) #[B, n_points, n_traj*D]

        attn_output = attn_output.contiguous().view(B, self.n_points, self.n_traj, D)  # After view

        traj_offset_feature = attn_output.transpose(1, 2).contiguous().view(B, self.n_traj, self.n_points*D)  #[B, n_traj, n_points*input_dim]
        ffn_output = self.ffn(traj_offset_feature).view(B, self.n_traj, self.n_points, D)

        if self.use_residual:
            ffn_output = ffn_output + attn_output.transpose(1, 2)

        traj_offset = self.traj_output_head(ffn_output) #[B, n_traj, n_points, 2]
        traj_score = self.score_head(ffn_output.view(B, self.n_traj, self.n_points*D))
        traj_probs = torch.softmax(traj_score, dim=-1) #[B, n_traj, score_prob]

        return traj_offset, traj_probs



class TRAJ_Decoder(nn.Module):
    def __init__(self, input_dim: int, num_traj: int):
        super(TRAJ_Decoder, self).__init__()

        self.anchor_free_decoder = Anchor_Free_Decoder(input_dim=input_dim, n_traj=num_traj)
        self.anchor_based_decoder = Anchor_Based_Decoder(input_dim=input_dim, n_traj=num_traj)

        # 需要把 score 和 traj 做 winner takes all 的设计

        pass

    def forward(self, x):
        B, T, D = x.shape

        init_traj, traj_feature, _ = self.anchor_free_decoder(x)

        traj_offset, traj_probs = self.anchor_based_decoder(traj_feature, x)

        final_traj = init_traj + traj_offset

        _, best_traj_idx = torch.max(traj_probs, dim=-1)

        best_traj = torch.gather(final_traj, 1, best_traj_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.n_points, 2))

        return best_traj