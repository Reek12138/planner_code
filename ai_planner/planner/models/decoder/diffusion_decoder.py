"""
Diffusion Decoder for Velocity Prediction

Architecture: Conditional Diffusion Model
- Input: noisy velocity v_t [B, 3], timestep t [B]
- Condition: state_emb [B, d_model], img_feat [B, P, d_model]
- Output: predicted noise epsilon [B, 3]

Uses cross-attention to fuse image features and self-attention for velocity refinement.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步正弦位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class CrossAttention(nn.Module):
    """简化版Cross-Attention"""
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        return self.to_out(out)


class GEGLU(nn.Module):
    """GLU with GELU activation"""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward with GEGLU"""
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        dim_out = dim_out or dim
        hidden_dim = int(dim * mult)
        
        self.net = nn.Sequential(
            GEGLU(dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaGN(nn.Module):
    """Adaptive GroupNorm - 通过时间编码调制特征"""
    def __init__(self, dim: int, time_emb_dim: int, num_groups: int = 8):
        super().__init__()
        self.num_groups = num_groups
        self.norm = nn.GroupNorm(num_groups, dim)
        self.scale_shift = nn.Linear(time_emb_dim, dim * 2)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, C] -> [B, C, 1] for groupnorm
        B = x.shape[0]
        x = x.unsqueeze(-1)  # [B, C, 1]
        x = self.norm(x)
        x = x.squeeze(-1)  # [B, C]
        
        scale, shift = self.scale_shift(time_emb).chunk(2, dim=-1)
        return x * (1 + scale) + shift


class DiffusionDecoderBlock(nn.Module):
    """
    单个Diffusion Decoder Block:
    1. Self-Attention on velocity tokens
    2. Cross-Attention to image features
    3. Cross-Attention to state embedding
    4. Feed-forward with time conditioning
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        time_emb_dim: int = 512,
    ):
        super().__init__()
        
        # Self-attention for velocity refinement
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention to image features
        self.cross_attn_img = CrossAttention(
            query_dim=d_model,
            context_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Cross-attention to state (used only if state provided separately)
        self.cross_attn_state = CrossAttention(
            query_dim=d_model,
            context_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        # Feed-forward with time embedding
        self.ada_norm = AdaGN(d_model, time_emb_dim)
        self.ff = FeedForward(d_model, dim_out=d_model, mult=dim_ff/d_model, dropout=dropout)
        self.norm4 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        v_tokens: torch.Tensor,      # [B, Nv, d_model] velocity tokens
        img_feat: torch.Tensor,      # [B, P, d_model] image features
        state_emb: torch.Tensor,     # [B, d_model] or [B, 1, d_model] state embedding
        time_emb: torch.Tensor,      # [B, time_emb_dim] time embedding
    ) -> torch.Tensor:
        B = v_tokens.shape[0]
        
        # 1. Self-attention
        v2 = self.norm1(v_tokens)
        v2, _ = self.self_attn(v2, v2, v2)
        v_tokens = v_tokens + v2
        
        # 2. Cross-attention to image features
        v2 = self.norm2(v_tokens)
        v2 = self.cross_attn_img(v2, img_feat)
        v_tokens = v_tokens + v2
        
        # 3. Cross-attention to state (treat state as single token)
        v2 = self.norm3(v_tokens)
        if state_emb.dim() == 2:
            state_emb = state_emb.unsqueeze(1)  # [B, 1, d_model]
        v2 = self.cross_attn_state(v2, state_emb)
        v_tokens = v_tokens + v2
        
        # 4. Feed-forward with AdaGN
        v2 = self.norm4(v_tokens)
        # Apply AdaGN modulation per token
        v2_flat = v2.view(-1, v2.shape[-1])  # [B*Nv, d_model]
        time_expanded = time_emb.unsqueeze(1).expand(-1, v_tokens.shape[1], -1).reshape(-1, time_emb.shape[-1])
        v2_flat = self.ada_norm(v2_flat, time_expanded)
        v2_flat = self.ff(v2_flat)
        v2 = v2_flat.view(B, v_tokens.shape[1], -1)
        v_tokens = v_tokens + v2
        
        return v_tokens


class DiffusionPlannerDecoder(nn.Module):
    """
    Diffusion-based Planner Decoder
    
    Replaces the autoregressive Transformer Decoder with a diffusion model
    that iteratively denoises velocity from random noise.
    
    Architecture:
    - Time embedding projection
    - Velocity tokenization + positional encoding
    - N x DiffusionDecoderBlock (Self-Attn + Cross-Attn to img/state + FF)
    - Output head predicting noise
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        time_emb_dim: int = 512,
        num_velocity_tokens: int = 4,  # Number of tokens to represent velocity
        max_timesteps: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_velocity_tokens = num_velocity_tokens
        self.max_timesteps = max_timesteps
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        # Velocity input projection (3D velocity -> token sequence)
        self.vel_proj = nn.Linear(3, d_model * num_velocity_tokens)
        self.vel_pos_embed = nn.Parameter(torch.randn(1, num_velocity_tokens, d_model) * 0.02)
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            DiffusionDecoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_ff=dim_ff,
                dropout=dropout,
                time_emb_dim=time_emb_dim,
            )
            for _ in range(num_layers)
        ])
        
        # Final norm and output head
        self.final_norm = nn.LayerNorm(d_model)
        self.noise_pred_head = nn.Sequential(
            nn.Linear(d_model * num_velocity_tokens, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3)  # Predict noise in 3D velocity space
        )
        
    def forward(
        self,
        v_noisy: torch.Tensor,       # [B, 3] noisy velocity
        timestep: torch.Tensor,      # [B] timestep (int)
        img_feat: torch.Tensor,      # [B, S, P, d_model] or [B, P, d_model] image features
        state_emb: torch.Tensor,     # [B, d_model] state embedding
    ) -> torch.Tensor:
        """
        Predict the noise added to velocity at given timestep.
        
        Returns:
            noise_pred: [B, 3] predicted noise
        """
        B = v_noisy.shape[0]
        
        # 1. Time embedding
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(B)
        time_emb = self.time_mlp(timestep.float())  # [B, time_emb_dim]
        
        # 2. Velocity tokenization
        v_tokens = self.vel_proj(v_noisy)  # [B, d_model * num_tokens]
        v_tokens = v_tokens.view(B, self.num_velocity_tokens, self.d_model)
        v_tokens = v_tokens + self.vel_pos_embed
        
        # 3. Prepare image features (use last frame if multi-frame)
        if img_feat.dim() == 4:
            img_feat = img_feat[:, -1, :, :]  # [B, P, d_model]
        
        # 4. Pass through decoder blocks
        for block in self.blocks:
            v_tokens = block(v_tokens, img_feat, state_emb, time_emb)
        
        # 5. Predict noise
        v_tokens = self.final_norm(v_tokens)
        v_flat = v_tokens.view(B, -1)  # [B, num_tokens * d_model]
        noise_pred = self.noise_pred_head(v_flat)  # [B, 3]
        
        return noise_pred


class DiffusionPlanner(nn.Module):
    """
    Wrapper for diffusion training and sampling
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_velocity_tokens: int = 4,
    ):
        super().__init__()
        
        self.num_train_timesteps = num_train_timesteps
        self.d_model = d_model
        
        # Diffusion decoder
        self.decoder = DiffusionPlannerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            num_velocity_tokens=num_velocity_tokens,
            max_timesteps=num_train_timesteps,
        )
        
        # Diffusion schedule (linear beta schedule)
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: add noise to velocity"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise"""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].view(-1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Posterior q(x_{t-1} | x_t, x_0)"""
        posterior_mean_coef1 = self.betas[t] * torch.sqrt(self.alphas_cumprod_prev[t]) / (1.0 - self.alphas_cumprod[t])
        posterior_mean_coef2 = torch.sqrt(self.alphas[t]) * (1.0 - self.alphas_cumprod_prev[t]) / (1.0 - self.alphas_cumprod[t])
        
        posterior_mean = posterior_mean_coef1.view(-1, 1) * x_start + posterior_mean_coef2.view(-1, 1) * x_t
        posterior_variance = self.posterior_variance[t].view(-1, 1)
        posterior_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, img_feat: torch.Tensor, state_emb: torch.Tensor):
        """Model prediction p(x_{t-1} | x_t)"""
        # Predict noise
        noise_pred = self.decoder(x_t, t, img_feat, state_emb)
        
        # Predict x_0
        x_recon = self.predict_start_from_noise(x_t, t, noise_pred)
        
        # Clamp for stability
        x_recon = torch.clamp(x_recon, -10, 10)
        
        # Compute posterior
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, img_feat: torch.Tensor, state_emb: torch.Tensor):
        """Single denoising step"""
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, t, img_feat, state_emb)
        
        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        return model_mean + torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def sample(
        self,
        img_feat: torch.Tensor,
        state_emb: torch.Tensor,
        batch_size: int,
        num_inference_steps: int = 50,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Generate velocity from noise using DDPM sampling
        
        Args:
            img_feat: [B, S, P, d_model] or [B, P, d_model]
            state_emb: [B, d_model]
            batch_size: number of samples
            num_inference_steps: number of denoising steps (can be < num_train_timesteps)
        
        Returns:
            velocity: [B, 3]
        """
        # Start from random noise
        shape = (batch_size, 3)
        x_t = torch.randn(shape, device=device)
        
        # Create sampling schedule
        timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch, img_feat, state_emb)
        
        return x_t
    
    def forward(
        self,
        v_gt: torch.Tensor,          # [B, 3] ground truth velocity (normalized)
        img_feat: torch.Tensor,      # [B, S, P, d_model] or [B, P, d_model]
        state_emb: torch.Tensor,     # [B, d_model]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass
        
        Returns:
            loss: scalar
            noise_pred: [B, 3] predicted noise
        """
        B = v_gt.shape[0]
        device = v_gt.device
        
        # Sample random timestep
        t = torch.randint(0, self.num_train_timesteps, (B,), device=device, dtype=torch.long)
        
        # Sample noise
        noise = torch.randn_like(v_gt)
        
        # Add noise to velocity
        v_noisy = self.q_sample(v_gt, t, noise)
        
        # Predict noise
        noise_pred = self.decoder(v_noisy, t, img_feat, state_emb)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss, noise_pred
