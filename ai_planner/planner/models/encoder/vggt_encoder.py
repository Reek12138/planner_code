# /home/zhuyi/planner_code/ai_planner/planner/models/encoder/vggt_encoder.py
# -*- coding: utf-8 -*-

"""
VGGT Frame Encoder (Planner use)

- 基于 原版 VGGT 的 Aggregator
- 输入：连续时间帧图像 [B,S,3,H,W] 或 [S,3,H,W]，float，建议范围 [0,1]
- 输出：聚合后的时空特征 tokens（取最后一层聚合特征）

输出说明：
    feat: Tensor [B, S, P, 2C]
        - P = (camera_token=1) + (register_tokens=num_register_tokens) + (patch_tokens=H/ps * W/ps)
        - 2C = concat(frame-attn features, global-attn features)，其中 C=embed_dim
    patch_start_idx: int
        patch token 起始 index（前面是 camera + register）
"""

import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn


# =============================================================================
# 1) 引入 third_party/vggt（相对路径）
# =============================================================================
# 当前文件：
#   /home/zhuyi/planner_code/ai_planner/planner/models/encoder/vggt_encoder.py
# vggt 目录：
#   /home/zhuyi/planner_code/third_party/vggt
# =============================================================================
# _CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# _PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "../../../../"))  
# _VGGT_ROOT = os.path.join(_PROJECT_ROOT, "third_party", "vggt")

# if _VGGT_ROOT not in sys.path:
#     sys.path.insert(0, _VGGT_ROOT)

sys.path.insert(0, "/workspace/zhuy25@xiaopeng.com/planner_code/third_party/vggt")

# 现在可以正常 import 原版 vggt
from vggt.models.aggregator import Aggregator  # noqa: E402


# =============================================================================
# 2) VGGTFrameEncoder
# =============================================================================
class VGGTFrameEncoder(nn.Module):
    """
    使用 原版 VGGT Aggregator 的连续帧 Encoder

    - 可选加载你保存的 aggregator 权重（vggt_aggregator.pt）
    - 建议你保存的 pt 内含：
        payload = {
            "aggregator_state_dict": agg.state_dict(),
            "aggregator_cfg": {...}   # 可选，但强烈建议
        }
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        pretrained_agg_path: Optional[str] = None,
        map_location: str = "cpu",
        strict: bool = True,
        rebuild_from_cfg: bool = True,
    ):
        super().__init__()

        # 先按传入参数构造一个 aggregator
        self.agg = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # 如提供权重，则加载
        if pretrained_agg_path is not None:
            self.load_aggregator(
                ckpt_path=pretrained_agg_path,
                map_location=map_location,
                strict=strict,
                rebuild_from_cfg=rebuild_from_cfg,
            )

    def load_aggregator(
        self,
        ckpt_path: str,
        map_location: str = "cpu",
        strict: bool = True,
        rebuild_from_cfg: bool = True,
    ):
        """
        从你保存的 vggt_aggregator.pt 加载 aggregator 权重。

        Args:
            ckpt_path: 你保存的 pt 文件路径
            map_location: torch.load 的 map_location
            strict: load_state_dict 是否严格匹配
            rebuild_from_cfg:
                True：如果 ckpt 里有 aggregator_cfg，就按 cfg 重建 Aggregator 再 load，最稳
                False：直接对当前 Aggregator load（要求你 init 参数完全一致）
        """
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Aggregator ckpt not found: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location=map_location, weights_only=True)

        if "aggregator_state_dict" not in payload:
            raise KeyError(f"ckpt 缺少 aggregator_state_dict: {ckpt_path}")

        state = payload["aggregator_state_dict"]
        cfg = payload.get("aggregator_cfg", None)

        # 可选：按 cfg 重建 Aggregator（避免结构差一点就 load 不上）
        if rebuild_from_cfg and isinstance(cfg, dict):
            # 只取 Aggregator 构造函数支持的字段（其余忽略）
            allowed_keys = {
                "img_size",
                "patch_size",
                "embed_dim",
                "depth",
                "num_heads",
                "mlp_ratio",
                "num_register_tokens",
                "block_fn",
                "qkv_bias",
                "proj_bias",
                "ffn_bias",
                "patch_embed",
                "aa_order",
                "aa_block_size",
                "qk_norm",
                "rope_freq",
                "init_values",
            }
            agg_kwargs = {k: v for k, v in cfg.items() if k in allowed_keys and v is not None}

            # 如果你保存脚本里没存 patch_embed 字符串，这里就别传，让它用默认
            # agg_kwargs.setdefault("patch_embed", "dinov2_vitl14_reg")

            self.agg = Aggregator(**agg_kwargs)

        # load 权重
        ret = self.agg.load_state_dict(state, strict=strict)

        # PyTorch 新版 strict=True 会直接报错；strict=False 返回 IncompatibleKeys
        if not strict:
            missing = getattr(ret, "missing_keys", [])
            unexpected = getattr(ret, "unexpected_keys", [])
            if missing:
                print("[VGGTFrameEncoder] missing keys (show up to 20):", missing[:20], "..." if len(missing) > 20 else "")
            if unexpected:
                print("[VGGTFrameEncoder] unexpected keys (show up to 20):", unexpected[:20], "..." if len(unexpected) > 20 else "")

        print(f"[VGGTFrameEncoder] loaded aggregator weights from: {ckpt_path}")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            images:
                Tensor
                - [B, S, 3, H, W] 或
                - [S, 3, H, W]
                值域建议 [0,1]（内部会按 ImageNet mean/std 做 normalize）

        Returns:
            feat:
                Tensor [B, S, P, 2C]
                - P = camera + 4*register + patch tokens
                - 2C = frame-attn + global-attn concat
            patch_start_idx:
                int
                patch token 起始 index（前面是 camera + register）
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)  # [1,S,3,H,W]
        if images.dim() != 5:
            raise ValueError(f"images must be [B,S,3,H,W] or [S,3,H,W], got shape={tuple(images.shape)}")

        aggregated_tokens_list, patch_start_idx = self.agg(images)

        # 取最后一层（最融合）
        feat = aggregated_tokens_list[-1]  # [B,S,P,2C]
        return feat, patch_start_idx


# =============================================================================
# 3) 最小测试用例
# =============================================================================
if __name__ == "__main__":
    """
    直接运行：
        python vggt_encoder.py

    你可以通过设置环境变量 VGGT_AGG_CKPT 来加载你保存的权重，例如：
        export VGGT_AGG_CKPT=/home/zhuyi/planner_code/third_party/ckpt/aggregator/vggt_aggregator.pt
        python vggt_encoder.py
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- fake input ----
    B, S = 2, 4
    H, W = 518, 518
    images = torch.rand(B, S, 3, H, W, device=device)

    ckpt_path = "/workspace/zhuy25@xiaopeng.com/planner_code/third_party/ckpt/aggregator/vggt_aggregator.pt"
    if not os.path.isfile(ckpt_path):
        ckpt_path = None  # 不存在就不加载

    encoder = VGGTFrameEncoder(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        pretrained_agg_path=ckpt_path,
        map_location="cuda",
        strict=True,
        rebuild_from_cfg=True,
    ).to(device)

    encoder.eval()

    # ------------------------------
    # 追加：推理时间 & 显存占用测试（不改变其他逻辑）
    # ------------------------------
    # warmup
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = encoder(images)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # timed run
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        feat, patch_start_idx = encoder(images)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0

    peak_mem_mb = None
    cur_mem_mb = None
    if device.startswith("cuda"):
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        cur_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    print("=" * 60)
    print("VGGTFrameEncoder test")
    print("loaded ckpt       :", ckpt_path if ckpt_path else "(None, random init)")
    print("input images shape:", tuple(images.shape))
    print("output feat shape :", tuple(feat.shape))
    print("patch_start_idx   :", patch_start_idx)

    # 常用取法示例
    # camera token (第 0 个 token)
    camera_feat = feat[:, :, 0, :]  # [B,S,2C]
    # patch tokens (从 patch_start_idx 开始)
    patch_feat = feat[:, :, patch_start_idx:, :].mean(dim=2)  # [B,S,2C]

    print("camera_feat shape :", tuple(camera_feat.shape))
    print("patch_feat  shape :", tuple(patch_feat.shape))

    # 追加打印：推理时间 / 显存
    print(f"inference time   : {elapsed_ms:.3f} ms")
    if peak_mem_mb is not None:
        print(f"cuda mem current : {cur_mem_mb:.2f} MB")
        print(f"cuda mem peak    : {peak_mem_mb:.2f} MB")

    # 可选：检查 ckpt 是否包含 patch_embed 权重（判断是不是“完整 backbone+aggregator”）
    if ckpt_path:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = payload.get("aggregator_state_dict", {})
        has_patch_embed = any(k.startswith("patch_embed.") for k in sd.keys())
        print("ckpt has patch_embed.* keys:", has_patch_embed)
        if has_patch_embed:
            sample_keys = [k for k in sd.keys() if k.startswith("patch_embed.")][:8]
            print("sample patch_embed keys:", sample_keys)

    print("=" * 60)
