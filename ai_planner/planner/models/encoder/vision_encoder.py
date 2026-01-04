# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from collections import deque
from PIL import Image

from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


# ========= 抽象主干接口 =========
class VisionBackbone(nn.Module):
    """
    视觉主干抽象接口：输入单帧图像，输出 (tokens, deepstack_list, coords_uv)
      - tokens: [N_tokens, D_backbone] （注意：这里的 tokens 是“合并（merger）之后”的）
      - deepstack_list: List[[N_tokens_l, D_backbone]]（可选）
      - coords_uv: [N_tokens, 2]，与 tokens 一一对应，(u,v)∈[0,1]，行优先（row-major）
    """
    def encode_pil(self, img: Image.Image) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    @property
    def out_dim(self) -> int:
        raise NotImplementedError


# ========= Qwen3-VL ViT 主干实现（固定 resize 后再经 processor；返回合并后 token + 坐标） =========
class QwenVitBackbone(VisionBackbone):
    """
    使用 Qwen3VLVisionModel 作为视觉主干。
    - 进入 processor 之前先把图片固定缩放到 image_size（默认 448×448）
    - 不依赖 processor 返回的 grid_thw；按固定尺寸+patch_size 构造 grid_thw
    - 依据 spatial_merge_size 计算“合并后网格”的 (u,v) 坐标并返回
    """
    def __init__(
        self,
        vit_dir: str,
        proc_dir: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        image_size: Tuple[int, int] = (448, 448),
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.image_size = image_size

        # 1) 加载 ViT-only
        cfg = AutoConfig.from_pretrained(vit_dir)
        self.vit = Qwen3VLVisionModel(cfg)
        state = torch.load(f"{vit_dir}/pytorch_model.bin", map_location="cpu")
        self.vit.load_state_dict(state, strict=True)
        self.vit.to(self.device, dtype=self.dtype)
        self.vit.eval()

        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

        # 2) Processor（仅用于标准化；resize 已由我们手动完成）
        self.processor = AutoProcessor.from_pretrained(proc_dir)
        # 记录输出维度（来自 merger 输出维度）
        self._out_dim = cfg.out_hidden_size

        # 关键参数：patch 与 merge
        self.patch_size = int(getattr(cfg, "patch_size", 14))
        self.merge_size = int(getattr(cfg, "spatial_merge_size", 2))

        # 固定网格（所有图片一致）
        H, W = self.image_size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"image_size 必须是 patch_size 的整数倍；当前 {self.image_size} / patch_size={self.patch_size}"
        self.H_p = H // self.patch_size
        self.W_p = W // self.patch_size

        # 合并后网格大小
        assert self.H_p % self.merge_size == 0 and self.W_p % self.merge_size == 0, \
            f"(H_p, W_p)=({self.H_p},{self.W_p}) 需能被 merge_size={self.merge_size} 整除"
        self.Hm = self.H_p // self.merge_size
        self.Wm = self.W_p // self.merge_size
        self.N_tokens = self.Hm * self.Wm

        # 预先生成“合并后网格”的归一化 (u,v) 坐标模板（行优先）
        yy, xx = torch.meshgrid(
            torch.arange(self.Hm, device=self.device),
            torch.arange(self.Wm, device=self.device),
            indexing="ij"
        )
        u = (xx.to(self.dtype) + 0.5) / float(self.Wm)
        v = (yy.to(self.dtype) + 0.5) / float(self.Hm)
        self.coords_template = torch.stack([u, v], dim=-1).reshape(-1, 2)  # [N,2]

    @torch.no_grad()
    def encode_pil(self, img: Image.Image) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        # 先固定 resize，再交给 processor
        img = img.convert("RGB").resize(self.image_size, Image.BICUBIC)

        # 关键：用 processor 的 grid_thw，保证与 patch_embed 对齐
        proc_out = self.processor(images=[img], text=[""], return_tensors="pt")
        pix  = proc_out["pixel_values"].to(self.device, dtype=self.dtype)     # [1,3,H,W]
        grid = proc_out["image_grid_thw"].to(self.device)                     # [[T,H_p,W_p]]

        merged_feats, deepstack_feats = self.vit(pix, grid_thw=grid)          # [N_merge, D]

        # 用“真实 grid_thw + merge_size”算合并后坐标（行优先）
        T, H_p, W_p = grid[0].tolist()
        m = int(self.vit.config.spatial_merge_size)
        Hm, Wm = H_p // m, W_p // m
        yy, xx = torch.meshgrid(
            torch.arange(Hm, device=self.device),
            torch.arange(Wm, device=self.device),
            indexing="ij",
        )
        u = (xx.to(self.dtype) + 0.5) / float(Wm)
        v = (yy.to(self.dtype) + 0.5) / float(Hm)
        coords_one = torch.stack([u, v], dim=-1).reshape(-1, 2)   # [Hm*Wm,2]
        coords_uv  = coords_one.repeat(int(T), 1)                 # [T*Hm*Wm,2] == merged_feats.shape[0]

        return merged_feats, deepstack_feats, coords_uv

    @torch.no_grad()
    def encode_pil_batch(self, imgs: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 统一 resize（保证视觉尺寸稳定）
        resized = [im.convert("RGB").resize(self.image_size, Image.BICUBIC) for im in imgs]

        proc = self.processor(images=resized, text=[""] * len(resized), return_tensors="pt")
        pix  = proc["pixel_values"].to(self.device, dtype=self.dtype)          # [M,3,H,W]
        grid = proc["image_grid_thw"].to(self.device)                           # [M,3]

        merged_feats, _ = self.vit(pix, grid_thw=grid)                          # [sum_i N_i_merge, D]
        D = merged_feats.size(-1)
        m = int(self.vit.config.spatial_merge_size)

        # 计算每张图的合并后 token 数 N_i = T_i * (H_p/m) * (W_p/m)
        Ns = []
        for (T, H_p, W_p) in grid.tolist():
            Ns.append(int(T) * (H_p // m) * (W_p // m))

        # 按 N_i 拆分
        feats_list = list(torch.split(merged_feats, Ns, dim=0))

        # 生成每张图的 coords，并 pad 到 N_max（若你确认 Ns 全相等，可以直接 stack）
        coords_list = []
        for (T, H_p, W_p) in grid.tolist():
            Hm, Wm = (H_p // m), (W_p // m)
            yy, xx = torch.meshgrid(
                torch.arange(Hm, device=self.device),
                torch.arange(Wm, device=self.device),
                indexing="ij",
            )
            u = (xx.to(self.dtype) + 0.5) / float(Wm)
            v = (yy.to(self.dtype) + 0.5) / float(Hm)
            coords_one = torch.stack([u, v], dim=-1).reshape(-1, 2)   # [Hm*Wm,2]
            coords_list.append(coords_one.repeat(int(T), 1))          # [N_i,2]

        N_max = max(Ns)
        feats_padded  = []
        coords_padded = []
        for f, c in zip(feats_list, coords_list):
            n = f.size(0)
            if n < N_max:
                feats_padded.append(F.pad(f, (0, 0, 0, N_max - n)))   # [N_max,D]
                coords_padded.append(F.pad(c, (0, 0, 0, N_max - n)))  # [N_max,2]
            else:
                feats_padded.append(f)
                coords_padded.append(c)

        feats_padded  = torch.stack(feats_padded,  dim=0)             # [M,N_max,D]
        coords_padded = torch.stack(coords_padded, dim=0)             # [M,N_max,2]

        # 如果你已经固定了 processor 的尺寸策略，通常 Ns 都相等，此时你也可以：
        # feats_padded = merged_feats.view(M, N_max, D); coords_padded = coords_list[0].unsqueeze(0).repeat(M,1,1)

        # 这里返回共享的模板更方便：若 Ns 全相等，就返回 coords_padded[0]
        if len(set(Ns)) == 1:
            return feats_padded, coords_padded[0]
        else:
            # 若仍有不等，下游就用 coords_padded 按 batch 取（你可以把返回签名改为返回 coords_padded）
            return feats_padded, coords_padded[0]  # 或改成返回 coords_padded

    @property
    def out_dim(self) -> int:
        return self._out_dim


# ========= 训练期：Batched 编码器 =========
class VisionEncoderBatched(nn.Module):
    """
    训练期推荐：一次性把 B*T 张图过主干，reshape 回 [B, T, N, D_proj]
    并返回 frame_mask、coords_uv。
    """
    def __init__(self, backbone: VisionBackbone, proj_dim: Optional[int] = None):
        super().__init__()
        self.backbone = backbone
        d_in = backbone.out_dim
        if proj_dim is not None and proj_dim != d_in:
            self.proj = nn.Linear(d_in, proj_dim, bias=True).to(
                next(backbone.parameters()).device,
                dtype=next(backbone.parameters()).dtype,
            )
            self.out_dim = proj_dim
        else:
            self.proj = nn.Identity()
            self.out_dim = d_in

        self._coords_uv_template: Optional[torch.Tensor] = None  # [N,2]

    @torch.no_grad()
    def _encode_images_backbone(self, imgs: List[Image.Image]):
        feats, coords_uv = self.backbone.encode_pil_batch(imgs)   # [M,N,D_b], [N,2]
        return feats, coords_uv

    def encode_batch_sequences(
        self,
        batch_seqs: List[List[Image.Image]],
        T_fixed: Optional[int] = None,
    ):
        """
        输入：batch 的图像序列（长度B，每个是长度Ti的PIL列表）
        输出：
          feats: [B, T_max, N, D_proj]
          frame_mask: [B, T_max]  (True=pad)
          coords_uv: [N, 2]
        """
        # 1) pad 到同一长度
        T_list = [len(seq) for seq in batch_seqs]
        T_max = T_fixed or max(T_list)
        pad_img = Image.new("RGB", self.backbone.image_size)
        imgs_flat, mask = [], []
        for seq in batch_seqs:
            cur = seq[:T_max] + [pad_img] * max(0, T_max - len(seq))
            imgs_flat.extend(cur)
            mask.append([False]*min(len(seq), T_max) + [True]*max(0, T_max - len(seq)))

        # 2) 主干一次过 B*T_max（主干冻结可 no_grad；投影层要梯度）
        with torch.no_grad():
            feats_m, coords_uv = self._encode_images_backbone(imgs_flat)  # [B*T,N,D_b], [N,2]

        # 3) 线性投影（需要梯度）
        feats_m = self.proj(feats_m)  # [B*T,N,D_proj]

        # 4) reshape 回 [B, T_max, N, D]
        B = len(batch_seqs)
        N = feats_m.shape[1]
        D = feats_m.shape[2]
        feats = feats_m.view(B, T_max, N, D).contiguous()               # [B,T,N,D]
        frame_mask = torch.tensor(mask, device=feats.device)            # [B,T]

        return feats, frame_mask, coords_uv.to(feats.device, feats.dtype)

    # 可选：把 encode_batch_sequences 作为 forward 语义（按需）
    def forward(self, batch_seqs: List[List[Image.Image]], T_fixed: Optional[int] = None):
        return self.encode_batch_sequences(batch_seqs, T_fixed=T_fixed)


# ========= 推理期：FIFO 编码器（可选，实时/滑窗） =========
class VisionEncoderFIFO(nn.Module):
    """
    在线/实时场景：维护最近 T 帧窗口。
    返回 [T, N, D_proj]；支持取 coords_uv 模板。
    """
    def __init__(
        self,
        backbone: VisionBackbone,
        fifo_len: int = 6,
        proj_dim: Optional[int] = None,
        use_mean_pool: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.fifo_len = fifo_len
        self.use_mean_pool = use_mean_pool
        self._buffer: deque[torch.Tensor] = deque(maxlen=fifo_len)  # 每项: [N, D_proj或D_backbone]
        self._coords_uv: Optional[torch.Tensor] = None

        d_in = backbone.out_dim
        if proj_dim is not None and proj_dim != d_in:
            self.proj = nn.Linear(d_in, proj_dim, bias=True).to(
                next(backbone.parameters()).device,
                dtype=next(backbone.parameters()).dtype,
            )
            self.out_dim = proj_dim
        else:
            self.proj = nn.Identity()
            self.out_dim = d_in

    @torch.no_grad()
    def _encode_one(self, img: Image.Image):
        tokens, _deepstack, coords = self.backbone.encode_pil(img)     # [N,D_b], _, [N,2]
        tokens = tokens.to(next(self.backbone.parameters()).device)
        coords = coords.to(tokens.device, dtype=torch.float16 if tokens.dtype==torch.float16 else torch.float32)
        return tokens, coords

    def push(self, img: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(img, str):
            img = Image.open(img)
        with torch.no_grad():
            tokens, coords = self._encode_one(img)
        tokens = self.proj(tokens)                                     # [N,D_proj]
        self._buffer.append(tokens.detach())
        if self._coords_uv is None:
            self._coords_uv = coords.detach()
        return tokens

    def clear(self):
        self._buffer.clear()

    def get_patch_coords(self) -> torch.Tensor:
        assert self._coords_uv is not None, "请先 push 至少一帧以建立坐标模板"
        return self._coords_uv  # [N,2]

    def get_sequence(self, with_coords: bool = False):
        assert len(self._buffer) > 0, "FIFO 当前为空，请先 push 至少一帧。"
        seq = torch.stack(list(self._buffer), dim=0)  # [T,N,D]
        if self.use_mean_pool:
            seq = seq.mean(dim=1)                     # [T,D]
            if with_coords:
                raise ValueError("use_mean_pool=True 时没有 token 维，无法返回坐标")
            return seq
        if with_coords:
            return seq, self.get_patch_coords()
        return seq

    @torch.no_grad()
    def forward(self, img: Union[str, Image.Image]) -> torch.Tensor:
        self.push(img)
        return self.get_sequence()


# ================= Demo =================
if __name__ == "__main__":
    # 路径替换成你的
    vit_dir = "/home/zhuyi/planner_code/ai_planner/utils/qwen3_vit_only"
    proc_dir = "/home/zhuyi/planner_code/qwen3_vl_instruct_2B"

    backbone = QwenVitBackbone(
        vit_dir=vit_dir,
        proc_dir=proc_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        image_size=(224, 224),
        freeze_backbone=True,
    )

    # ===== 训练期：批处理 =====
    encoder_b = VisionEncoderBatched(backbone=backbone, proj_dim=1024).eval()

    # 假造一个 batch：B=2，每条序列 T=[4,6]
    img_path = "/home/zhuyi/Pictures/Figure_1.png"
    seq1 = [Image.open(img_path)] * 3
    seq2 = [Image.open(img_path)] * 3
    feats, frame_mask, coords_uv = encoder_b([seq1, seq2])  # [B,T,N,D], [B,T], [N,2]

    print("[Batched] feats:", tuple(feats.shape))
    print("[Batched] frame_mask:", tuple(frame_mask.shape), "sum_mask=", frame_mask.sum().item())
    print("[Batched] coords_uv:", tuple(coords_uv.shape))

    # ===== 推理期：FIFO =====
    encoder_f = VisionEncoderFIFO(backbone=backbone, fifo_len=6, proj_dim=1024, use_mean_pool=False).eval()
    for _ in range(6):
        encoder_f.push(img_path)
    seq_f, coords_f = encoder_f.get_sequence(with_coords=True)
    print("[FIFO] seq:", tuple(seq_f.shape))
    print("[FIFO] coords:", tuple(coords_f.shape))

    # 主干是否冻结
    n_trainable = sum(p.requires_grad for p in backbone.parameters())
    print("主干可训练参数数目:", n_trainable)  # 应为 0

    # 投影层可训练参数
    n_trainable_proj_b = sum(p.requires_grad for p in encoder_b.proj.parameters())
    n_trainable_proj_f = sum(p.requires_grad for p in encoder_f.proj.parameters())
    print("投影层(训练版)可训练参数数目:", n_trainable_proj_b)
    print("投影层(FIFO)可训练参数数目:", n_trainable_proj_f)
