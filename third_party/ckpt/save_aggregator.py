import sys
_VGGT_ROOT = "/workspace/zhuy25@xiaopeng.com/planner_code/third_party/vggt"
if _VGGT_ROOT not in sys.path:
    sys.path.insert(0, _VGGT_ROOT)

import torch
from vggt.models.vggt import VGGT

ckpt_dir = "/publicdata/huggingface.co/facebook/VGGT-1B"  # 本地模型目录
save_path = "/workspace/zhuy25@xiaopeng.com/planner_code/third_party/ckpt/aggregator/vggt_aggregator.pt"

model = VGGT.from_pretrained(ckpt_dir, map_location="cuda")
print("model loaded")
agg = model.aggregator

payload = {
    "aggregator_state_dict": agg.state_dict(),
    # 强烈建议把构造参数也存下来，未来才能100%对得上结构
    "aggregator_cfg": {
        "img_size": agg.patch_embed.img_size if hasattr(agg.patch_embed, "img_size") else 518,
        "patch_size": agg.patch_size,
        "embed_dim": agg.camera_token.shape[-1],
        "depth": agg.depth,
        "num_register_tokens": agg.register_token.shape[2],
        "aa_order": agg.aa_order,
        "aa_block_size": agg.aa_block_size,
        # rope 相关如果你未来要完全一致，也可以补充存
        "rope_freq": getattr(getattr(agg, "rope", None), "frequency", None),
    }
}

torch.save(payload, save_path)
print("saved:", save_path)
