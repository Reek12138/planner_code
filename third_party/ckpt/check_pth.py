import torch
ckpt = torch.load("/home/zhuyi/planner_code/third_party/ckpt/aggregator/vggt_aggregator.pt", map_location="cpu")
keys = ckpt["aggregator_state_dict"].keys()
print("num keys:", len(keys))
print("has patch_embed:", any(k.startswith("patch_embed.") for k in keys))
print([k for k in keys if k.startswith("patch_embed.")][:20])
sd = ckpt["aggregator_state_dict"]
print("has frame_blocks:", any(k.startswith("frame_blocks.") for k in sd))
print("has global_blocks:", any(k.startswith("global_blocks.") for k in sd))
