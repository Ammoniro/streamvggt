from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from vggt.models.vggt import VGGT
import torch

print("Loading pretrained DINO-vitl model.")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
vitl_path = "/home/chensiyu/vitl/dinov2_vitl14_reg4_pretrain.pth"
vitl = vit_large(
    img_size=518,
    patch_size=14,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=False,
    block_chunks=0,
    init_values=1.0,
).to(device)
ckpt_vitl = torch.load(vitl_path, map_location=device)
vitl.load_state_dict(ckpt_vitl, strict=True)


print("Loading pretrained vggt model")
vggt_path = "/data/chensiyu/pretrained/vggt/model.pt"
vggt = VGGT().to(device)
ckpt_vggt = torch.load(vggt_path, map_location=device)
vggt.load_state_dict(ckpt_vggt, strict=True)