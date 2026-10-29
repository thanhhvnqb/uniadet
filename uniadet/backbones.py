"""Frozen CLIP (AdaptCLIP weights/blocks) and official DINO hub encoders."""
import torch
from torch import nn
import torch.nn.functional as F


class CLIPEncoder(nn.Module):
    def __init__(self, visual, layers):
        super().__init__()
        self.visual = visual
        self.layers = tuple(layers)
        if sorted(set(layers)) != list(layers) or not layers or layers[0] < 1 or layers[-1] > len(visual.transformer.resblocks):
            raise ValueError("layers must be unique, ascending, one-based block indices")
        dim = visual.proj.shape[1] if visual.proj is not None else visual.conv1.out_channels
        self.dims = [dim] * len(layers)
        self.patch_size = visual.conv1.kernel_size[0]

    @torch.no_grad()
    def forward(self, images):
        v = self.visual
        if any(s % self.patch_size for s in images.shape[-2:]):
            raise ValueError("Image dimensions must be divisible by patch size")
        x = v.conv1(images.to(v.conv1.weight.dtype))
        grid = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        cls = v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat((cls, x), 1)
        pos = v.positional_embedding.to(x.dtype)
        side = int((pos.shape[0]-1)**0.5)
        spatial = pos[1:].reshape(side, side, -1).permute(2, 0, 1)[None]
        spatial = F.interpolate(spatial, size=grid, mode="bicubic", align_corners=False)
        pos = torch.cat((pos[:1], spatial[0].flatten(1).T))
        x = v.ln_pre(x + pos).transpose(0, 1)
        features = []
        for i, block in enumerate(v.transformer.resblocks, 1):
            x = block(x)
            if i in self.layers:
                tokens = v.ln_post(x.transpose(0, 1))
                if v.proj is not None:
                    tokens = tokens @ v.proj
                features.append((tokens[:, 0].float(), tokens[:, 1:].float(), grid))
            if i == self.layers[-1]:
                break
        return features


class DINOEncoder(nn.Module):
    def __init__(self, model, layers, patch_size):
        super().__init__()
        self.model, self.layers, self.patch_size = model, tuple(layers), patch_size
        if not layers or sorted(set(layers)) != list(layers) or layers[0] < 1 or layers[-1] > getattr(model, "n_blocks", len(model.blocks)):
            raise ValueError("Invalid one-based DINO feature layers")
        self.dims = [model.embed_dim] * len(layers)

    @torch.no_grad()
    def forward(self, images):
        if any(s % self.patch_size for s in images.shape[-2:]):
            raise ValueError("Image dimensions must be divisible by patch size")
        outputs = self.model.get_intermediate_layers(images, n=[i-1 for i in self.layers], return_class_token=True, norm=True)
        grid = tuple(s // self.patch_size for s in images.shape[-2:])
        return [(cls.float(), patches.float(), grid) for patches, cls in outputs]


def create_encoder(name="clip", layers=(12, 15, 18, 21, 24), weights=None, dino_repo=None):
    if name == "clip":
        from lib import load
        model, _ = load(weights or "ViT-L/14@336px", device="cpu")
        # Keep only vision tower; no text modules enter UniADet or the optimizer.
        return CLIPEncoder(model.visual, layers).requires_grad_(False).eval()
    if name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=weights is None)
        if weights:
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        return DINOEncoder(model, layers, 14).requires_grad_(False).eval()
    if name == "dinov3":
        if not weights:
            raise ValueError("DINOv3 requires --weights pointing to authorized pretrained weights")
        repo = dino_repo or "facebookresearch/dinov3"
        model = torch.hub.load(repo, "dinov3_vitl16", source="local" if dino_repo else "github", weights=weights)
        return DINOEncoder(model, layers, 16).requires_grad_(False).eval()
    raise ValueError(f"Unknown backbone: {name}")
