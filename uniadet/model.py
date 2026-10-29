"""Language-free task/layer decoupling (paper equations 8–12)."""
import torch
from torch import nn
import torch.nn.functional as F


class DecoupledHead(nn.Module):
    def __init__(self, dims, temperature=0.07):
        super().__init__()
        if not dims or temperature <= 0:
            raise ValueError("Need feature dimensions and positive temperature")
        self.temperature = temperature
        self.cls = nn.ParameterList([nn.Parameter(torch.randn(2, d) * d**-0.5) for d in dims])
        self.seg = nn.ParameterList([nn.Parameter(torch.randn(2, d) * d**-0.5) for d in dims])

    def forward(self, features, size):
        if len(features) != len(self.cls):
            raise ValueError("Feature levels do not match head")
        global_logits, local_logits = [], []
        for (global_token, patches, grid), wc, ws in zip(features, self.cls, self.seg):
            global_logits.append(F.linear(F.normalize(global_token.float(), dim=-1), F.normalize(wc, dim=-1)) / self.temperature)
            logits = F.linear(F.normalize(patches.float(), dim=-1), F.normalize(ws, dim=-1)) / self.temperature
            logits = logits.transpose(1, 2).reshape(patches.shape[0], 2, *grid)
            local_logits.append(F.interpolate(logits, size=size, mode="bilinear", align_corners=False))
        return global_logits, local_logits


class NormalMemory:
    """Separate class/view banks; exact cosine NN with bounded working memory."""
    def __init__(self, chunk_size=1024):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.banks = {}

    def add(self, key, features):
        values = [F.normalize(p.detach().float().flatten(0, 1), dim=-1) for _, p, _ in features]
        if key in self.banks:
            values = [torch.cat((a, b)) for a, b in zip(self.banks[key], values)]
        self.banks[key] = values

    def predict(self, key, features, size):
        if key not in self.banks:
            raise KeyError(f"No normal reference images for {key}")
        maps = []
        if len(features) != len(self.banks[key]):
            raise ValueError("Memory feature levels mismatch")
        for (_, patches, grid), bank in zip(features, self.banks[key]):
            query = F.normalize(patches.float(), dim=-1)
            flat = query.flatten(0, 1)
            distances = []
            for q in flat.split(self.chunk_size):
                nearest = torch.full((q.shape[0],), -1., device=q.device)
                for m in bank.split(self.chunk_size):
                    nearest = torch.maximum(nearest, (q @ m.to(q.device).T).amax(-1))
                distances.append((1 - nearest).clamp(0, 2))
            distance = torch.cat(distances).reshape(query.shape[0], 1, *grid)
            maps.append(F.interpolate(distance, size=size, mode="bilinear", align_corners=False))
        return torch.stack(maps).mean(0)[:, 0]


class UniADet(nn.Module):
    def __init__(self, encoder, temperature=0.07, lambda_f=0.5, lambda_p=0.5):
        super().__init__()
        if not (0 <= lambda_f <= 1 and 0 <= lambda_p <= 1):
            raise ValueError("Fusion weights must be in [0, 1]")
        self.encoder = encoder.requires_grad_(False).eval()
        self.head = DecoupledHead(encoder.dims, temperature)
        self.lambda_f, self.lambda_p = lambda_f, lambda_p

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.encoder(images)
        return self.head(features, images.shape[-2:])

    @torch.no_grad()
    def predict(self, images, memory=None, key=None, zero_score_pooling="layer"):
        if zero_score_pooling not in ("layer", "map"):
            raise ValueError("zero_score_pooling must be layer or map")
        features = self.encoder(images)
        global_logits, local_logits = self.head(features, images.shape[-2:])
        score = torch.stack([x.softmax(-1)[:, 1] for x in global_logits]).mean(0)
        layer_maps = torch.stack([x.softmax(1)[:, 1] for x in local_logits])
        anomaly_map = layer_maps.mean(0)
        if memory is not None:
            reference_map = memory.predict(key, features, images.shape[-2:])
            anomaly_map = (1-self.lambda_f) * anomaly_map + self.lambda_f * reference_map
        # Sec. 3.5 fuses each layer's global score and spatial maximum.
        # max(mean(maps)) != mean(max(maps)) when layers localize differently.
        if memory is None and zero_score_pooling == "layer":
            peak = layer_maps.flatten(2).amax(2).mean(0)
        else:
            peak = anomaly_map.flatten(1).amax(1)
        score = (1-self.lambda_p) * score + self.lambda_p * peak
        return score, anomaly_map
