"""AdaptCLIP meta.json format with paired class-aware augmentation."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class AnomalyDataset(Dataset):
    def __init__(self, root, split="test", image_size=518, backbone="clip", caa=False, records=None, training=False,
                 caa_probability=0.5, caa_grids=(2, 3), mosaic_probability=0.5, rotation=True,
                 horizontal_flip=True, vertical_flip=True):
        self.root = Path(root)
        metadata = json.loads((self.root / "meta.json").read_text())
        self.records = records if records is not None else [dict(item, cls_name=category) for category, items in metadata[split].items() for item in items]
        if not self.records:
            raise ValueError(f"No records in {root} split {split}")
        self.image_size, self.caa = image_size, caa
        self.training = training
        self.rotation = rotation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.caa_probability = caa_probability
        self.caa_grids = tuple(caa_grids)
        self.mosaic_probability = mosaic_probability
        self.mean = (0.48145466, 0.4578275, 0.40821073) if backbone in ("clip", "eva02_clip") else (0.485, 0.456, 0.406)
        self.std = (0.26862954, 0.26130258, 0.27577711) if backbone in ("clip", "eva02_clip") else (0.229, 0.224, 0.225)
        self.groups = {}
        for i, item in enumerate(self.records):
            self.groups.setdefault(item["cls_name"], []).append(i)

    def __len__(self):
        return len(self.records)

    def raw(self, index):
        item = self.records[index]
        with Image.open(self.root / item["img_path"]) as source:
            image = source.convert("RGB")
        mask = Image.new("L", image.size)
        has_mask = True
        if item["anomaly"]:
            path = self.root / item.get("mask_path", "")
            has_mask = path.is_file()
            if has_mask:
                with Image.open(path) as source:
                    mask = Image.fromarray((np.array(source.convert("L")) > 0).astype("uint8") * 255)
                if mask.size != image.size:
                    mask = mask.resize(image.size, Image.Resampling.NEAREST)
        return image, mask, has_mask

    def augment(self, index, image, mask):
        item = self.records[index]
        n = random.choice(self.caa_grids)
        if random.random() < self.mosaic_probability:
            # Preserve the candidate and sample the remaining tiles within its class.
            pool = self.groups[item["cls_name"]]
            if not item["anomaly"]:
                pool = [i for i in pool if not self.records[i]["anomaly"]]
            indices = [index] + random.choices(pool, k=n*n-1)
            random.shuffle(indices)
            canvas, target = Image.new("RGB", (self.image_size,)*2), Image.new("L", (self.image_size,)*2)
            for tile, idx in enumerate(indices):
                im, ma, valid = self.raw(idx)
                if not valid:
                    raise ValueError("CAA training requires pixel masks for anomaly images")
                row, col = divmod(tile, n)
                box = (col*self.image_size//n, row*self.image_size//n, (col+1)*self.image_size//n, (row+1)*self.image_size//n)
                size = (box[2]-box[0], box[3]-box[1])
                canvas.paste(im.resize(size, Image.Resampling.BICUBIC), box)
                target.paste(ma.resize(size, Image.Resampling.NEAREST), box)
            return canvas, target
        w, h = image.size
        boxes = [(c*w//n, r*h//n, (c+1)*w//n, (r+1)*h//n) for r in range(n) for c in range(n)]
        boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
        if item["anomaly"]:
            boxes = [b for b in boxes if mask.crop(b).getbbox() is not None]
        if not boxes:
            return image, mask
        box = random.choice(boxes)
        return image.crop(box), mask.crop(box)

    @staticmethod
    def augment_geometry(image, mask, rotation=True, horizontal_flip=True, vertical_flip=True):
        """Apply the same lossless random rotation and flips to image and target."""
        if rotation:
            angle = random.choice((0, 90, 180, 270))
            image = image.rotate(angle, expand=True)
            mask = mask.rotate(angle, expand=True)
        if horizontal_flip and random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if vertical_flip and random.random() < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        return image, mask

    def __getitem__(self, index):
        item = self.records[index]
        image, mask, valid = self.raw(index)
        if self.caa and not valid:
            raise ValueError("Training requires pixel masks for anomalous images")
        if self.caa and random.random() < self.caa_probability:
            image, mask = self.augment(index, image, mask)
        if self.training:
            image, mask = self.augment_geometry(image, mask, rotation=self.rotation,
                                                horizontal_flip=self.horizontal_flip,
                                                vertical_flip=self.vertical_flip)
        size = (self.image_size,)*2
        image = TF.normalize(TF.to_tensor(image.resize(size, Image.Resampling.BICUBIC)), self.mean, self.std)
        mask = torch.from_numpy((np.array(mask.resize(size, Image.Resampling.NEAREST)) > 0).astype("float32"))
        return {"image": image, "mask": mask, "label": int(item["anomaly"]), "category": item["cls_name"], "view": str(item.get("view_id", "0")), "has_mask": valid, "path": item["img_path"]}


def normal_references(root, shots, seed):
    """Sample without replacement from train only, independently per class/view."""
    metadata = json.loads((Path(root)/"meta.json").read_text())
    groups = {}
    for category, records in metadata.get("train", {}).items():
        for item in records:
            if not item["anomaly"]:
                groups.setdefault((category, str(item.get("view_id", "0"))), []).append(dict(item, cls_name=category))
    if not groups:
        raise ValueError("Few-shot evaluation requires normal images in the train split")
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        if len(groups[key]) < shots:
            raise ValueError(f"{key}: requested {shots} references but only {len(groups[key])} available")
        selected.extend(rng.sample(groups[key], shots))
    return selected
