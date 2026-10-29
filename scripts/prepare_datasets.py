"""Prepare available paper benchmarks using AnomalyCLIP/AdaptCLIP metadata schema."""
import csv
import json
from pathlib import Path
import zipfile

BASE = Path(__file__).resolve().parents[1] / "datasets"
EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
summary = {}

def images(path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXT)

def record(root, image, anomaly, mask=None):
    return dict(img_path=str(image.relative_to(root)), mask_path=str(mask.relative_to(root)) if mask else "", anomaly=int(anomaly))

def save(name, root, meta):
    for split in meta.values():
        for rows in split.values():
            for row in rows:
                assert (root / row["img_path"]).is_file(), row
                if row["mask_path"]:
                    assert (root / row["mask_path"]).is_file(), row
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    summary[name] = dict(root=str(root.resolve()), **{s: {"normal": sum(not r["anomaly"] for rows in cats.values() for r in rows), "anomaly": sum(r["anomaly"] for rows in cats.values() for r in rows)} for s, cats in meta.items()})
    print(name, summary[name], flush=True)

def tree(name, root, normal="good", classification=False, special_masks=None):
    meta = {"train": {}, "test": {}}
    for category in sorted(p for p in root.iterdir() if p.is_dir()):
        for split in meta:
            folder = category / split
            if not folder.is_dir():
                continue
            rows = []
            for defect in sorted(p for p in folder.iterdir() if p.is_dir()):
                anomaly = defect.name != normal
                for image in images(defect):
                    mask = None
                    if anomaly and not classification:
                        maskdir = category / (special_masks or "ground_truth")
                        if not special_masks:
                            maskdir /= defect.name
                        candidates = [m for m in images(maskdir) if m.stem.lower() in (image.stem.lower(), image.stem.lower() + "_mask", image.stem.lower() + "_segmentation")]
                        assert len(candidates) == 1, (image, candidates)
                        mask = candidates[0]
                    rows.append(record(root, image, anomaly, mask))
            meta[split][category.name] = rows
    save(name, root, meta)

visa = BASE / "VisA"
meta = {"train": {}, "test": {}}
with (visa / "split_csv/1cls.csv").open() as f:
    for r in csv.DictReader(f):
        anomaly = r["label"] == "anomaly"
        meta[r["split"]].setdefault(r["object"], []).append(record(visa, visa/r["image"], anomaly, visa/r["mask"] if anomaly else None))
save("visa", visa, meta)
tree("mvtec", BASE/"MVTec AD")
tree("btad", BASE/"btad/BTech_Dataset_transformed", normal="ok")
tree("ksdd", BASE/"KSDD/SDD")
archive = BASE/"DTD/Zero-shot texture AD/DTD-Synthetic.zip"
with zipfile.ZipFile(archive) as z:
    if not (archive.parent/"DTD-Synthetic").exists():
        z.extractall(archive.parent)
tree("dtd", archive.parent/"DTD-Synthetic")
tree("headct", BASE/"HeadCT_anomaly_detection", classification=True)
tree("isic", BASE/"ISBI-ISIC", special_masks="ISBI2016_ISIC_Part1_Test_GroundTruth")
for name, root in [("brainmri", BASE/"Brain MRI/brain_tumor_dataset"), ("br35h", BASE/"Br35H")]:
    rows = [record(root, p, label == "yes") for label in ("no", "yes") for p in images(root/label)]
    save(name, root, {"train": {}, "test": {"brain": rows}})
for name, root in [("clinicdb", BASE/"CVC-Kvasir/CVC-ClinicDB"), ("colondb", BASE/"CVC-Kvasir/CVC-ColonDB"), ("kvasir", BASE/"CVC-Kvasir/Kvasir"), ("endo", BASE/"EndoTect_2020_Segmentation_Test_Dataset/EndoTect_2020_Segmentation_Test_Dataset")]:
    masks = {p.stem: p for p in images(root/"masks")}
    rows = [record(root, p, True, masks[p.stem]) for p in images(root/"images")]
    save(name, root, {"train": {}, "test": {"colon": rows}})
Path("runs/preparation/datasets.json").write_text(json.dumps(summary, indent=2))
