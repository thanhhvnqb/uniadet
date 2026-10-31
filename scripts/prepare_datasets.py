"""Prepare available paper benchmarks using AnomalyCLIP/AdaptCLIP metadata schema."""
import argparse
import csv
import json
from pathlib import Path
import zipfile
import re

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

def prepare_realiad():
    root = BASE / "Real-IAD"
    split_dir = root / "realiad_jsons"
    archive = root / "realiad_jsons.zip"
    if not split_dir.exists() and archive.is_file():
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
    split_files = sorted(split_dir.glob("*.json"))
    if len(split_files) != 30:
        raise ValueError(f"Expected 30 official category JSONs in {split_dir}; download and extract realiad_jsons.zip first")
    meta = {"train": {}, "test": {}}
    for split_file in split_files:
        category = split_file.stem
        source = json.loads(split_file.read_text())
        for split in meta:
            rows = []
            for item in source[split]:
                image = root / category / item["image_path"]
                anomaly = item["anomaly_class"] != source["meta"]["normal_class"]
                mask = root / category / item["mask_path"] if item["mask_path"] else None
                if anomaly and mask is None:
                    raise ValueError(f"Missing anomaly mask: {image}")
                row = record(root, image, anomaly, mask)
                view = re.search(r"_(C[1-5])_", image.name)
                if view is None:
                    raise ValueError(f"Missing camera ID: {image}")
                row["view_id"] = view.group(1)
                rows.append(row)
            meta[split][category] = rows
    save("realiad", root, meta)


def prepare_visa_mvtec():
    visa = BASE / "VisA"
    meta = {"train": {}, "test": {}}
    with (visa / "split_csv/1cls.csv").open() as f:
        for r in csv.DictReader(f):
            anomaly = r["label"] == "anomaly"
            meta[r["split"]].setdefault(r["object"], []).append(record(visa, visa/r["image"], anomaly, visa/r["mask"] if anomaly else None))
    save("visa", visa, meta)
    tree("mvtec", BASE/"MVTecAD")


def prepare_existing():
    prepare_visa_mvtec()
    tree("btad", BASE/"btad/BTech_Dataset_transformed", normal="ok")
    tree("ksdd", BASE/"KSDD/SDD")
    archive = BASE/"DTD/Zero-shot texture AD/DTD-Synthetic.zip"
    with zipfile.ZipFile(archive) as z:
        if not (archive.parent/"DTD-Synthetic").exists():
            z.extractall(archive.parent)
    tree("dtd", archive.parent/"DTD-Synthetic")
    tree("headct", BASE/"HeadCT_anomaly_detection", classification=True)
    tree("isic", BASE/"ISBI-ISIC", special_masks="ISBI2016_ISIC_Part1_Test_GroundTruth")
    for name, root in [("brainmri", BASE/"BrainMRI/brain_tumor_dataset"), ("br35h", BASE/"Br35H")]:
        rows = [record(root, p, label == "yes") for label in ("no", "yes") for p in images(root/label)]
        save(name, root, {"train": {}, "test": {"brain": rows}})
    for name, root in [("clinicdb", BASE/"CVC-Kvasir/CVC-ClinicDB"), ("colondb", BASE/"CVC-Kvasir/CVC-ColonDB"), ("kvasir", BASE/"CVC-Kvasir/Kvasir"), ("endo", BASE/"EndoTect_2020_Segmentation_Test_Dataset")]:
        masks = {p.stem: p for p in images(root/"masks")}
        rows = [record(root, p, True, masks[p.stem]) for p in images(root/"images")]
        save(name, root, {"train": {}, "test": {"colon": rows}})


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=("all", "existing", "realiad", "visa_mvtec"), default="all",
                        help="all benchmarks, existing benchmarks excluding Real-IAD, Real-IAD only, or VisA/MVTec only")
    parser.add_argument("--base", type=Path, default=BASE, help="dataset parent directory")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "runs/preparation")
    args = parser.parse_args()
    BASE = args.base.resolve()
    if args.datasets in ("all", "existing"):
        prepare_existing()
    if args.datasets == "visa_mvtec":
        prepare_visa_mvtec()
    if args.datasets in ("all", "realiad"):
        prepare_realiad()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "datasets.json"
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    previous.update(summary)
    summary_path.write_text(json.dumps(previous, indent=2))


if __name__ == "__main__":
    main()
