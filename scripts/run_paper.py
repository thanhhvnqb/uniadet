"""Run CLIP paper protocol on the locally available benchmarks; restart skips completed jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "4")
PYTHON = str(ROOT / ".venv/bin/python")
OUT = ROOT / "runs/paper_clip"
OUT.mkdir(parents=True, exist_ok=True)
data = json.loads((ROOT / "runs/preparation/datasets.json").read_text())
status = {"started": datetime.now().isoformat(), "jobs": []}

def run(name, command, completed):
    if completed.exists():
        print("SKIP", name, flush=True)
        return
    print("START", name, datetime.now().isoformat(), flush=True)
    job = dict(name=name, command=command, started=datetime.now().isoformat(), state="running")
    status["jobs"].append(job)
    (OUT/"status.json").write_text(json.dumps(status, indent=2))
    with (OUT/(name+".log")).open("a") as log:
        result = subprocess.run([PYTHON, "-u", *command], stdout=log, stderr=subprocess.STDOUT)
    job.update(state="complete" if result.returncode == 0 else "failed", finished=datetime.now().isoformat(), returncode=result.returncode)
    (OUT/"status.json").write_text(json.dumps(status, indent=2))
    if result.returncode:
        raise SystemExit(f"Failed {name}; see {OUT/(name+'.log')}")
    print("DONE", name, datetime.now().isoformat(), flush=True)

for auxiliary in ("visa", "mvtec"):
    output = OUT/("train_"+auxiliary)
    run("train_"+auxiliary, ["train.py", "--data", data[auxiliary]["root"], "--backbone", "clip", "--epochs", "15", "--lr", "0.001", "--batch-size", "4", "--workers", "4", "--seed", "10", "--device", "cuda", "--output", str(output)], output/"epoch_15.pt")
    checkpoint = output/"epoch_15.pt"
    targets = ["visa"] if auxiliary == "mvtec" else [k for k in data if k != "visa"]
    for target in targets:
        shots = (0, 1, 2, 4) if data[target]["train"]["normal"] else (0,)
        for shot in shots:
            name = f"eval_{target}_{shot}shot_seed10"
            output = OUT/name
            run(name, ["test.py", "--data", data[target]["root"], "--checkpoint", str(checkpoint), "--shots", str(shot), "--seed", "10", "--device", "cuda", "--workers", "4", "--output", str(output)], output/"metrics.json")
status["finished"] = datetime.now().isoformat()
(OUT/"status.json").write_text(json.dumps(status, indent=2))
