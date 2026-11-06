"""GPU preflight, isolated workers and seed aggregation for baseline reproduction."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from uniadet.evaluation_metrics import METRIC_COLUMNS
ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def command_output(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30).strip()

def probe_gpu(gpu, minimum):
    info = command_output(["nvidia-smi", "-i", gpu, "--query-gpu=index,uuid,name,memory.free,utilization.gpu,driver_version", "--format=csv,noheader,nounits"])
    fields = [v.strip() for v in info.split(",")]
    if len(fields) != 6 or int(fields[3]) < minimum or int(fields[4]) > 5:
        raise RuntimeError(f"GPU {gpu} unavailable or below {minimum} MiB free: {info}")
    processes = command_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if processes:
        raise RuntimeError(f"GPU {gpu} has existing compute processes: {processes}")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    probe = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); "
                           "x=torch.ones(1,device='cuda:0'); assert x.item()==1; "
                           "print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"],
                           env=environment, capture_output=True, text=True, timeout=60)
    if probe.returncode:
        raise RuntimeError(f"GPU {gpu} failed CUDA probe: {probe.stderr}")
    return dict(physical_gpu=gpu, nvidia_smi=info, cuda_probe=probe.stdout.strip())

def run_job(record):
    job = json.loads(Path(record).read_text())
    if hashlib.sha256(Path(job["snapshot"]).read_bytes()).hexdigest() != job["snapshot_sha256"]:
        raise RuntimeError("Config snapshot changed")
    job.update(pid=os.getpid(), started=now(), state="running", command_statuses=[],
               worker_environment={key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE")})
    write_json(record, job)
    started = time.monotonic()
    prefix = f'GPU={job["gpu"]} {job["variant"]} seed={job["seed"]} source={job.get("training_dataset", "unknown")}'
    progress_connected = True
    def progress(message):
        nonlocal progress_connected
        if progress_connected:
            try:
                print(f"[{now()}] [{prefix}] {message}", flush=True)
            except BrokenPipeError:
                # A detached terminal/tee must not abort a healthy child. Its
                # output is still drained to the stage log below. Redirect the
                # closed stream too, avoiding another failure at interpreter exit.
                progress_connected = False
                sys.stdout = open(os.devnull, "w")
    try:
        # No silent fallback to CPU; parent checked each physical GPU via isolated CUDA.
        if os.environ.get("CUDA_VISIBLE_DEVICES") != job["gpu"]:
            raise RuntimeError("Worker CUDA mapping is incorrect")
        progress("PREFLIGHT: checking GPU/CUDA")
        job["launch_gpu_probe"] = probe_gpu(job["gpu"], job["minimum_free_mib"])
        write_json(record, job)
        import torch
        for index, command in enumerate(job["commands"]):
            if index == 0 and job.get("reuse_training"):
                checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=True)
                differences = {key: (value, checkpoint["config"].get(key))
                               for key, value in job["training"].items() if checkpoint["config"].get(key) != value}
                if checkpoint["epoch"] != 15 or differences:
                    raise RuntimeError(f"Reused checkpoint epoch/config mismatch: {differences}")
                del checkpoint
                progress(f"REUSE TRAINING: verified {job['checkpoint']}; training skipped")
                job["command_statuses"].append(dict(command=command, returncode=0, reused=True, finished=now()))
                write_json(record, job)
                continue
            phase = "TRAIN" if index == 0 else "TEST"
            job["stage"] = phase.lower()
            write_json(record, job)
            progress(f"{phase} START: {shlex.join(command)}")
            if hashlib.sha256(Path(job["snapshot"]).read_bytes()).hexdigest() != job["snapshot_sha256"]:
                raise RuntimeError("Config snapshot changed before command execution")
            stage = Path(job["training"]["output"]) if index == 0 else Path(job["metrics"]).parent
            # Training requires an empty output directory. Capture outside it,
            # then place the log beside the outputs even when the command fails.
            source_name = job.get("training_dataset", "legacy")
            stage_log = Path(record).parent / f"{phase.lower()}_{source_name}.log"
            try:
                with stage_log.open("w") as log:
                    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            progress(f"{phase}: {line.rstrip()}")
                        returncode = process.wait()
            finally:
                stage.mkdir(parents=True, exist_ok=True)
                stage_log.replace(stage / ("train.log" if index == 0 else "test.log"))
            job["command_statuses"].append(dict(command=command, returncode=returncode, finished=now()))
            write_json(record, job)
            progress(f"{phase} END: exit={returncode}; log={stage / ('train.log' if index == 0 else 'test.log')}")
            if returncode:
                raise RuntimeError(f"Command exited {returncode}: {shlex.join(command)}")
            if index == 0:
                progress(f"VERIFY CHECKPOINT: {job['checkpoint']}")
                checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=True)
                config = checkpoint["config"]
                differences = {key: (value, config.get(key)) for key, value in job["training"].items()
                               if config.get(key) != value}
                if checkpoint["epoch"] != 15 or differences:
                    raise RuntimeError(f"Checkpoint epoch/config mismatch: {differences}")
                del checkpoint
        for evaluation in job.get("evaluations", [job]):
            saved = json.loads(Path(evaluation["evaluation_config"]).read_text())
            if (saved["checkpoint_epoch"] != 15 or saved["evaluation"]["seed"] != job["seed"] or
                    saved["evaluation"]["shots"] != job["shots"] or
                    Path(saved["evaluation"]["checkpoint"]).resolve() != Path(job["checkpoint"]).resolve()):
                raise RuntimeError("Evaluation protocol/checkpoint mismatch")
            if not Path(evaluation["metrics"]).is_file():
                raise RuntimeError("Evaluation has not produced metrics")
            test_log = Path(job["metrics"]).parent / "test.log"
            evaluation_log = Path(evaluation["metrics"]).parent / "test.log"
            if test_log != evaluation_log:
                evaluation_log.write_bytes(test_log.read_bytes())
        job.update(state="complete", returncode=0)
        progress(f"COMPLETE: elapsed={time.monotonic()-started:.1f}s")
    except Exception as error:
        job.update(state="failed", error=str(error), returncode=1)
        progress(f"FAILED: {error}")
        raise
    finally:
        job.update(finished=now(), seconds=time.monotonic()-started)
        write_json(record, job)

def aggregate(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    jobs = [json.loads(Path(p).read_text()) for p in plan["jobs"]]
    # A terminated process may not reach its Python finally block.
    exits = Path(plan_path).parent / "exits.tsv"
    if exits.exists():
        for line in exits.read_text().splitlines():
            record, code = line.rsplit("\t", 1)
            job = next(j for j in jobs if j["record"] == record)
            if int(code) != 0 or job["state"] != "complete":
                job.update(state="failed", returncode=int(code) or 1)
                write_json(record, job)
    # Expand evaluations without combining datasets or auxiliary checkpoints.
    evaluated = []
    for job in jobs:
        evaluations = job.get("evaluations", [dict(metrics=job["metrics"], dataset="Real-IAD",
                                                   training_dataset="VisA")])
        for evaluation in evaluations:
            item = dict(job, **{k: v for k, v in evaluation.items() if k != "state"})
            item["id"] = f'{job["id"]}:{evaluation["dataset"]}'
            item["state"] = evaluation.get("state", job["state"])
            item["macro"] = {}
            item["categories"] = {}
            if item["state"] == "complete":
                report = json.loads(Path(item["metrics"]).read_text())
                item["categories"] = {k: v for k, v in report.items() if k not in ("average", "macro_mean")}
                for task, metric, column in METRIC_COLUMNS:
                    values = [(v.get(task) or {}).get(metric) for v in item["categories"].values()]
                    values = [v for v in values if v is not None]
                    mean = statistics.mean(values) if values else None
                    item["macro"][column] = (mean, len(values))
                    alias = (report.get("average", report.get("macro_mean"))[task] or {}).get(metric)
                    if (mean is None) != (alias is None) or mean is not None and abs(mean-alias) > 1e-9:
                        raise ValueError(f"Raw category macro disagrees with saved aggregate: {item['id']} {column}")
            evaluated.append(item)
    for manifest_path in plan["manifests"]:
        manifest = json.loads(Path(manifest_path).read_text())
        selected_jobs = [j for j in jobs if j["record"] in manifest["jobs"]]
        selected = [j for j in evaluated if j["record"] in manifest["jobs"]]
        directory = Path(manifest_path).parent
        rows, summaries = [], []
        for dataset in sorted({j["dataset"] for j in selected}):
            dataset_jobs = [j for j in selected if j["dataset"] == dataset]
            for job in dataset_jobs:
                common = dict(variant=job["variant"], regime=job["regime"], shots=job["shots"], seed=job["seed"],
                    dataset=dataset, training_dataset=job["training_dataset"], checkpoint_epoch=15,
                    status=job["state"], source=job["metrics"])
                for category, values in job["categories"].items():
                    rows.append(dict(common, category=category, **{c: 100*(values.get(t) or {})[m]
                        if (values.get(t) or {}).get(m) is not None else None for t, m, c in METRIC_COLUMNS}))
                rows.append(dict(common, category="macro_mean", **{c: 100*job["macro"][c][0]
                    if job["macro"].get(c, (None, 0))[0] is not None else None for _, _, c in METRIC_COLUMNS}))
            for _, _, column in METRIC_COLUMNS:
                values, deltas, counts = [], [], {}
                for job in dataset_jobs:
                    value, count = job["macro"].get(column, (None, 0))
                    counts[job["seed"]] = count
                    if value is None:
                        continue
                    values.append(100*value)
                    baseline = next((j for j in evaluated if j["variant"] == "baseline" and
                        j["seed"] == job["seed"] and j["regime"] == job["regime"] and
                        j["shots"] == job["shots"] and j["dataset"] == dataset and
                        j["training_dataset"] == job["training_dataset"]), None)
                    base = baseline["macro"].get(column, (None, 0))[0] if baseline else None
                    if base is not None:
                        deltas.append(100*(value-base))
                summaries.append(dict(variant=manifest["variant"], regime=manifest["regime"], shots=manifest["shots"],
                    dataset=dataset, training_dataset=dataset_jobs[0]["training_dataset"], metric=column,
                    completed_seeds=sum(j["state"] == "complete" for j in dataset_jobs), valid_seeds=len(values),
                    expected_seeds=len(dataset_jobs), valid_categories_per_seed=json.dumps(counts),
                    mean_percent=statistics.mean(values) if values else None,
                    sample_std_percent=statistics.stdev(values) if len(values)>1 else None,
                    paired_seeds=len(deltas), paired_delta_pp=statistics.mean(deltas) if deltas else None,
                    seeds_improved=sum(d>0 for d in deltas) if deltas else None))
        for name, data in (("results.csv", rows), ("summary.csv", summaries)):
            with (directory / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        manifest.update(state="complete" if all(j["state"] == "complete" for j in selected_jobs) else "incomplete",
            finished=now(), outcomes=[{k: j.get(k) for k in
                ("id", "gpu", "seed", "training_dataset", "pid", "state", "returncode", "seconds", "error")} for j in selected_jobs])
        write_json(manifest_path, manifest)
        primary = [r for r in summaries if r["metric"] in ("P-AUPR", "I-AUROC")]
        (directory / "discussion.md").write_text(
            f'# {manifest["variant"]}: {manifest["regime"]}, shots={manifest["shots"]}\n\n'
            f'Status: {manifest["state"]}. Frozen backbone, labeled auxiliary test split, epoch 15. '
            'VisA evaluation uses MVTecAD training; all other targets use VisA training. '
            'Each dataset is aggregated independently across seeds 10/20/30.\n\n'
            'Primary metrics (percent; paired deltas in percentage points):\n\n```json\n' + json.dumps(primary, indent=2) + '\n```\n\n'
            'See results.csv for per-category changes, summary.csv for sample standard deviations (ddof=1), '
            'and manifest.json for job outcomes/configs/logs. No statistical significance is established. '
            'Detailed interpretation requires review of completed results.\n')
        lessons = directory / "lessons.md"
        if not lessons.exists():
            lessons.write_text(
                '# Lessons\n\nProtocol: frozen backbone, labeled auxiliary test split, epoch 15, seeds 10/20/30. '
                'Keep target datasets separate and match auxiliary source, shots and seeds in comparisons. '
                'Evidence: results.csv, summary.csv, manifest.json. No scientific conclusion is recorded automatically.\n')
    return 0 if all(j["state"] == "complete" for j in jobs) else 1
