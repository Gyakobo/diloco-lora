"""
Run every config in one or more sweep files, one run at a time (same format as the
diloco-replication repo).

    python sweep.py configs/0_pilot.yaml
    python sweep.py configs/1_core.yaml --dry_run      # list runs + time estimate only

A sweep file has a `base` (shared settings) plus `sweep` (every combination of the listed
values becomes a run) and/or `runs` (explicit list, each merged on top of `base`).
`grids` is a list of bundle-lists: one bundle (a dict of settings that vary TOGETHER, e.g.
task + its batch size, or outer_lr + momentum) is taken from each list, all combinations,
and crossed with `sweep`.

Each run is its own `python train.py ...` process, so a crash or OOM doesn't kill the sweep.
Finished runs are skipped: re-running the same command resumes where it stopped.
"""
import argparse
import itertools
import subprocess
import sys
import time

import yaml

# Seconds per worker step (batch 16) for GPT-2 small on the RTX 3070 Ti.
# PLACEHOLDER: update both numbers from the pilot (printed in each run's .json as `minutes`).
SEC_PER_WORKER_STEP = {"dolly": 0.20, "agnews": 0.08}


def expand(path):
    with open(path) as f:
        spec = yaml.safe_load(f)
    base = spec.get("base", {}) or {}
    # `grids`: list of bundle-lists; one bundle is picked from each list (cartesian product)
    grids = spec.get("grids", [])
    bundles = [{k: v for b in combo for k, v in b.items()} for combo in itertools.product(*grids)] \
        if grids else [{}]
    runs = []
    if "sweep" in spec:
        keys = list(spec["sweep"].keys())
        for bundle in bundles:
            for values in itertools.product(*[spec["sweep"][k] for k in keys]):
                runs.append({**base, **bundle, **dict(zip(keys, values))})
    elif grids:
        runs += [{**base, **b} for b in bundles]
    for r in spec.get("runs", []) or []:
        runs.append({**base, **r})
    return runs


def to_args(run):
    return [x for k, v in run.items() for x in (f"--{k}", str(v))]


def est_minutes(run):
    workers = run.get("k", 4) if run.get("mode", "diloco") == "diloco" else 1
    scale = run.get("batch_size", 16) / 16
    if "SmolLM2" in str(run.get("model_name", "")):
        scale *= 3
    return run.get("steps", 1000) * workers * scale * SEC_PER_WORKER_STEP[run.get("task", "dolly")] / 60


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    runs = [r for c in a.configs for r in expand(c)]
    total = sum(est_minutes(r) for r in runs)
    print(f"{len(runs)} runs, about {total / 60:.1f} GPU-hours if none are already done\n")
    for i, r in enumerate(runs, 1):
        print(f"  {i:3d}. ~{est_minutes(r):4.0f} min  {r}")
    if a.dry_run:
        return

    failed, t0 = [], time.time()
    for i, r in enumerate(runs, 1):
        print(f"\n===== [{i}/{len(runs)}] {r}  (sweep time so far {(time.time() - t0) / 3600:.1f} h)",
              flush=True)
        if subprocess.call([sys.executable, "train.py", *to_args(r)]) != 0:
            print("!!!!! run failed, continuing", flush=True)
            failed.append(r)
    print(f"\nsweep finished in {(time.time() - t0) / 3600:.1f} h, {len(failed)} failed")
    for r in failed:
        print("  failed:", r)


if __name__ == "__main__":
    main()
