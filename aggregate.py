"""
Summarise every finished run in results/ (no GPU needed).

    python aggregate.py                 # summary table (mean +- std over seeds) -> results/summary.csv
    python aggregate.py --tune          # best Nesterov outer lr per (task, split, agg) on seed 0
    python aggregate.py --paired        # Nesterov vs plain averaging, paired by seed

Communication is reported in MB per worker assuming bf16 transport (2 bytes per value).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

KEYS = ["task", "model_name", "mode", "agg", "split", "alpha", "k", "H", "rank",
        "outer_lr", "outer_momentum", "inner_state", "down_rank", "lr", "steps"]


def load(out_dir="results"):
    rows = []
    for p in glob.glob(os.path.join(out_dir, "*.json")):
        with open(p) as f:
            s = json.load(f)
        c = s["config"]
        r = {k: c.get(k) for k in KEYS + ["seed"]}
        if r["split"] != "dirichlet":
            r["alpha"] = None
        if r["mode"] != "diloco":
            r.update(k=None, H=None, outer_lr=None, outer_momentum=None, split="all")
        r.update(val_loss=s["final_val_loss"], val_acc=s["final_val_acc"], diverged=s["diverged"],
                 up_mb=s["comm_up"] * 2 / 1e6, down_mb=s["comm_down"] * 2 / 1e6,
                 minutes=s["minutes"], name=s["name"])
        rows.append(r)
    return pd.DataFrame(rows)


def summary(df):
    g = df.groupby([k for k in KEYS], dropna=False)
    out = g.agg(n=("seed", "count"), loss_mean=("val_loss", "mean"), loss_std=("val_loss", "std"),
                acc_mean=("val_acc", "mean"), acc_std=("val_acc", "std"),
                up_mb=("up_mb", "mean"), down_mb=("down_mb", "mean"),
                diverged=("diverged", "sum")).reset_index()
    return out.sort_values(["task", "split", "agg", "outer_momentum", "outer_lr"])


def tune(df):
    d = df[(df["mode"] == "diloco") & (df["seed"] == 0) & (df["outer_momentum"] > 0)]
    idx = d.groupby(["task", "split", "agg"])["val_loss"].idxmin()
    return d.loc[idx, ["task", "split", "agg", "outer_lr", "val_loss"]]


def paired(df):
    """For each (task, split, agg): mean and 95% t-interval of loss(Nesterov) - loss(avg) over seeds."""
    from scipy import stats
    d = df[df["mode"] == "diloco"]
    avg = d[d["outer_momentum"] == 0]
    nes = d[d["outer_momentum"] > 0]
    res = []
    for key, a in avg.groupby(["task", "split", "agg"]):
        n = nes[(nes["task"] == key[0]) & (nes["split"] == key[1]) & (nes["agg"] == key[2])]
        if n.empty:
            continue
        # the Nesterov lr with the most seeds (the one carried into 1b_core_seeds)
        best_lr = n.groupby("outer_lr")["seed"].count().idxmax()
        n = n[n["outer_lr"] == best_lr]
        m = a.merge(n, on="seed", suffixes=("_avg", "_nes"))
        diff = (m["val_loss_nes"] - m["val_loss_avg"]).to_numpy()
        if len(diff) >= 2:
            ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
        else:
            ci = (np.nan, np.nan)
        res.append(dict(task=key[0], split=key[1], agg=key[2], nest_lr=best_lr, seeds=len(diff),
                        diff_mean=diff.mean(), ci_lo=ci[0], ci_hi=ci[1]))
    return pd.DataFrame(res)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--paired", action="store_true")
    a = p.parse_args()
    df = load(a.out_dir)
    if df.empty:
        raise SystemExit("no finished runs found")
    pd.set_option("display.width", 200, "display.max_rows", 500)
    if a.tune:
        print(tune(df).to_string(index=False))
    elif a.paired:
        print("negative diff = Nesterov better\n")
        print(paired(df).to_string(index=False))
    else:
        s = summary(df)
        s.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
        print(s.to_string(index=False))
