"""
DiLoCo for LoRA fine-tuning, with k workers simulated one after another on a single GPU.

The question this code is built to answer: DiLoCo's outer step (SGD with Nesterov momentum on
the averaged pseudo-gradient) is what makes it work for pre-training. Does that outer step
still help when the workers only train LoRA adapters, and does the answer depend on whether
averaging the adapters is EXACT (equals the average of the workers' weight updates) or not?

Aggregation methods (--agg):
  fedit   average A and B separately (FedIT, Zhang et al. 2024). INEXACT:
          mean(B_i) mean(A_i) != mean(B_i A_i).
  ffa     A is frozen at its random init; only B is trained and averaged
          (FFA-LoRA, Sun et al. 2024). EXACT, because B A is linear in B.
  fedex   average A and B, then fold the residual mean(B_i A_i) - mean(B) mean(A) into the
          frozen base weight (FedEx-LoRA, Singhal et al. 2025). EXACT, but the residual is dense.
  merge   every round the workers start fresh adapters; the server averages the dense updates
          s B_i A_i exactly and applies the outer optimizer to the dense weight, then merges it.
          EXACT. Uplink is low-rank; downlink is dense (or rank-truncated with --down_rank).
  full    no LoRA: ordinary DiLoCo full fine-tuning (communication reference point).

The outer optimizer is SGD(outer_lr, outer_momentum, nesterov). outer_lr=1, momentum=0 is plain
averaging (FedAvg); outer_lr=0.7, momentum=0.9, nesterov is DiLoCo.

Modes:
  diloco   k workers, sync every H inner steps
  sync     one model, batch k*B, i.e. data-parallel training that communicates every step
  local    one worker alone, batch B (no collaboration)

    python train.py --task dolly --agg fedit --k 4 --H 50 --outer_lr 0.7
    python train.py --task dolly --mode sync --batch_size 64

Writes results/<name>.csv (one row per evaluation) and results/<name>.json (config + summary).
A run whose .json exists is skipped.
"""
import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields, replace

import numpy as np
import torch
import torch.nn.functional as F

import lora as L
from data import Sampler, collate, load_task, partition


@dataclass
class TrainConfig:
    mode: str = "diloco"          # diloco | sync | local
    agg: str = "fedit"            # fedit | ffa | fedex | merge | full
    # --- DiLoCo ---
    k: int = 4
    H: int = 50
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    nesterov: bool = True
    inner_state: str = "keep"     # keep | reset: keep each worker's AdamW state across rounds?
    down_rank: int = 0            # merge only: truncate the broadcast update to this rank (0 = dense)
    # --- data ---
    task: str = "dolly"           # dolly | agnews
    split: str = "iid"            # iid | group | dirichlet
    alpha: float = 0.5            # Dirichlet concentration (split=dirichlet)
    split_seed: int = 0           # fixed: every method sees the same partition
    agnews_train: int = 20000
    block_size: int = 512
    # --- model / LoRA ---
    model_name: str = "gpt2"
    rank: int = 16
    lora_alpha: float = 32.0
    targets: str = ""             # regex; empty = attention + MLP projections of the family
    model_dropout: float = 0.0    # dropout inside the base model (0 = deterministic comparisons)
    # --- inner optimizer (AdamW) ---
    steps: int = 1000             # inner steps per worker (= total steps for sync/local)
    batch_size: int = 16          # sequences per worker step
    micro_batch: int = 8          # sequences per forward pass (memory only)
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup: int = 50
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # --- bookkeeping ---
    eval_interval: int = 100      # inner steps
    eval_examples: int = 500      # validation examples used for periodic evaluation
    seed: int = 0
    dtype: str = "bf16"           # bf16 | fp32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: str = "data"
    out_dir: str = "results"
    name: str = ""
    force: bool = False

    def run_name(self):
        if self.name:
            return self.name
        m = self.model_name.split("/")[-1]
        ag = self.agg if self.agg == "full" else f"{self.agg}_r{self.rank}"
        sp = self.split if self.split != "dirichlet" else f"dir{self.alpha}"
        if self.mode == "diloco":
            n = (f"{self.task}_{m}_{ag}_k{self.k}_H{self.H}_olr{self.outer_lr}_m{self.outer_momentum}"
                 f"{'_nest' if self.nesterov and self.outer_momentum > 0 else ''}_{sp}"
                 f"{'_reset' if self.inner_state == 'reset' else ''}"
                 f"{f'_dr{self.down_rank}' if self.down_rank else ''}")
        else:
            n = f"{self.task}_{m}_{ag}_{self.mode}_bs{self.batch_size}_{sp}"
        return f"{n}_lr{self.lr}_steps{self.steps}_s{self.seed}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def lr_at(step, cfg):
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    lo = cfg.lr * cfg.min_lr_ratio
    return lo + 0.5 * (cfg.lr - lo) * (1 + math.cos(math.pi * min(1.0, progress)))


def autocast(cfg):
    dev = cfg.device.split(":")[0]
    if cfg.dtype == "bf16":
        return torch.autocast(device_type=dev, dtype=torch.bfloat16)
    return torch.autocast(device_type=dev, enabled=False)


def build_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32)
    return model.to(cfg.device), tok


def token_loss_sum(model, x, m, y, cfg):
    """Sum of next-token losses over the labelled tokens, and their count."""
    with autocast(cfg):
        logits = model(input_ids=x, attention_mask=m).logits
    logits = logits[:, :-1].float()
    tgt = y[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                           ignore_index=-100, reduction="sum")
    return loss, (tgt != -100).sum()


def make_inner_opt(params, cfg):
    return torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999),
                             weight_decay=cfg.weight_decay,
                             fused=cfg.device.startswith("cuda"))


def inner_step(model, params, opt, sampler, step, cfg, pad_id):
    """One AdamW step on one batch, loss averaged over labelled tokens of the WHOLE batch."""
    for g in opt.param_groups:
        g["lr"] = lr_at(step, cfg)
    batch = sampler.next()
    n_tok = sum(len(ids) - plen for ids, plen in batch)   # labelled tokens (incl. EOS)
    total = 0.0
    for i in range(0, len(batch), cfg.micro_batch):
        x, m, y = collate(batch[i: i + cfg.micro_batch], pad_id, cfg.device)
        s, _ = token_loss_sum(model, x, m, y, cfg)
        (s / n_tok).backward()
        total += s.item()
    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return total / n_tok


@torch.no_grad()
def evaluate(model, val, cfg, pad_id, label_ids=None, val_labels=None, n=None):
    """Validation loss (per labelled token) and, for classification, accuracy."""
    model.eval()
    val = val[: n or len(val)]
    tot, cnt = 0.0, 0
    for i in range(0, len(val), cfg.micro_batch):
        x, m, y = collate(val[i: i + cfg.micro_batch], pad_id, cfg.device)
        s, c = token_loss_sum(model, x, m, y, cfg)
        tot, cnt = tot + s.item(), cnt + c.item()
    acc = None
    if label_ids is not None:
        acc = classify_accuracy(model, val, val_labels[: len(val)], label_ids, cfg, pad_id)
    model.train()
    return tot / cnt, acc


@torch.no_grad()
def classify_accuracy(model, val, labels, label_ids, cfg, pad_id):
    """Pick the label whose tokens are most likely after the prompt (sum of log-probs)."""
    correct = 0
    for i in range(0, len(val), max(1, cfg.micro_batch // len(label_ids))):
        chunk = val[i: i + max(1, cfg.micro_batch // len(label_ids))]
        seqs = [(ids[:plen] + lab, plen) for ids, plen in chunk for lab in label_ids]
        x, m, y = collate(seqs, pad_id, cfg.device)
        with autocast(cfg):
            logits = model(input_ids=x, attention_mask=m).logits
        lp = torch.log_softmax(logits[:, :-1].float(), -1)
        tgt = y[:, 1:]
        mask = tgt != -100
        tok_lp = lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
        score = tok_lp.sum(-1).view(len(chunk), len(label_ids))
        pred = score.argmax(-1).cpu().numpy()
        correct += int((pred == labels[i: i + len(chunk)]).sum())
    return correct / len(val)


def sgd_step(g, buf, lr, momentum, nesterov):
    """The update torch.optim.SGD would apply for gradient g (buf is updated in place)."""
    if momentum > 0:
        buf.mul_(momentum).add_(g)
        g = g + momentum * buf if nesterov else buf.clone()
    return -lr * g


def svd_truncate(M, r):
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


# ---------------------------------------------------------------------------
# setup shared by all modes
# ---------------------------------------------------------------------------
def setup(cfg, model=None, task=None):
    torch.manual_seed(cfg.seed)
    if model is None:
        model, tok = build_model(cfg)
        pad_id = tok.pad_token_id
    else:                                   # tests pass a tiny model + task directly
        pad_id = task["pad_id"]
    if task is None:
        task = load_task(cfg, tok)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = cfg.model_dropout
    model.train()
    layers = []
    if cfg.agg != "full":
        layers = L.wrap_lora(model, cfg.rank, cfg.lora_alpha, cfg.targets or None, seed=cfg.seed)
        if cfg.agg == "ffa":
            for _, m in layers:
                m.lora_A.requires_grad_(False)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return model, task, pad_id, layers, trainable


def comm_params(cfg, model, layers):
    """Tensors that are synchronised in parameter space by the outer optimizer."""
    if cfg.agg in ("fedit", "fedex"):
        return L.adapter_params(layers, "AB")
    if cfg.agg == "ffa":
        return L.adapter_params(layers, "B")
    if cfg.agg == "full":
        return [p for p in model.parameters() if p.requires_grad]
    return []   # merge: adapters are re-initialised every round; the dense weights are synced


def comm_cost(cfg, layers, model):
    """(uplink params per worker, downlink params) for ONE communication round."""
    c = L.count(layers) if layers else None
    if cfg.agg == "full":
        n = sum(p.numel() for p in model.parameters())
        return n, n
    if cfg.agg == "fedit":
        return c["A"] + c["B"], c["A"] + c["B"]
    if cfg.agg == "ffa":
        return c["B"], c["B"]
    if cfg.agg == "fedex":
        return c["A"] + c["B"], c["A"] + c["B"] + c["dense"]
    if cfg.agg == "merge":
        if cfg.down_rank:
            down = sum(cfg.down_rank * (m.in_features + m.out_features) for _, m in layers)
        else:
            down = c["dense"]
        return c["A"] + c["B"], down
    raise ValueError(cfg.agg)


# ---------------------------------------------------------------------------
# DiLoCo
# ---------------------------------------------------------------------------
def run_diloco(cfg, model, task, pad_id, layers, trainable, log):
    train = task["train"]
    parts = partition(task["train_groups"], cfg.k, cfg.split, cfg.alpha, cfg.split_seed)
    samplers = [Sampler(train, parts[i], cfg.batch_size, seed=cfg.seed * 1000 + i)
                for i in range(cfg.k)]

    # ONE physical model for all workers (they all restart from the global state every round).
    # Workers differ only in their AdamW states, so we keep k optimizers on the same tensors.
    inner_opts = [make_inner_opt(trainable, cfg) for _ in range(cfg.k)]

    cps = comm_params(cfg, model, layers)
    gparams = [torch.nn.Parameter(p.detach().clone()) for p in cps]     # global copy
    outer_opt = (torch.optim.SGD(gparams, lr=cfg.outer_lr, momentum=cfg.outer_momentum,
                                 nesterov=cfg.nesterov and cfg.outer_momentum > 0)
                 if gparams else None)
    avg = [torch.zeros_like(p) for p in cps]
    need_dense = cfg.agg in ("fedit", "fedex", "merge")   # fedit: only to measure agg error
    dense_avg = [torch.zeros_like(m.weight) for _, m in layers] if need_dense else []
    # merge: Nesterov buffers and (optional) error feedback for the truncated broadcast
    buf = [torch.zeros_like(m.weight) for _, m in layers] if cfg.agg == "merge" else []
    ef = [torch.zeros_like(m.weight) for _, m in layers] if cfg.agg == "merge" and cfg.down_rank else []
    up, down = comm_cost(cfg, layers, model)

    def load_global():
        with torch.no_grad():
            for p, g in zip(cps, gparams):
                p.copy_(g)
            if cfg.agg == "merge":
                for _, m in layers:
                    m.lora_B.zero_()          # adapters carry nothing between rounds

    assert cfg.steps % cfg.H == 0, "steps must be divisible by H"
    T = cfg.steps // cfg.H
    steps_done, losses = 0, []
    load_global()
    log(steps_done, 0, None, dict())

    for t in range(T):
        torch._foreach_zero_(avg) if avg else None
        for d in dense_avg:
            d.zero_()
        if cfg.agg == "merge":
            # fresh adapters, identical across workers; round 0 uses the wrap_lora init
            gen = torch.Generator().manual_seed(cfg.seed if t == 0 else cfg.seed * 1_000_003 + t)
            for _, m in layers:
                m.reset_lora(gen)
            a_init = [m.lora_A.detach().clone() for _, m in layers]
        for i in range(cfg.k):
            with torch.no_grad():
                for p, g in zip(cps, gparams):
                    p.copy_(g)
                if cfg.agg == "merge":
                    for (_, m), a0 in zip(layers, a_init):
                        m.lora_A.copy_(a0)
                        m.lora_B.zero_()
            if cfg.inner_state == "reset":
                inner_opts[i] = make_inner_opt(trainable, cfg)
            for h in range(cfg.H):
                losses.append(inner_step(model, trainable, inner_opts[i], samplers[i],
                                         steps_done + h, cfg, pad_id))
            with torch.no_grad():
                for a, p in zip(avg, cps):
                    a.add_(p, alpha=1.0 / cfg.k)
                for d, (_, m) in zip(dense_avg, layers):
                    d.add_(m.delta(), alpha=1.0 / cfg.k)
        steps_done += cfg.H

        stats = dict()
        with torch.no_grad():
            if cfg.agg == "fedit" or cfg.agg == "fedex":
                # aggregation error of naive factor averaging, relative to the exact mean update
                # agg_err:     ||s mean(B)mean(A) - mean(s B_i A_i)|| / ||mean(s B_i A_i)||
                # agg_err_upd: same error relative to the size of this round's exact update
                num = den = den_upd = 0.0
                for (_, m), d, gA, gB in zip(layers, _factor_means(avg, layers, dense_avg),
                                             gparams[0::2], gparams[1::2]):
                    num += float(d[0].norm() ** 2)
                    den += float(d[1].norm() ** 2)
                    den_upd += float((d[1] - m.scale * (gB @ gA)).norm() ** 2)
                if den > 0:
                    stats["agg_err"] = math.sqrt(num / den)
                if den_upd > 0:
                    stats["agg_err_upd"] = math.sqrt(num / den_upd)
            if cfg.agg == "fedex":
                # residual = exact mean update - update implied by the averaged factors
                residuals = [d - m.scale * (aB @ aA) for (_, m), d, aA, aB in
                             zip(layers, dense_avg, avg[0::2], avg[1::2])]

            if outer_opt is not None:
                for g, a in zip(gparams, avg):
                    g.grad = g - a                     # pseudo-gradient
                outer_opt.step()
                outer_opt.zero_grad(set_to_none=True)

            if cfg.agg == "fedex":
                # the residual is folded into the frozen base weight, scaled like the outer step
                # (with outer_lr = 1 and no momentum this is exactly FedEx-LoRA)
                for (_, m), r in zip(layers, residuals):
                    m.weight.add_(r, alpha=cfg.outer_lr)

            if cfg.agg == "merge":
                step_norm = 0.0
                for j, ((_, m), d) in enumerate(zip(layers, dense_avg)):
                    # pseudo-gradient in weight space = -(mean dense update)
                    upd = sgd_step(-d, buf[j], cfg.outer_lr, cfg.outer_momentum, cfg.nesterov)
                    if cfg.down_rank:
                        upd = upd + ef[j]
                        tr = svd_truncate(upd, cfg.down_rank)
                        ef[j] = upd - tr
                        upd = tr
                    m.weight.add_(upd)
                    step_norm += float(upd.norm() ** 2)
                stats["step_norm"] = math.sqrt(step_norm)

        load_global()
        if steps_done % cfg.eval_interval < cfg.H or t == T - 1:
            stats.update(comm_up=up * (t + 1), comm_down=down * (t + 1))
            if log(steps_done, t + 1, float(np.mean(losses)), stats):
                return
            losses = []


def _factor_means(avg, layers, dense_avg):
    """For fedit/fedex: per layer (error, exact) where error = s*mean(B)mean(A) - mean(sBA)."""
    out = []
    for (_, m), aA, aB, d in zip(layers, avg[0::2], avg[1::2], dense_avg):
        out.append((m.scale * (aB @ aA) - d, d))
    return out


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def run_single(cfg, model, task, pad_id, layers, trainable, log):
    """sync: batch k*B, communicates every step. local: batch B, one worker, no communication."""
    parts = partition(task["train_groups"], 1, "iid", seed=cfg.split_seed)
    sampler = Sampler(task["train"], parts[0], cfg.batch_size, seed=cfg.seed * 1000)
    opt = make_inner_opt(trainable, cfg)
    n_train = sum(p.numel() for p in trainable)
    losses = []
    log(0, 0, None, dict())
    for step in range(cfg.steps):
        losses.append(inner_step(model, trainable, opt, sampler, step, cfg, pad_id))
        if (step + 1) % cfg.eval_interval == 0 or step == cfg.steps - 1:
            stats = dict()
            if cfg.mode == "sync":       # all-reduce of the gradient every step
                stats.update(comm_up=n_train * (step + 1), comm_down=n_train * (step + 1))
            if log(step + 1, step + 1 if cfg.mode == "sync" else 0, float(np.mean(losses)), stats):
                return
            losses = []


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run(cfg, model=None, task=None, write=True, verbose=True):
    torch.backends.cuda.matmul.allow_tf32 = cfg.dtype != "fp32"
    name = cfg.run_name()
    json_path = os.path.join(cfg.out_dir, name + ".json")
    if write and os.path.exists(json_path) and not cfg.force:
        print(f"[skip] {name} already done")
        return None
    model, task, pad_id, layers, trainable = setup(cfg, model, task)
    rows, t0 = [], time.time()
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    workers = cfg.k if cfg.mode == "diloco" else 1

    def log(step, rounds, train_loss, stats):
        vl, acc = evaluate(model, task["val"], cfg, pad_id, task.get("label_ids"),
                           task.get("val_labels"), n=cfg.eval_examples)
        row = dict(step=step, comm_rounds=rounds, examples=step * cfg.batch_size * workers,
                   train_loss=train_loss, val_loss=vl, val_acc=acc,
                   agg_err=stats.get("agg_err"), agg_err_upd=stats.get("agg_err_upd"), step_norm=stats.get("step_norm"),
                   comm_up=stats.get("comm_up", 0), comm_down=stats.get("comm_down", 0),
                   lr=lr_at(max(step - 1, 0), cfg), elapsed_s=round(time.time() - t0, 1))
        rows.append(row)
        if verbose:
            tl = f"{train_loss:.4f}" if train_loss is not None else "  -   "
            a = f" | acc {acc:.3f}" if acc is not None else ""
            e = f" | agg_err {row['agg_err']:.3f}" if row["agg_err"] is not None else ""
            print(f"step {step:5d} | rounds {rounds:4d} | train {tl} | val {vl:.4f}{a}{e} "
                  f"| {(time.time() - t0) / 60:5.1f} min", flush=True)
        return not math.isfinite(vl)

    if verbose:
        n_tr = sum(p.numel() for p in trainable)
        print(f"run: {name}\ntrainable: {n_tr / 1e6:.2f}M | device {cfg.device} | dtype {cfg.dtype}")

    if cfg.mode == "diloco":
        run_diloco(cfg, model, task, pad_id, layers, trainable, log)
    elif cfg.mode in ("sync", "local"):
        run_single(cfg, model, task, pad_id, layers, trainable, log)
    else:
        raise ValueError(cfg.mode)

    # final evaluation on the FULL validation set
    final_loss, final_acc = evaluate(model, task["val"], cfg, pad_id, task.get("label_ids"),
                                     task.get("val_labels"))
    last = rows[-1]
    summary = dict(config=asdict(cfg), name=name, final_val_loss=final_loss, final_val_acc=final_acc,
                   diverged=not math.isfinite(final_loss), steps_done=last["step"],
                   comm_rounds=last["comm_rounds"], comm_up=last["comm_up"], comm_down=last["comm_down"],
                   trainable_params=sum(p.numel() for p in trainable),
                   minutes=round((time.time() - t0) / 60, 2),
                   peak_mem_gb=(round(torch.cuda.max_memory_allocated() / 1e9, 2)
                                if cfg.device.startswith("cuda") else None))
    if verbose:
        print(f"done: final val loss {final_loss:.4f}"
              f"{f' | acc {final_acc:.4f}' if final_acc is not None else ''} | {summary['minutes']} min")
    if write:
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, name + ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
    return rows, summary, model


def parse_args():
    p = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        if f.type in (bool, "bool"):
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"),
                           nargs="?", const=True, default=f.default)
        else:
            t = {"int": int, "float": float, "str": str}.get(f.type, f.type)
            p.add_argument(f"--{f.name}", type=t, default=f.default)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    run(parse_args())
