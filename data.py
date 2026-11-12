"""
Fine-tuning tasks, worker partitions, and batching.

Two tasks:
  dolly   databricks-dolly-15k instruction tuning. Each example has one of 8 categories,
          which gives a NATURAL non-i.i.d. split (the same split FedIT used).
          Metric: validation loss on response tokens.
  agnews  AG News topic classification written as text completion ("... Topic: Sports").
          4 labels, so label-skewed (Dirichlet) splits are easy to control.
          Metrics: validation loss on the label tokens + accuracy (pick the most likely label).

Every example is stored as (token ids, prompt length). The loss is computed on the tokens
after the prompt only.
"""
import os
import random

import numpy as np
import torch

DOLLY_TEMPLATE = "### Instruction:\n{instruction}\n\n{context}### Response:\n"
AGNEWS_LABELS = [" World", " Sports", " Business", " Sci/Tech"]
AGNEWS_TEMPLATE = ("Classify the topic of this news article as World, Sports, Business, or Sci/Tech."
                   "\n\nArticle: {text}\nTopic:")


# ---------------------------------------------------------------------------
# loading + tokenizing (cached)
# ---------------------------------------------------------------------------
def load_task(cfg, tok):
    """Returns dict(train=[(ids, plen)], train_groups=np.array, val=[(ids, plen)], test=[...],
                    val_labels / test_labels = np.array or None, label_ids=list or None).

    val  (1000) is used for monitoring and to pick the early-stopping checkpoint.
    test (1000) is only evaluated at the end, at the final and at the best-val checkpoint.
    """
    tag = cfg.model_name.replace("/", "_")
    path = os.path.join(cfg.data_dir, f"{cfg.task}_{tag}_b{cfg.block_size}_v2.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    if cfg.task == "dolly":
        d = _dolly(cfg, tok)
    elif cfg.task == "agnews":
        d = _agnews(cfg, tok)
    else:
        raise ValueError(cfg.task)
    os.makedirs(cfg.data_dir, exist_ok=True)
    torch.save(d, path)
    return d


def _encode(tok, prompt, target, block):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p) >= block - 8:            # prompt alone (almost) fills the window: drop
        return None
    ids = (p + t)[:block]
    return ids, len(p)


def _dolly(cfg, tok):
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    cats = sorted(set(ds["category"]))
    items, groups = [], []
    for ex in ds:
        ctx = f"### Context:\n{ex['context']}\n\n" if ex["context"].strip() else ""
        prompt = DOLLY_TEMPLATE.format(instruction=ex["instruction"].strip(), context=ctx)
        enc = _encode(tok, prompt, ex["response"].strip(), cfg.block_size)
        if enc:
            items.append(enc)
            groups.append(cats.index(ex["category"]))
    rng = np.random.default_rng(1234)                # fixed split, independent of run seed
    perm = rng.permutation(len(items))
    val_idx, test_idx, tr_idx = perm[:1000], perm[1000:2000], perm[2000:]
    return dict(train=[items[i] for i in tr_idx], train_groups=np.array(groups)[tr_idx],
                val=[items[i] for i in val_idx], test=[items[i] for i in test_idx],
                val_labels=None, test_labels=None, label_ids=None, group_names=cats)


def _agnews(cfg, tok):
    from datasets import load_dataset
    tr = load_dataset("fancyzhx/ag_news", split="train")
    te = load_dataset("fancyzhx/ag_news", split="test")
    rng = np.random.default_rng(1234)
    tr_idx = rng.permutation(len(tr))[:cfg.agnews_train]
    te_perm = rng.permutation(len(te))
    te_idx, test_idx = te_perm[:1000], te_perm[1000:2000]

    # truncate the article so prompt + label always fit in the window
    overhead = len(tok(AGNEWS_TEMPLATE.format(text=""), add_special_tokens=False)["input_ids"])
    budget = cfg.block_size - overhead - 16

    def enc(ex):
        text_ids = tok(ex["text"], add_special_tokens=False)["input_ids"][:budget]
        prompt = AGNEWS_TEMPLATE.format(text=tok.decode(text_ids))
        return _encode(tok, prompt, AGNEWS_LABELS[ex["label"]], cfg.block_size)

    train, groups = [], []
    for i in tr_idx:
        ex = tr[int(i)]
        e = enc(ex)
        if e:
            train.append(e)
            groups.append(ex["label"])
    def held_out(indices):
        xs, ys = [], []
        for i in indices:
            ex = te[int(i)]
            e = enc(ex)
            if e:
                xs.append(e)
                ys.append(ex["label"])
        return xs, np.array(ys)
    val, labels = held_out(te_idx)
    test, test_labels = held_out(test_idx)
    label_ids = [tok(l, add_special_tokens=False)["input_ids"] for l in AGNEWS_LABELS]
    return dict(train=train, train_groups=np.array(groups), val=val, test=test,
                val_labels=labels, test_labels=test_labels, label_ids=label_ids,
                group_names=[l.strip() for l in AGNEWS_LABELS])


# ---------------------------------------------------------------------------
# partitions
# ---------------------------------------------------------------------------
def partition(groups, k, mode, alpha=0.5, seed=0):
    """Split example indices over k workers.

    iid        random equal split
    group      whole groups (categories / labels) per worker, balanced by example count;
               with k >= number of groups some groups are split evenly
    dirichlet  for every group, the share of each worker ~ Dirichlet(alpha) (Hsu et al., 2019)
    The split is fixed by `seed` (we use a constant, so all methods see the same split).
    """
    rng = np.random.default_rng(seed)
    n = len(groups)
    if k == 1 or mode == "iid":
        return [np.sort(p) for p in np.array_split(rng.permutation(n), k)]
    ids = np.unique(groups)
    if mode == "group":
        parts = [[] for _ in range(k)]
        if k <= len(ids):
            # biggest group first, each to the worker with the fewest examples so far
            order = sorted(ids, key=lambda g: -(groups == g).sum())
            for g in order:
                w = int(np.argmin([len(p) for p in parts]))
                parts[w].extend(np.flatnonzero(groups == g))
        else:
            # more workers than groups: worker w gets (a slice of) group w % n_groups
            owners = {g: [w for w in range(k) if w % len(ids) == j] for j, g in enumerate(ids)}
            for g in ids:
                idx = rng.permutation(np.flatnonzero(groups == g))
                for w, chunk in zip(owners[g], np.array_split(idx, len(owners[g]))):
                    parts[w].extend(chunk)
        return [np.sort(np.array(p, dtype=np.int64)) for p in parts]
    if mode == "dirichlet":
        parts = [[] for _ in range(k)]
        for g in ids:
            idx = rng.permutation(np.flatnonzero(groups == g))
            props = rng.dirichlet(alpha * np.ones(k))
            cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
            for w, chunk in enumerate(np.split(idx, cuts)):
                parts[w].extend(chunk)
        return [np.sort(np.array(p, dtype=np.int64)) for p in parts]
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------
def collate(examples, pad_id, device):
    """Right-padded (input_ids, attention_mask, labels); labels = -100 on prompt and padding."""
    T = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), T), pad_id, dtype=torch.long)
    m = torch.zeros((len(examples), T), dtype=torch.long)
    y = torch.full((len(examples), T), -100, dtype=torch.long)
    for i, (ids, plen) in enumerate(examples):
        x[i, : len(ids)] = torch.tensor(ids)
        m[i, : len(ids)] = 1
        y[i, plen: len(ids)] = torch.tensor(ids[plen:])
    return x.to(device), m.to(device), y.to(device)


class Sampler:
    """Random batches (with replacement across epochs) from one worker's examples."""

    def __init__(self, examples, idx, batch_size, seed):
        self.examples, self.idx, self.B = examples, np.asarray(idx), batch_size
        self.rng = random.Random(seed)
        self.order, self.pos = [], 0

    def next(self):
        out = []
        while len(out) < self.B:
            if self.pos >= len(self.order):
                self.order = list(self.idx)
                self.rng.shuffle(self.order)
                self.pos = 0
            out.append(self.examples[self.order[self.pos]])
            self.pos += 1
        return out
