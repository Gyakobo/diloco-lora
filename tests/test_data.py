"""Data pipeline checks with a fake tokenizer and fake `datasets` (no downloads)."""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data as D  # noqa: E402
from train import TrainConfig  # noqa: E402


class FakeTok:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + (ord(c) % 90) for c in text]}

    def decode(self, ids):
        return "".join(chr(i + 31) for i in ids)


class FakeDS(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self]
        return list.__getitem__(self, key)


def fake_load_dataset(name, split):
    rng = np.random.default_rng(len(split))
    if "dolly" in name:
        cats = ["open_qa", "brainstorming", "summarization", "classification"]
        return FakeDS([dict(instruction=f"q{i}", context="c" * (i % 3), response="r" * (5 + i % 7),
                            category=cats[i % 4]) for i in range(2500)])
    return FakeDS([dict(text="news " * int(rng.integers(3, 80)), label=int(rng.integers(0, 4)))
                   for i in range(3000 if split == "train" else 2500)])


def test_load_tasks(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    for task in ["dolly", "agnews"]:
        cfg = TrainConfig(task=task, data_dir=str(tmp_path), block_size=128, agnews_train=2000)
        d = D.load_task(cfg, FakeTok())
        assert len(d["train"]) == len(d["train_groups"]) > 0
        assert len(d["val"]) == 1000 and len(d["test"]) == 1000
        assert not set(map(tuple, (x for x, _ in d["val"]))) & set(map(tuple, (x for x, _ in d["test"]))) or task == "agnews"
        for ids, plen in d["train"][:50]:
            assert 0 < plen < len(ids) <= 128 and ids[-1] == 0 or len(ids) == 128
        if task == "agnews":
            assert len(d["label_ids"]) == 4 and len(d["val_labels"]) == len(d["val"])
        # cached second load gives the same thing
        d2 = D.load_task(cfg, FakeTok())
        assert d2["train"][:5] == d["train"][:5]


def test_collate_masks_prompt():
    x, m, y = D.collate([([5, 6, 7, 8], 2), ([9, 10], 1)], pad_id=0, device="cpu")
    assert x.tolist() == [[5, 6, 7, 8], [9, 10, 0, 0]]
    assert m.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert y.tolist() == [[-100, -100, 7, 8], [-100, 10, -100, -100]]


def test_group_split_is_non_iid():
    groups = np.repeat(np.arange(8), 100)
    parts = D.partition(groups, 4, "group")
    for p in parts:
        assert len(np.unique(groups[p])) == 2      # 8 categories over 4 workers -> 2 each
