"""
Correctness checks. Run on a tiny randomly initialised GPT-2 and random tokens, so no download
or GPU is needed:

    python -m pytest tests/ -v

Key identities:
  * a freshly wrapped LoRA model computes exactly the base model (B = 0)
  * with ONE worker, outer_lr = 1 and no momentum, DiLoCo must reproduce plain LoRA training,
    for every aggregation method whose state persists across rounds (fedit, ffa, fedex)
  * fedex and merge are EXACT aggregations: after one round they give the same model
  * fedit is NOT exact: with several workers it gives a different model, and its measured
    aggregation error is zero for one worker and positive for several
"""
import copy
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lora as L  # noqa: E402
from data import partition  # noqa: E402
from train import TrainConfig, run, sgd_step, svd_truncate  # noqa: E402

VOCAB = 100


def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(123)
    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=VOCAB,
                                      n_positions=64, bos_token_id=0, eos_token_id=0))


@pytest.fixture(scope="module")
def base_model():
    return tiny_model()


@pytest.fixture(scope="module")
def task():
    rng = np.random.default_rng(0)

    def ex():
        # learnable sequences: arithmetic progressions mod VOCAB, so validation loss can drop
        n, start, step = int(rng.integers(12, 40)), int(rng.integers(1, VOCAB)), int(rng.integers(1, 4))
        return [(start + i * step) % (VOCAB - 1) + 1 for i in range(n)], int(rng.integers(4, 10))
    train = [ex() for _ in range(400)]
    val = [ex() for _ in range(40)]
    test = [ex() for _ in range(40)]
    return dict(train=train, train_groups=rng.integers(0, 4, len(train)), val=val, test=test,
                val_labels=None, test_labels=None, label_ids=None, pad_id=0)


def cfg(**kw):
    base = dict(rank=4, lora_alpha=8, steps=12, H=3, k=1, batch_size=4, micro_batch=2,
                lr=3e-3, warmup=2, eval_interval=3, eval_examples=40, dtype="fp32",
                device="cpu", seed=0, block_size=64)
    base.update(kw)
    return TrainConfig(**base)


def go(c, base_model, task):
    rows, summary, model = run(c, model=copy.deepcopy(base_model), task=task,
                               write=False, verbose=False)
    return rows, summary, model


def vals(rows):
    return np.array([r["val_loss"] for r in rows])


# ---------------------------------------------------------------------------
def test_wrap_is_identity(base_model):
    m = copy.deepcopy(base_model).eval()
    x = torch.randint(1, VOCAB, (2, 16))
    before = m(x).logits
    L.wrap_lora(m, r=4, alpha=8)
    after = m(x).logits
    torch.testing.assert_close(before, after, rtol=0, atol=1e-5)


@pytest.mark.parametrize("agg", ["fedit", "ffa", "fedex"])
@pytest.mark.parametrize("H", [1, 3])
def test_k1_avg_equals_local(base_model, task, agg, H):
    local, _, _ = go(cfg(mode="local", agg=agg), base_model, task)
    dil, _, _ = go(cfg(mode="diloco", agg=agg, k=1, H=H, outer_lr=1.0, outer_momentum=0.0),
                   base_model, task)
    np.testing.assert_allclose(vals(dil), vals(local), rtol=0, atol=1e-4)


def test_merge_one_round_equals_local(base_model, task):
    local, _, _ = go(cfg(mode="local", agg="fedit"), base_model, task)
    mer, _, _ = go(cfg(mode="diloco", agg="merge", k=1, H=12, outer_lr=1.0, outer_momentum=0.0,
                       inner_state="reset", eval_interval=12), base_model, task)
    assert abs(vals(mer)[-1] - vals(local)[-1]) < 1e-4


def test_exact_methods_agree_and_fedit_differs(base_model, task):
    common = dict(mode="diloco", k=3, H=12, outer_lr=1.0, outer_momentum=0.0,
                  inner_state="reset", eval_interval=12, lr=1e-2)
    fedex, _, _ = go(cfg(agg="fedex", **common), base_model, task)
    merge, _, _ = go(cfg(agg="merge", **common), base_model, task)
    fedit, _, _ = go(cfg(agg="fedit", **common), base_model, task)
    assert abs(vals(fedex)[-1] - vals(merge)[-1]) < 1e-4
    assert abs(vals(fedit)[-1] - vals(merge)[-1]) > 1e-4
    assert fedit[-1]["agg_err"] > 0


def test_fedit_agg_err_zero_for_one_worker(base_model, task):
    rows, _, _ = go(cfg(mode="diloco", agg="fedit", k=1, H=3, outer_lr=1.0, outer_momentum=0.0),
                    base_model, task)
    errs = [r["agg_err"] for r in rows if r["agg_err"] is not None]
    assert errs and max(errs) < 1e-5


def test_full_rank_downlink_equals_dense(base_model, task):
    common = dict(mode="diloco", agg="merge", k=2, H=3, inner_state="reset")
    dense, _, _ = go(cfg(**common), base_model, task)
    trunc, _, _ = go(cfg(down_rank=1000, **common), base_model, task)
    np.testing.assert_allclose(vals(trunc), vals(dense), rtol=0, atol=1e-4)


@pytest.mark.parametrize("agg", ["fedit", "ffa", "fedex", "merge", "full"])
def test_training_reduces_loss(base_model, task, agg):
    rows, s, _ = go(cfg(mode="diloco", agg=agg, k=2, H=3, steps=24,
                        inner_state="reset" if agg == "merge" else "keep"), base_model, task)
    v = vals(rows)
    assert np.isfinite(v).all() and v[-1] < v[0]
    assert s["comm_up"] > 0 and s["comm_down"] >= s["comm_up"]


def test_sgd_step_matches_torch():
    torch.manual_seed(0)
    for nest in (True, False):
        p = torch.nn.Parameter(torch.randn(5, 4))
        q = p.detach().clone()
        opt = torch.optim.SGD([p], lr=0.7, momentum=0.9, nesterov=nest)
        buf = torch.zeros_like(q)
        for _ in range(4):
            g = torch.randn(5, 4)
            p.grad = g.clone()
            opt.step()
            q += sgd_step(g, buf, 0.7, 0.9, nest)
        torch.testing.assert_close(p.detach(), q)


def test_svd_truncate():
    M = torch.randn(8, 3) @ torch.randn(3, 6)
    torch.testing.assert_close(svd_truncate(M, 3), M, rtol=1e-4, atol=1e-4)
    assert torch.linalg.matrix_rank(svd_truncate(M, 1)) == 1


@pytest.mark.parametrize("mode", ["iid", "group", "dirichlet"])
@pytest.mark.parametrize("k", [2, 4, 8])
def test_partition_is_a_partition(mode, k):
    groups = np.random.default_rng(0).integers(0, 4, 1000)
    parts = partition(groups, k, mode, alpha=0.3, seed=0)
    allidx = np.concatenate(parts)
    assert len(parts) == k
    assert len(allidx) == 1000 and len(np.unique(allidx)) == 1000


def test_comm_cost_ordering(base_model):
    from train import comm_cost
    m = copy.deepcopy(base_model)
    layers = L.wrap_lora(m, r=4, alpha=8)
    c = {a: comm_cost(cfg(agg=a), layers, m) for a in ["fedit", "ffa", "fedex", "merge"]}
    assert c["ffa"][0] < c["fedit"][0]                 # B only
    assert c["fedex"][1] > c["fedit"][1]               # dense residual on the downlink
    assert c["merge"][1] == L.count(layers)["dense"]


def test_classification_accuracy_runs(base_model, task):
    t = dict(task)
    t["label_ids"] = [[5], [6], [7, 8], [9]]
    t["val_labels"] = np.random.default_rng(0).integers(0, 4, len(task["val"]))
    t["test_labels"] = np.random.default_rng(1).integers(0, 4, len(task["test"]))
    rows, s, _ = go(cfg(mode="local", agg="fedit", steps=3, eval_interval=3), base_model, t)
    assert 0.0 <= s["final_val_acc"] <= 1.0 and 0.0 <= s["best_test_acc"] <= 1.0


def test_run_names_distinguish_settings():
    a = TrainConfig(mode="diloco", batch_size=4).run_name()
    b = TrainConfig(mode="diloco", batch_size=8).run_name()
    assert a != b


def test_clash_with_different_settings_is_refused(base_model, task, tmp_path):
    c = cfg(mode="local", agg="fedit", steps=3, eval_interval=3, out_dir=str(tmp_path), name="x")
    run(c, model=copy.deepcopy(base_model), task=task, verbose=False)
    assert run(c, model=copy.deepcopy(base_model), task=task, verbose=False) is None   # skip
    with pytest.raises(SystemExit):
        run(cfg(mode="local", agg="fedit", steps=3, eval_interval=3, out_dir=str(tmp_path),
                name="x", lr=1e-4), model=copy.deepcopy(base_model), task=task, verbose=False)


@pytest.mark.parametrize("agg", ["fedit", "merge", "full"])
def test_best_checkpoint_is_restored(base_model, task, agg):
    """Scoring the restored best-val checkpoint on VAL must give back the best val loss."""
    import train as T
    c = cfg(mode="diloco", agg=agg, k=2, H=3, steps=24, eval_examples=40,
            inner_state="reset" if agg == "merge" else "keep")
    t = dict(task)
    t["test"] = task["val"]            # so the 'test' score of the checkpoint is its val score
    rows, s, _ = go(c, base_model, t)
    assert abs(s["best_test_loss"] - s["best_val_loss"]) < 1e-4
    assert s["best_val_loss"] == min(r["val_loss"] for r in rows)
