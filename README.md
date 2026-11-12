# DiLoCo for LoRA Fine-Tuning

Author: [Andrew Gyakobo](https://github.com/Gyakobo)

Does DiLoCo's outer optimizer (SGD with Nesterov momentum on the averaged pseudo-gradient)
still help when workers only fine-tune **LoRA adapters**, and does the answer depend on
whether averaging the adapters is **exact**?

Averaging LoRA factors separately is inexact: `mean(B_i) mean(A_i) != mean(B_i A_i)`.
Federated-LoRA work fixes this with plain averaging; DiLoCo adds outer momentum on top.
This repo crosses the two:

| `--agg` | What is averaged | Exact? | Uplink / downlink per round |
|---|---|---|---|
| `fedit` | A and B separately (FedIT) | no | A+B / A+B |
| `ffa` | B only, A frozen (FFA-LoRA) | yes | B / B |
| `fedex` | A and B + dense residual into the base (FedEx-LoRA) | yes | A+B / A+B+dense |
| `merge` | dense updates `s·B_i A_i`, outer step on the dense weight | yes | A+B / dense (or rank-`r_d`) |
| `full` | all weights, no LoRA (reference) | yes | all / all |

Outer optimizer: `--outer_lr 1 --outer_momentum 0` is plain averaging (FedAvg);
`--outer_lr 0.7 --outer_momentum 0.9 --nesterov` is DiLoCo.

Setup: GPT-2 small (124M) and SmolLM2-360M, Dolly-15k (natural non-i.i.d. split by category)
and AG News (label-skewed Dirichlet split), up to 8 workers simulated on one RTX 3070 Ti (8 GB).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v            # 32 correctness checks, CPU only, ~20 s
python train.py --task dolly --agg fedit --k 4 --H 50 --batch_size 8 --micro_batch 4
```

## Running the study (in this order)

```bash
python sweep.py configs/0_pilot.yaml          # pick inner lr, check over-fitting, time runs
# -> update lr in the configs and SEC_PER_WORKER_STEP in sweep.py
python sweep.py configs/1_core.yaml           # main grid, seed 0
python aggregate.py --tune                    # best Nesterov outer lr -> put into 1b
python sweep.py configs/1b_core_seeds.yaml configs/2_references.yaml configs/2b_full_ft.yaml
python sweep.py configs/3_mechanism.yaml configs/4_noniid.yaml configs/6_downlink.yaml
python sweep.py configs/5_scale.yaml
python aggregate.py && python aggregate.py --paired
```

`python sweep.py <config> --dry_run` lists the runs and an estimated time. Sweeps resume
where they stopped.

## Correctness guarantees (tests/)

* a freshly wrapped LoRA model computes exactly the base model
* one worker + plain averaging reproduces ordinary LoRA training for `fedit`, `ffa`, `fedex`
* `fedex` and `merge` give the same model after one round (both exact); `fedit` does not
* `fedit`'s measured aggregation error is 0 for one worker and > 0 for several
* the manual Nesterov step used by `merge` equals `torch.optim.SGD`
* a full-rank truncated downlink equals the dense one

## License
MIT
