"""
Minimal, explicit LoRA (Hu et al., 2021) for Hugging Face causal LMs.

We write our own instead of using `peft` because the experiments need direct control
over the A and B factors: averaging them, freezing A, re-initialising them every round,
and merging dense updates into the base weight.

Every wrapped layer computes
    y = x W^T + b + s * (x A^T) B^T,        s = alpha / r
with W frozen, A of shape (r, in), B of shape (out, r). B starts at zero, so a freshly
wrapped model computes exactly the same function as the base model.

GPT-2's attention/MLP layers are `Conv1D` (weight stored as (in, out)); they are converted
to the (out, in) layout on wrapping, so every LoRA layer looks the same afterwards.
"""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# default target modules per model family (attention + MLP projections)
DEFAULT_TARGETS = {
    "gpt2": r"(c_attn|c_proj|c_fc)$",
    "qwen2": r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    "llama": r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
}


class LoRALinear(nn.Module):
    def __init__(self, weight, bias, r, alpha, dropout=0.0):
        super().__init__()
        out_f, in_f = weight.shape
        # base weight kept in fp32: dense updates are merged into it many times, and bf16
        # would round small updates away. Compute still runs in bf16 under autocast.
        self.weight = nn.Parameter(weight.detach().float().clone(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.detach().float().clone(),
                                                           requires_grad=False)
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self):
        return self.weight.shape[1]

    @property
    def out_features(self):
        return self.weight.shape[0]

    def reset_lora(self, generator=None):
        """A ~ Kaiming-uniform (as in the LoRA reference code), B = 0."""
        bound = 1.0 / math.sqrt(self.in_features)   # kaiming_uniform_(a=sqrt(5)) bound
        with torch.no_grad():
            a = torch.rand(self.lora_A.shape, generator=generator) * 2 * bound - bound
            self.lora_A.copy_(a.to(self.lora_A.device))
            self.lora_B.zero_()

    def delta(self):
        """Dense weight update this adapter represents: s * B A, shape (out, in)."""
        return self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x):
        # the model is kept in fp32 and run under autocast, so F.linear handles the casting
        y = F.linear(x, self.weight, self.bias)
        return y + self.scale * F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)


def _conv1d_cls():
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except Exception:  # pragma: no cover
        return None


def wrap_lora(model, r, alpha, targets=None, dropout=0.0, seed=0):
    """Freeze every base parameter and replace matching Linear/Conv1D layers by LoRALinear.

    Returns the list of (name, LoRALinear) in a fixed order (module traversal order), which is
    the order used everywhere else (averaging, communication accounting).
    """
    family = model.config.model_type
    pattern = re.compile(targets or DEFAULT_TARGETS.get(family, r"(q_proj|v_proj)$"))
    Conv1D = _conv1d_cls()
    for p in model.parameters():
        p.requires_grad_(False)

    to_replace = []
    for name, mod in model.named_modules():
        if not pattern.search(name):
            continue
        if isinstance(mod, nn.Linear):
            to_replace.append((name, mod.weight, mod.bias))
        elif Conv1D is not None and isinstance(mod, Conv1D):
            to_replace.append((name, mod.weight.t(), mod.bias))   # (in,out) -> (out,in)
    if not to_replace:
        raise ValueError(f"no modules matched {pattern.pattern!r}")

    gen = torch.Generator().manual_seed(seed)
    layers = []
    for name, w, b in to_replace:
        lora = LoRALinear(w, b, r, alpha, dropout).to(w.device)
        lora.reset_lora(gen)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, lora)
        layers.append((name, lora))
    return layers


def lora_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def adapter_params(layers, which="AB"):
    """Flat list of adapter tensors in a fixed order. which: 'AB' or 'B'."""
    out = []
    for _, m in layers:
        if "A" in which:
            out.append(m.lora_A)
        out.append(m.lora_B)
    return out


def count(layers):
    """Parameter counts used for communication accounting."""
    a = sum(m.lora_A.numel() for _, m in layers)
    b = sum(m.lora_B.numel() for _, m in layers)
    dense = sum(m.weight.numel() for _, m in layers)
    return dict(A=a, B=b, dense=dense)
