import torch
from torch import nn

def resolve_dtype(name: str):
    n = name.lower()
    if n == "fp16":  return torch.float16
    if n == "bf16":  return torch.bfloat16
    if n == "fp32":  return torch.float32
    raise ValueError(f"Unknown PRECISION_DTYPE: {name}")

NORM_TYPES = (
    torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
    torch.nn.LayerNorm, torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
)

@torch.no_grad()
def cast_non_norm_only(model: nn.Module, lowp: torch.dtype, keep_norm_fp32: bool = True):
    for module in model.modules():
        is_norm = isinstance(module, NORM_TYPES)

        for name, p in module.named_parameters(recurse=False):
            if not p.is_floating_point():
                continue
            if keep_norm_fp32 and is_norm:
                if p.dtype != torch.float32:
                    p.data = p.data.to(dtype=torch.float32)
            else:
                if p.dtype != lowp:
                    p.data = p.data.to(dtype=lowp)

        for name, b in module.named_buffers(recurse=False):
            if not b.is_floating_point():
                continue
            if keep_norm_fp32 and is_norm:
                if b.dtype != torch.float32:
                    b.data = b.data.to(dtype=torch.float32)
            else:
                if b.dtype != lowp:
                    b.data = b.data.to(dtype=lowp)