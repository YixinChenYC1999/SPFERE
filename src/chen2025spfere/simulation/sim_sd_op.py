import torch
from typing import Dict

def sd_clone(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        out[k] = v.detach().clone()
    return out

def sd_add(base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in base.keys():
        out[k] = base[k] + delta[k]
    return out

def sd_sub(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in a.keys():
        out[k] = a[k] - b[k]
    return out