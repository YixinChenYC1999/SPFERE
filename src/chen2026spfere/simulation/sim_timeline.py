import torch
from typing import Dict, List, Tuple
from chen2026spfere.simulation.sim_sd_op import sd_add, sd_clone, sd_sub

def _timeline_append(timeline: List[Tuple[float, Dict[str, torch.Tensor]]],
                     t: float, sd: Dict[str, torch.Tensor]) -> None:
    timeline.append((float(t), sd_clone(sd)))  # 存副本更安全
    
def _timeline_pick_sd_at(
    timeline: List[Tuple[float, Dict[str, torch.Tensor]]], 
    t: float
) -> Tuple[float, Dict[str, torch.Tensor]]:
    sel_idx = 0
    for i, (tt, _) in enumerate(timeline):
        if tt <= t:
            sel_idx = i
        else:
            break
    sel_t, sel_sd = timeline[sel_idx]
    return float(sel_t), sel_sd