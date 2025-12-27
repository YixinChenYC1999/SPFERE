import time
import torch
from torch import nn
from typing import Dict, List, Tuple, Any, Callable
from chen2026spfere.utils.server_logger import fl_logger
from chen2026spfere.server.server_aggregation import server_aggregation
from chen2026spfere.simulation.sim_sd_op import sd_clone

def legacy_aggregate_apply(
    model_ctor: Callable[[], nn.Module],
    base_sd: Dict[str, torch.Tensor],
    batch_recs: List[Dict[str, Any]],
    *,
    lr: float,
    q: float,
    ratio_map: Dict[int, float] | None = None,
    freq_map: Dict[int, float] | None = None,
    device: torch.device,
    last_loss=None,
    aggr_method=None,
    aggr_accum_fp32=None,
    top_k=None,
    server_logger_filename=None
) -> Tuple[Dict[str, torch.Tensor], float, float]:
    start = time.perf_counter()
    client_state_dicts: List[Dict[str, torch.Tensor]] = []
    client_last_loss: List[float] = []
    client_test_acc: List[float] = []
    client_n_eff: List[int] = []
    client_interrupt_flags: List[int] = []
    client_freq: List[float] = []

    for rec in batch_recs:
        cid = int(rec.get("cid", -1))
        client_state_dicts.append(rec["delta"])
        client_last_loss.append(float(rec.get("loss", 0.0)))
        client_test_acc.append(float(rec.get("acc", 0.0)))
        client_n_eff.append(rec.get("n_eff"))
        client_interrupt_flags.append(int(rec.get("interrupted", 0)))
        client_freq.append(float((freq_map or {}).get(cid, 1.0)))
    tmp = model_ctor()
    tmp.load_state_dict(sd_clone(base_sd), strict=True)

    updated, variance, cosine_similarity, worst_10, best_10 = server_aggregation(
        global_model=tmp,
        client_state_dicts=client_state_dicts,
        client_last_loss=client_last_loss,
        client_test_acc=client_test_acc,
        client_ratio=client_n_eff,
        client_interrupt_flags=client_interrupt_flags,
        client_freq=client_freq,
        q=q,
        learning_rate=lr,
        server_logger_filename = server_logger_filename,
        mode=aggr_method,
        aggr_accum_fp32=aggr_accum_fp32,
        top_k=top_k,
        last_loss=last_loss
    )
    end = time.perf_counter()
    fl_logger(f"[Aggregation time counter] Aggregation used {end - start:.6f} sec(s)", server_logger_filename)
    return sd_clone(updated.state_dict()), variance, cosine_similarity, worst_10, best_10