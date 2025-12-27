import time
from typing import Dict, Tuple, Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from chen2026spfere.utils.low_dtype_casting import resolve_dtype
from chen2026spfere.client.client_train_test import train as legacy_train
from chen2026spfere.simulation.sim_sd_op import sd_clone, sd_sub

def _count_samples(loader: DataLoader) -> int:
    total = 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        total += len(x)
    return total

def local_train_once(
    model_ctor: Callable[[], nn.Module],
    base_sd: Dict[str, torch.Tensor],
    cid: int,
    train_loader: DataLoader,
    device: torch.device,
    *,
    total_global_round: int,
    global_round_count: int,
    epochs: int,
    client_use_optimizer=None,
    client_lr=None,
    client_lr_gamma=None,
    precision_compression=False,
    precision_dtype=None,
    client_logger_filename: str = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    
    model = model_ctor()
    model.load_state_dict(sd_clone(base_sd), strict=True)

    start_time = time.time()
    interrupt_flag = {'flag': 0}

    t0 = time.perf_counter()
    last_loss, n_eff = legacy_train(
        global_round_count=global_round_count,
        total_round=total_global_round,
        start_time=start_time,
        train_timeout=0,
        interrupt_flag=interrupt_flag,
        model=model,
        device=device,
        train_loader=train_loader,
        epochs=epochs,
        use_optimizer=client_use_optimizer,
        lr=client_lr,
        gamma=client_lr_gamma,
        log_filepath=client_logger_filename,
        cid=cid, precision_compression=precision_compression, low_dtype=resolve_dtype(precision_dtype)
    )
    compute_ms = (time.perf_counter() - t0) * 1000.0 # record the real computing time on the GPU

    new_sd = sd_clone(model.state_dict())
    delta = sd_sub(new_sd, base_sd)

    stats: Dict[str, Any] = {
        "loss": float(last_loss),
        "samples": _count_samples(train_loader),
        "n_eff": int(n_eff),
        "compute_ms": compute_ms,
        "interrupted": int(interrupt_flag.get('flag', 0)),
        "round": int(global_round_count)
    }
    return new_sd, delta, stats