from typing import Dict, Callable
import torch
from torch import nn
from torch.utils.data import DataLoader

from chen2026spfere.utils.server_logger import fl_logger
from chen2026spfere.utils.performance_test import test as server_test, recalibrate_bn_stats
from chen2026spfere.utils.low_dtype_casting import resolve_dtype
from chen2026spfere.simulation.sim_sd_op import sd_clone


def server_eval_once(
    global_sd: Dict[str, torch.Tensor],
    model_ctor: Callable[[], nn.Module],
    device: torch.device,
    variance: float,
    cosine_similarity: float,
    worst_10: float,
    best_10: float,
    *,
    test_loader: DataLoader,
    calib_loader: DataLoader,
    global_round: int,
    global_stage: str,
    wb_logger=None,
    wb_logger_enable,
    precision_compression=False,
    precision_dtype=None,
    server_logger_filename: str = None,
):
    m = model_ctor()
    m.load_state_dict(sd_clone(global_sd), strict=True)
    if calib_loader != None:
        used = recalibrate_bn_stats(m, calib_loader, device, max_batches=50)
        fl_logger(f"[BN] recalibrated with {used} samples.", server_logger_filename)
    test_acc, test_loss = server_test(
        model=m,
        device=device,
        test_loader=test_loader,
        global_round=global_round,
        global_stage=global_stage,
        variance=variance,
        cosine_similarity=cosine_similarity,
        worst_10=worst_10, 
        best_10=best_10,
        log_filepath=server_logger_filename,
        wb_logger=wb_logger,
        wb_logger_enable=wb_logger_enable,
        precision_compression=precision_compression, low_dtype=resolve_dtype(precision_dtype)
    )
    return test_acc, test_loss