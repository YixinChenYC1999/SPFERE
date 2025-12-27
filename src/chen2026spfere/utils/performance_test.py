import torch
import torch.nn.functional as F
from torch import amp
from chen2026spfere.utils.server_logger import fl_logger
from torch import nn
from torch.utils.data import DataLoader

def fmt_float(x, digits=8):
    return f"{x:.{digits}f}" if x is not None else "None"

def test(model, device, test_loader, global_round, global_stage, variance, cosine_similarity, worst_10, best_10, log_filepath=None, wb_logger=None, wb_logger_enable=False, precision_compression=False, low_dtype=None):
    model.eval()
    
    use_amp = bool(precision_compression) and (low_dtype in (torch.float16, torch.bfloat16))
    if device.type == "cpu" and low_dtype == torch.float16:
        use_amp = False
    amp_dtype = low_dtype if use_amp else torch.float32
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            if use_amp:
                data = data.to(dtype=low_dtype)
                with amp.autocast(device_type=device.type, dtype=amp_dtype):
                    output = model(data)
                    test_loss += F.cross_entropy(output, target, reduction="sum").item()
            else:
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_test_samples = len(test_loader.dataset)
    test_loss /= total_test_samples
    test_acc = 100. * correct / total_test_samples
    log_message = (
        f"[=] Test set on global round {global_round} stage {global_stage}:"
        f" Average loss: {fmt_float(test_loss)}, Accuracy: {correct}/{total_test_samples} ({test_acc:.2f}%)"
    )
    fl_logger(log_message, log_filepath)
    if wb_logger is not None and wb_logger_enable is True:
        wb_logger.log({"acc": test_acc, "loss": test_loss, "variance": variance, "cosine_similarity": cosine_similarity, "worst_10": worst_10, "best_10": best_10})
        log_message = (
            f"[=] Variance: {fmt_float(variance)}, Cosine Similarity: {fmt_float(cosine_similarity)},"
            f" worst_10: {fmt_float(worst_10)}, best_10: {fmt_float(best_10)}"
        )
        fl_logger(log_message, log_filepath)
    return test_acc, test_loss


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
def _has_running_stats(m: nn.Module) -> bool:
    return hasattr(m, "running_mean") and hasattr(m, "running_var") and (m.running_mean is not None) and (m.running_var is not None)

@torch.no_grad()
def recalibrate_bn_stats(model: nn.Module, loader: DataLoader, device, max_batches=None) -> int:
    model.to(device)
    original_momentum: dict[int, float | None] = {}

    model.eval()
    for m in model.modules():
        if isinstance(m, BN_TYPES) or _has_running_stats(m):
            m.reset_running_stats()
            original_momentum[id(m)] = m.momentum
            m.momentum = None
            m.train()
        else:
            pass

    seen = 0
    param_dtype = next(model.parameters()).dtype

    for i, batch in enumerate(loader):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True).to(dtype=param_dtype)

        _ = model(x)
        seen += x.size(0)

        if (max_batches is not None) and (i + 1 >= max_batches):
            break

    for m in model.modules():
        if isinstance(m, BN_TYPES) or _has_running_stats(m):
            m.momentum = original_momentum.get(id(m), m.momentum)

    return seen