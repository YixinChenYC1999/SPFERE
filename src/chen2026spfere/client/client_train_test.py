import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch import amp
from chen2026spfere.utils.client_logger import fl_logger

NORM_TYPES = (
    torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
    torch.nn.LayerNorm, torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
)
def assert_model_policy(model: nn.Module, device: torch.device,
                        lowp: torch.dtype | None, keep_norm_fp32: bool,
                        max_report: int = 10):
    import torch.nn as nn
    NORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
            nn.InstanceNorm2d, nn.InstanceNorm3d)

    # device check
    for p in model.parameters():
        assert p.device == device, f"Param on {p.device}, expected {device}"

    # precision check
    mismatches = []
    if lowp is None:
        for name, module in model.named_modules():
            for pname, p in module.named_parameters(recurse=False):
                if p.is_floating_point() and p.dtype != torch.float32:
                    mismatches.append(f"{name}.{pname}: {p.dtype} != float32")
            for bname, b in module.named_buffers(recurse=False):
                if b.is_floating_point() and b.dtype != torch.float32:
                    mismatches.append(f"{name}.{bname}: {b.dtype} != float32")
    else:
        for name, module in model.named_modules():
            is_norm = isinstance(module, NORM)
            for pname, p in module.named_parameters(recurse=False):
                if not p.is_floating_point(): 
                    continue
                exp = torch.float32 if (keep_norm_fp32 and is_norm) else lowp
                if p.dtype != exp:
                    mismatches.append(f"{name}.{pname}: {p.dtype} != {exp}")
            for bname, b in module.named_buffers(recurse=False):
                if not b.is_floating_point():
                    continue
                exp = torch.float32 if (keep_norm_fp32 and is_norm) else lowp
                if b.dtype != exp:
                    mismatches.append(f"{name}.{bname}: {b.dtype} != {exp}")

    if mismatches:
        head = "\n  ".join(mismatches[:max_report])
        more = f"\n  ... (+{len(mismatches)-max_report} more)" if len(mismatches) > max_report else ""
        raise AssertionError("Dtype policy mismatches:\n  " + head + more)
                        
                        

def train(global_round_count, total_round, start_time, train_timeout, interrupt_flag, model, device, train_loader, epochs, use_optimizer="sgd", lr=0.01, gamma=1.0, log_filepath="client_log/log.log", cid=None, precision_compression=False, low_dtype=None):
    model.train()
    if use_optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else: # default use sgd with lr decay at 50% and 80%
        if global_round_count <= int(total_round*0.5):
            lr=lr
        elif global_round_count <= int(total_round*0.8):
            lr=lr*gamma
        else:
            lr=lr*gamma*gamma
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    last_loss = 0.0
    n_eff = 0

    use_amp = bool(precision_compression) and (low_dtype in (torch.float16, torch.bfloat16))
    if device.type == "cpu" and low_dtype == torch.float16:
        use_amp = False
    amp_dtype = low_dtype if use_amp else torch.float32
    
    for epoch in range(epochs):
        if train_timeout != 0 and interrupt_flag != 1:
            if time.time() - start_time > train_timeout:
                interrupt_flag['flag'] = 1
                log_message = "[X] Local training timeout at train process!"
                fl_logger(log_message, log_filepath)
                break
            
        correct, total = 0, 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            if use_amp:
                data = data.to(dtype=low_dtype)
                
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with amp.autocast(device_type=device.type, dtype=amp_dtype):
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            else:
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            n_eff += target.size(0)
            
            # acc count
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        last_loss = loss.item()
        acc = correct / total if total > 0 else 0.0
        if cid == None:
            log_message = f"[+] Training Epoch: {epoch+1}, Loss: {last_loss:.8f}, Acc: {acc*100:.2f}%, LR: {optimizer.param_groups[0]['lr']}"
        else:
            log_message = f"[+] cid_{cid:03d} Training Epoch: {epoch+1}, Loss: {last_loss:.8f}, Acc: {acc*100:.2f}%, LR: {optimizer.param_groups[0]['lr']}"
        fl_logger(log_message, log_filepath)
    return last_loss, n_eff