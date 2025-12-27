import numpy as np
import torch
from chen2026spfere.utils.server_logger import fl_logger

def fairness_cal(client_test_acc, server_logger_filename):
    fl_logger(f"[=] test acc for each client's local dataset: {client_test_acc}", server_logger_filename)
    variance = np.var(client_test_acc)
    fl_logger(f"[=] variance: {variance}", server_logger_filename)
    ones_vector = np.ones_like(client_test_acc)
    dot_product = np.dot(client_test_acc, ones_vector)
    norm_a = np.linalg.norm(client_test_acc)
    norm_b = np.linalg.norm(ones_vector)
    cosine_similarity = dot_product / (norm_a * norm_b)
    fl_logger(f"[=] cosine similarity: {cosine_similarity}", server_logger_filename)
    k = max(1, int(round(0.1 * len(client_test_acc))))
    sorted_acc = np.sort(client_test_acc)
    worst_10 = np.mean(sorted_acc[:k])
    best_10  = np.mean(sorted_acc[-k:])
    return variance, cosine_similarity, worst_10, best_10

def _acc_dtype_of(t: torch.Tensor, aggr_accum_fp32) -> torch.dtype:
        return torch.float32 if aggr_accum_fp32 and t.is_floating_point() else t.dtype

def fedavg(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, learning_rate, server_logger_filename, aggr_accum_fp32):
    fl_logger(f"[=] Applying FedAvg aggregation", server_logger_filename)
    fl_logger(f"[+] Client ratio: {client_ratio}", server_logger_filename)
    variance, cosine_similarity, worst_10, best_10 = fairness_cal(client_test_acc, server_logger_filename)
    with torch.no_grad():
        # add ratio
        client_weights = [x / sum(client_ratio) for x in client_ratio]
        sum_weights = sum(client_weights) + 1e-10
        client_weights_normalized = [w / sum_weights for w in client_weights]
        fl_logger(f"[+] Final normalized weights: {[float(w) for w in client_weights_normalized]}", server_logger_filename)
        # make global update
        global_state_dict = global_model.state_dict()
        global_update = {}
        for key, val in global_state_dict.items():
            if (not val.is_floating_point()) or ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key):
                global_update[key] = None
            else:
                acc_dtype = _acc_dtype_of(val, aggr_accum_fp32)
                global_update[key] = torch.zeros_like(val, dtype=acc_dtype, device=val.device)
        for client_idx, client_dict in enumerate(client_state_dicts):
            w = client_weights_normalized[client_idx]
            for key, acc in global_update.items():
                if acc is None:
                    continue
                delta_i = client_dict[key]
                if delta_i.device != acc.device or delta_i.dtype != acc.dtype:
                    delta_i = delta_i.to(device=acc.device, dtype=acc.dtype)
                acc.add_(delta_i, alpha=w)
        for key, acc in global_update.items():
            if acc is None:
                continue
            gparam = global_state_dict[key]
            acc.mul_(learning_rate)
            if acc.dtype != gparam.dtype:
                acc = acc.to(dtype=gparam.dtype)
            gparam.add_(acc)
            
        global_model.load_state_dict(global_state_dict)
    fl_logger(f"[=] FedAvg aggregation finished!", server_logger_filename)
    return global_model, variance, cosine_similarity, worst_10, best_10

def afl(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32, top_k=None):
    fl_logger(f"[=] Applying AFL aggregation (q={q})", server_logger_filename)
    fl_logger(f"[+] Client ratio: {client_ratio}", server_logger_filename)
    variance, cosine_similarity, worst_10, best_10 = fairness_cal(client_test_acc, server_logger_filename)
    with torch.no_grad():
        eps  = 1e-12
        tau  = 1.0 / max(float(q), 1e-6)
        topk = top_k
        floor = 0.0
        L = torch.as_tensor(client_last_loss, dtype=torch.float64) + eps
        r = torch.as_tensor(client_ratio,     dtype=torch.float64)
        r = r / (r.sum() + eps)
        x = torch.log(L) / tau
        def _finalize(alpha: torch.Tensor) -> torch.Tensor:
            a = alpha * r
            Z = a.sum() + eps
            w = a / Z
            if floor > 0.0:
                N = w.numel()
                w = (1.0 - floor) * w + floor / N
                w = w / (w.sum() + eps)
            return w
        if topk is not None:
            # topk
            N = L.numel()
            k = int(round(topk * N))
            k = max(1, min(k, N))
            top_idx = torch.topk(L, k=k).indices
            x_c = x[top_idx]
            x_c = x_c - (x_c.max() if x_c.numel() > 0 else 0.0)
            alpha = torch.zeros_like(L)
            if x_c.numel() > 0:
                alpha[top_idx] = torch.exp(x_c)
            w = _finalize(alpha)
        else:
            # softmax
            x_c = x - x.max()
            alpha = torch.exp(x_c)
            w = _finalize(alpha)
        client_weights_normalized = w
        fl_logger(f"[+] Final normalized weights: {[float(w) for w in client_weights_normalized]}", server_logger_filename)
        # make global update
        global_state_dict = global_model.state_dict()
        global_update = {}
        for key, val in global_state_dict.items():
            if (not val.is_floating_point()) or ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key):
                global_update[key] = None
            else:
                acc_dtype = _acc_dtype_of(val, aggr_accum_fp32)
                global_update[key] = torch.zeros_like(val, dtype=acc_dtype, device=val.device)
        for client_idx, client_dict in enumerate(client_state_dicts):
            w = client_weights_normalized[client_idx]
            for key, acc in global_update.items():
                if acc is None:
                    continue
                delta_i = client_dict[key]
                if delta_i.device != acc.device or delta_i.dtype != acc.dtype:
                    delta_i = delta_i.to(device=acc.device, dtype=acc.dtype)
                acc.add_(delta_i, alpha=w)
        for key, acc in global_update.items():
            if acc is None:
                continue
            gparam = global_state_dict[key]
            acc.mul_(learning_rate)
            if acc.dtype != gparam.dtype:
                acc = acc.to(dtype=gparam.dtype)
            gparam.add_(acc)

        global_model.load_state_dict(global_state_dict)
    fl_logger(f"[=] AFL aggregation finished!", server_logger_filename)
    return global_model, variance, cosine_similarity, worst_10, best_10

def fair(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32, global_ref_loss: float, quality_clamp_min: float = 0.0, eta: float = 1.0):
    fl_logger(f"[=] Applying FAIR aggregation (pre-agg + filter + re-agg)", server_logger_filename)
    fl_logger(f"[+] Client ratio: {client_ratio}", server_logger_filename)
    variance, cosine_similarity, worst_10, best_10 = fairness_cal(client_test_acc, server_logger_filename)
    with torch.no_grad():
        # -------------------- (A) m*D --------------------
        eps = 1e-10
        base_weights = []
        used_q = []
        for i, D_i in enumerate(client_ratio):
            m_i = float(global_ref_loss) - float(client_last_loss[i])
            if quality_clamp_min is not None:
                m_i = max(quality_clamp_min, m_i)
            base_weights.append(float(m_i) * float(D_i))
            used_q.append(m_i)
        total_w = sum(base_weights)
        if total_w <= eps:
            fl_logger("[!] All (m*D) weights are zero/near-zero; fallback to pure D-normalization.", server_logger_filename)
            base_weights = [float(D_i) for D_i in client_ratio]
            total_w = sum(base_weights) + eps
        weights_pre = [w / (total_w + eps) for w in base_weights]
        # -------------------- (B) pre-agg --------------------
        global_state_dict = global_model.state_dict()
        pre_accum = {}
        for key, val in global_state_dict.items():
            if (not val.is_floating_point()) or ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key):
                pre_accum[key] = None
            else:
                acc_dtype = _acc_dtype_of(val, aggr_accum_fp32)
                pre_accum[key] = torch.zeros_like(val, dtype=acc_dtype, device=val.device)
        for i, client_dict in enumerate(client_state_dicts):
            w = float(weights_pre[i])
            for key, acc in pre_accum.items():
                if acc is None:
                    continue
                delta_i = client_dict[key]
                if delta_i.device != acc.device or delta_i.dtype != acc.dtype:
                    delta_i = delta_i.to(device=acc.device, dtype=acc.dtype)
                acc.add_(delta_i, alpha=w)
        ref_tensor = next((v for v in pre_accum.values() if v is not None), None)
        if ref_tensor is None:
            fl_logger("[!] No float params to aggregate; skip FAIR filtering this round.", server_logger_filename)
            client_weights_normalized = weights_pre
        else:
            ref_device = next(v for v in pre_accum.values() if v is not None).device
            ref_dtype  = next(v for v in pre_accum.values() if v is not None).dtype
            flat_ref = []
            for key, acc in pre_accum.items():
                if acc is None:
                    continue
                t = acc
                if t.device != ref_device or t.dtype != ref_dtype:
                    t = t.to(device=ref_device, dtype=ref_dtype)
                flat_ref.append(t.reshape(-1))
            flat_ref = torch.cat(flat_ref, dim=0) if flat_ref else torch.zeros(1, device=ref_device, dtype=ref_dtype)
            # -------------------- (C) filter --------------------
            def _flatten_client(cd):
                vecs = []
                for key, val in global_state_dict.items():
                    if (not val.is_floating_point()) or ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key):
                        continue
                    t = cd[key]
                    if t.device != ref_device or t.dtype != ref_dtype:
                        t = t.to(device=ref_device, dtype=ref_dtype)
                    vecs.append(t.reshape(-1))
                return torch.cat(vecs, dim=0) if vecs else torch.zeros(1, device=ref_device, dtype=ref_dtype)
            sims = []
            flat_ref_norm = flat_ref.norm(p=2)
            for cd in client_state_dicts:
                v = _flatten_client(cd)
                a_norm = v.norm(p=2)
                b_norm = flat_ref_norm
                s = (v @ flat_ref) / (a_norm * b_norm + 1e-12) if (a_norm.item() > 0 and b_norm.item() > 0) else torch.tensor(0.0, device=ref_device, dtype=ref_dtype)
                sims.append(float(s.item()))
            sims_t   = torch.tensor(sims, device=ref_device, dtype=ref_dtype)
            mean_s   = float(sims_t.mean().item())
            median_s = float(torch.median(sims_t).item())
            std_s    = float(sims_t.std(unbiased=False).item())
            keep = [True] * len(client_state_dicts)
            if std_s > 0.0:
                if mean_s > median_s:
                    thr = median_s + eta * std_s
                    for i, s in enumerate(sims):
                        if s > thr:
                            keep[i] = False
                    fl_logger(f"[+] Filtering high-tail: sim > {thr:.6f}", server_logger_filename)
                else:
                    thr = median_s - eta * std_s
                    for i, s in enumerate(sims):
                        if s < thr:
                            keep[i] = False
                    fl_logger(f"[+] Filtering low-tail: sim < {thr:.6f}", server_logger_filename)
            else:
                fl_logger("[=] std == 0, skip filtering (all sims identical).", server_logger_filename)
            kept_idx = [i for i, k in enumerate(keep) if k]
            if not kept_idx:
                fl_logger("[!] All clients filtered out; disable filtering this round.", server_logger_filename)
                kept_idx = list(range(len(client_state_dicts)))
            kept_sum = sum(base_weights[i] for i in kept_idx) + eps
            client_weights = [ (base_weights[i] / kept_sum) if i in kept_idx else 0.0 for i in range(len(base_weights)) ]
            sum_weights = sum(client_weights) + 1e-10
            client_weights_normalized = [w / sum_weights for w in client_weights]
        fl_logger(f"[+] Kept {len(kept_idx)}/{len(client_state_dicts)}; "
                  f"dropped: {[i for i in range(len(client_state_dicts)) if i not in kept_idx]}",
                  server_logger_filename)
        fl_logger(f"[+] Final normalized weights: {[float(w) for w in client_weights_normalized]}", server_logger_filename)
        # make global update
        global_state_dict = global_model.state_dict()
        global_update = {}
        for key, val in global_state_dict.items():
            if (not val.is_floating_point()) or ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key):
                global_update[key] = None
            else:
                acc_dtype = _acc_dtype_of(val, aggr_accum_fp32)
                global_update[key] = torch.zeros_like(val, dtype=acc_dtype, device=val.device)
        for client_idx, client_dict in enumerate(client_state_dicts):
            w = client_weights_normalized[client_idx]
            for key, acc in global_update.items():
                if acc is None:
                    continue
                delta_i = client_dict[key]
                if delta_i.device != acc.device or delta_i.dtype != acc.dtype:
                    delta_i = delta_i.to(device=acc.device, dtype=acc.dtype)
                acc.add_(delta_i, alpha=w)
        for key, acc in global_update.items():
            if acc is None:
                continue
            gparam = global_state_dict[key]
            acc.mul_(learning_rate)
            if acc.dtype != gparam.dtype:
                acc = acc.to(dtype=gparam.dtype)
            gparam.add_(acc)

        global_model.load_state_dict(global_state_dict)
    fl_logger(f"[=] FAIR aggregation finished!", server_logger_filename)
    return global_model, variance, cosine_similarity, worst_10, best_10

def q_fel(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32):
    fl_logger(f"[=] Applying q_fel aggregation (q={q} with f={client_freq})", server_logger_filename)
    fl_logger(f"[+] Client ratio: {client_ratio}", server_logger_filename)
    variance, cosine_similarity, worst_10, best_10 = fairness_cal(client_test_acc, server_logger_filename)
    with torch.no_grad():
        raw_weights = [np.float_power(loss + 1e-10, q) for loss, freq in zip(client_last_loss, client_freq)]
        sum_raw = sum(raw_weights) + 1e-10
        normalized_weights = [w / sum_raw for w in raw_weights]
        # add ratio
        client_ratio_w = [x / sum(client_ratio) for x in client_ratio]
        client_weights = [w * r for w, r in zip(normalized_weights, client_ratio_w)]
        sum_weights = sum(client_weights) + 1e-10
        client_weights_normalized = [w / sum_weights for w in client_weights]
        fl_logger(f"[+] Final normalized weights: {[float(w) for w in client_weights_normalized]}", server_logger_filename)
        # make global update
        global_state_dict = global_model.state_dict()
        global_update = {}
        for key, val in global_state_dict.items():
            if (not val.is_floating_point()) or ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key):
                global_update[key] = None
            else:
                acc_dtype = _acc_dtype_of(val, aggr_accum_fp32)
                global_update[key] = torch.zeros_like(val, dtype=acc_dtype, device=val.device)
        for client_idx, client_dict in enumerate(client_state_dicts):
            w = client_weights_normalized[client_idx]
            for key, acc in global_update.items():
                if acc is None:
                    continue
                delta_i = client_dict[key]
                if delta_i.device != acc.device or delta_i.dtype != acc.dtype:
                    delta_i = delta_i.to(device=acc.device, dtype=acc.dtype)
                acc.add_(delta_i, alpha=w)
        for key, acc in global_update.items():
            if acc is None:
                continue
            gparam = global_state_dict[key]
            acc.mul_(learning_rate)
            if acc.dtype != gparam.dtype:
                acc = acc.to(dtype=gparam.dtype)
            gparam.add_(acc)

        global_model.load_state_dict(global_state_dict)
    fl_logger(f"[=] q_fel aggregation finished!", server_logger_filename)
    return global_model, variance, cosine_similarity, worst_10, best_10

def f_div(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32):
    fl_logger(f"[=] Applying f_div aggregation (q={q} with f={client_freq})", server_logger_filename)
    fl_logger(f"[+] Client ratio: {client_ratio}", server_logger_filename)
    variance, cosine_similarity, worst_10, best_10 = fairness_cal(client_test_acc, server_logger_filename)
    with torch.no_grad():
        raw_weights = [np.float_power(loss + 1e-10, q) / freq for loss, freq in zip(client_last_loss, client_freq)]
        sum_raw = sum(raw_weights) + 1e-10
        normalized_weights = [w / sum_raw for w in raw_weights]
        # add ratio
        client_ratio_w = [x / sum(client_ratio) for x in client_ratio]
        client_weights = [w * r for w, r in zip(normalized_weights, client_ratio_w)]
        sum_weights = sum(client_weights) + 1e-10
        client_weights_normalized = [w / sum_weights for w in client_weights]
        fl_logger(f"[+] Final normalized weights: {[float(w) for w in client_weights_normalized]}", server_logger_filename)
        # make global update
        global_state_dict = global_model.state_dict()
        global_update = {}
        for key, val in global_state_dict.items():
            if (not val.is_floating_point()) or ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key):
                global_update[key] = None
            else:
                acc_dtype = _acc_dtype_of(val, aggr_accum_fp32)
                global_update[key] = torch.zeros_like(val, dtype=acc_dtype, device=val.device)
        for client_idx, client_dict in enumerate(client_state_dicts):
            w = client_weights_normalized[client_idx]
            for key, acc in global_update.items():
                if acc is None:
                    continue
                delta_i = client_dict[key]
                if delta_i.device != acc.device or delta_i.dtype != acc.dtype:
                    delta_i = delta_i.to(device=acc.device, dtype=acc.dtype)
                acc.add_(delta_i, alpha=w)
        for key, acc in global_update.items():
            if acc is None:
                continue
            gparam = global_state_dict[key]
            acc.mul_(learning_rate)
            if acc.dtype != gparam.dtype:
                acc = acc.to(dtype=gparam.dtype)
            gparam.add_(acc)

        global_model.load_state_dict(global_state_dict)
    fl_logger(f"[=] f_div aggregation finished!", server_logger_filename)
    return global_model, variance, cosine_similarity, worst_10, best_10
  
def f_mul(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32):
    fl_logger(f"[=] Applying f_mul aggregation (q={q} with f={client_freq})", server_logger_filename)
    fl_logger(f"[+] Client ratio: {client_ratio}", server_logger_filename)
    variance, cosine_similarity, worst_10, best_10 = fairness_cal(client_test_acc, server_logger_filename)
    with torch.no_grad():
        raw_weights = [np.float_power(loss + 1e-10, q / freq) for loss, freq in zip(client_last_loss, client_freq)]
        sum_raw = sum(raw_weights) + 1e-10
        normalized_weights = [w / sum_raw for w in raw_weights]
        # add ratio
        client_ratio_w = [x / sum(client_ratio) for x in client_ratio]
        client_weights = [w * r for w, r in zip(normalized_weights, client_ratio_w)]
        sum_weights = sum(client_weights) + 1e-10
        client_weights_normalized = [w / sum_weights for w in client_weights]
        fl_logger(f"[+] Final normalized weights: {[float(w) for w in client_weights_normalized]}", server_logger_filename)
        # make global update
        global_state_dict = global_model.state_dict()
        global_update = {}
        for key, val in global_state_dict.items():
            if (not val.is_floating_point()) or ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key):
                global_update[key] = None
            else:
                acc_dtype = _acc_dtype_of(val, aggr_accum_fp32)
                global_update[key] = torch.zeros_like(val, dtype=acc_dtype, device=val.device)
        for client_idx, client_dict in enumerate(client_state_dicts):
            w = client_weights_normalized[client_idx]
            for key, acc in global_update.items():
                if acc is None:
                    continue
                delta_i = client_dict[key]
                if delta_i.device != acc.device or delta_i.dtype != acc.dtype:
                    delta_i = delta_i.to(device=acc.device, dtype=acc.dtype)
                acc.add_(delta_i, alpha=w)
        for key, acc in global_update.items():
            if acc is None:
                continue
            gparam = global_state_dict[key]
            acc.mul_(learning_rate)
            if acc.dtype != gparam.dtype:
                acc = acc.to(dtype=gparam.dtype)
            gparam.add_(acc)

        global_model.load_state_dict(global_state_dict)
    fl_logger(f"[=] f_mul aggregation finished!", server_logger_filename)
    return global_model, variance, cosine_similarity, worst_10, best_10

def f_add(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32):
    fl_logger(f"[=] Applying f_add aggregation (q={q} with f={client_freq})", server_logger_filename)
    fl_logger(f"[+] Client ratio: {client_ratio}", server_logger_filename)
    variance, cosine_similarity, worst_10, best_10 = fairness_cal(client_test_acc, server_logger_filename)
    with torch.no_grad():
        raw_weights = [np.float_power(loss + 1e-10, q + 1.0 / freq) for loss, freq in zip(client_last_loss, client_freq)]
        sum_raw = sum(raw_weights) + 1e-10
        normalized_weights = [w / sum_raw for w in raw_weights]
        # add ratio
        client_ratio_w = [x / sum(client_ratio) for x in client_ratio]
        client_weights = [w * r for w, r in zip(normalized_weights, client_ratio_w)]
        sum_weights = sum(client_weights) + 1e-10
        client_weights_normalized = [w / sum_weights for w in client_weights]
        fl_logger(f"[+] Final normalized weights: {[float(w) for w in client_weights_normalized]}", server_logger_filename)
        # make global update
        global_state_dict = global_model.state_dict()
        global_update = {}
        for key, val in global_state_dict.items():
            if (not val.is_floating_point()) or ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key):
                global_update[key] = None
            else:
                acc_dtype = _acc_dtype_of(val, aggr_accum_fp32)
                global_update[key] = torch.zeros_like(val, dtype=acc_dtype, device=val.device)
        for client_idx, client_dict in enumerate(client_state_dicts):
            w = client_weights_normalized[client_idx]
            for key, acc in global_update.items():
                if acc is None:
                    continue
                delta_i = client_dict[key]
                if delta_i.device != acc.device or delta_i.dtype != acc.dtype:
                    delta_i = delta_i.to(device=acc.device, dtype=acc.dtype)
                acc.add_(delta_i, alpha=w)
        for key, acc in global_update.items():
            if acc is None:
                continue
            gparam = global_state_dict[key]
            acc.mul_(learning_rate)
            if acc.dtype != gparam.dtype:
                acc = acc.to(dtype=gparam.dtype)
            gparam.add_(acc)

        global_model.load_state_dict(global_state_dict)
    fl_logger(f"[=] f_add aggregation finished!", server_logger_filename)
    return global_model, variance, cosine_similarity, worst_10, best_10

def server_aggregation(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename=None, mode="fedavg", aggr_accum_fp32=True, top_k=None, last_loss=None):
    fl_logger(f"[=] Aggregation Process...", server_logger_filename)
    if mode == "q_fel":
        global_model, variance, cosine_similarity, worst_10, best_10 = q_fel(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32)
    elif mode == "f_div":
        global_model, variance, cosine_similarity, worst_10, best_10 = f_div(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32) # fusion div
    elif mode == "f_mul":
        global_model, variance, cosine_similarity, worst_10, best_10 = f_mul(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32) # fusion mul
    elif mode == "f_add":
        global_model, variance, cosine_similarity, worst_10, best_10 = f_add(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32) # fusion add
    elif mode == "afl":
        global_model, variance, cosine_similarity, worst_10, best_10 = afl(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32, top_k=top_k) # afl
    elif mode == "fair":
        global_model, variance, cosine_similarity, worst_10, best_10 = fair(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, client_freq, q, learning_rate, server_logger_filename, aggr_accum_fp32, global_ref_loss=last_loss) # fair
    else:
        global_model, variance, cosine_similarity, worst_10, best_10 = fedavg(global_model, client_state_dicts, client_last_loss, client_test_acc, client_ratio, client_interrupt_flags, learning_rate, server_logger_filename, aggr_accum_fp32)
        fl_logger(f"[+] Using default fedavg!", server_logger_filename)
    fl_logger(f"[=] Aggregation Finished!", server_logger_filename)
    return global_model, variance, cosine_similarity, worst_10, best_10