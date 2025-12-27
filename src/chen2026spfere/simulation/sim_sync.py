import json, csv, math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable
import torch
from torch import nn
from torch.utils.data import DataLoader

from chen2026spfere.utils.server_logger import fl_logger
from chen2026spfere.simulation.sim_aggregation import legacy_aggregate_apply
from chen2026spfere.simulation.sim_sd_op import sd_clone, sd_sub
from chen2026spfere.simulation.sim_timeline import _timeline_append, _timeline_pick_sd_at
from chen2026spfere.simulation.sim_train import local_train_once
from chen2026spfere.simulation.sim_eval import server_eval_once

def sim_cold_start_profile_group_time(
    ratio_map: Dict[int, float],
    delay_map: Dict[int, float],
    timebase: float = 2048.0,
    alpha: float = 32.0,
    wait_until_m: int = 1,
    server_logger_filename: str = None,
) -> Tuple[Dict[int, float], Dict[int, List[int]], Dict[int, int]]:

    eps: float = 1e-9
    cids = sorted(set(ratio_map.keys()) & set(delay_map.keys()))
    if not cids:
        return {}, {}, {}

    compute_ms: Dict[int, float] = {}
    basis: Dict[int, float] = {}
    for cid in cids:
        comp = ratio_map[cid] * timebase + alpha
        compute_ms[cid] = comp
        basis[cid] = round(comp * delay_map[cid], 2)

    items = sorted(basis.items(), key=lambda x: (x[1], x[0]))  # [(cid, base), ...]
    groups_out: Dict[int, List[int]] = {}
    for g, start in enumerate(range(0, len(items), wait_until_m)):
        chunk = items[start:start + wait_until_m]
        groups_out[g] = [cid for cid, _ in chunk]

    group_bases: Dict[int, float] = {}
    for g, cid_list in groups_out.items():
        mx = max(basis[cid] for cid in cid_list) if cid_list else 0.0
        group_bases[g] = round(mx, 2)

    last_g = max(group_bases.keys())
    last_base = group_bases[last_g]
    need_intra: Dict[int, int] = {}
    for g, base in group_bases.items():
        r = (last_base) / max(base, eps)
        need_intra[g] = max(int(math.floor(r + 1e-9)), 1)

    group_time_ms: Dict[int, float] = {}
    for g, cid_list in groups_out.items():
        mx = max(basis[cid] for cid in cid_list) if cid_list else 0.0
        group_time_ms[g] = round(mx, 2)

    fl_logger(f"[CS] group_time_ms: {group_time_ms}, groups: {groups_out}, need_intra: {need_intra}, computing_time_ms: {basis}", server_logger_filename)
    return group_time_ms, groups_out, need_intra, basis

def build_schedule_by_time(
    need_intra: Dict[int, int],
    group_time_ms: Dict[int, float]
) -> Tuple[List[Tuple[float, int, int]], List[Dict[str, Any]]]:
    """
    返回：
    - schedule: [(end_ms, group_id, k)], k=1..need_intra[g]；以 end_ms 升序执行
    - gantt_rows: [{"group":g,"k":k,"start_ms":(k-1)*t_g,"end_ms":k*t_g}]
    """
    schedule: List[Tuple[float, int, int]] = []
    gantt_rows: List[Dict[str, Any]] = []
    for g, kmax in need_intra.items():
        t_g = float(group_time_ms[g])
        for k in range(1, int(kmax) + 1):
            start_ms = (k - 1) * t_g
            end_ms = k * t_g
            schedule.append((end_ms, g, k))
            gantt_rows.append({"group": int(g), "k": int(k), "start_ms": float(start_ms), "end_ms": float(end_ms)})
    schedule.sort(key=lambda x: (x[0], x[1], x[2]))
    return schedule, gantt_rows


def compute_client_p_by_param_share(
    group_time_ms: Dict[int, float],
    groups: Dict[int, List[int]],
    computing_time_ms: Dict[int, float],
    alpha_conv: float,
    alpha_fc: float,
    eta: float = 1.0,
    eps: float = 1e-12,
    server_logger_filename: str = None,
) -> Dict[int, float]:
    
    client_to_group = {cid: g for g, cids in groups.items() for cid in cids}

    p_map: Dict[int, float] = {}
    for k, t_full in computing_time_ms.items():
        g = client_to_group.get(k)
        if g is None or t_full <= 0 or alpha_fc <= eps:
            p_map[k] = 0.0
            continue

        T_eff = eta * group_time_ms[g]

        T_conv_k = alpha_conv * t_full
        T_fc_k   = alpha_fc   * t_full

        # p = 1 - sqrt( clamp( (T - T_conv)/T_fc , 0, 1 ) )
        ratio = (T_eff - T_conv_k) / max(T_fc_k, eps)
        ratio = 0.0 if ratio < 0.0 else (1.0 if ratio > 1.0 else ratio)
        p = 1.0 - (ratio ** 0.5)

        # p ∈ [0, 1)
        p_map[k] = 0.0 if p < 0.0 else min(p, 1.0 - 1e-6)
    fl_logger(f"p_map: {p_map}",server_logger_filename)
    return p_map

def run_one_global_round_real_timeline(
    T_rounds: int,
    round_id: int,
    global_sd: Dict[str, torch.Tensor],
    groups: Dict[int, List[int]],
    need_intra: Dict[int, int],
    model_ctor: Callable[[], nn.Module],
    train_loaders: Dict[int, DataLoader],
    client_test_loaders: Dict[int, DataLoader],
    test_loader: Dict[int, DataLoader],
    calib_loader: Dict[int, DataLoader],
    device: torch.device,
    ratio_map: Dict[int, float],
    delay_map: Dict[int, float],
    save_dir: str | Path = None,
    *,
    profile_cache: Dict[int, float] | None = None,
    running_time_cache: Dict[int, float] | None = None,
    
    wb_logger=None,
    number_of_aggregation=None,
    last_loss=None,
    a_conv=None, # feddrop usage
    a_fc=None, # feddrop usage
    sim_timebase=2048.0,
    sim_alpha=32.0,
    wait_until_m=1,
    g_lr=1.0,
    aggr_method=None,
    q_fel=1,
    top_k=None,
    client_epochs=2,
    client_use_optimizer=None,
    client_lr=None,
    client_lr_gamma=None,
    precision_compression=False,
    precision_dtype=None,
    aggr_accum_fp32=None,
    server_logger_filename:str=None,
    client_logger_filename:str=None,
) -> Tuple[Dict[str, torch.Tensor], Dict[int, float], Dict[int, List[int]], Dict[int, int]]:
    save_path: Path | None = Path(save_dir) if save_dir is not None else None
    if save_path is not None:
        (save_path / f"G{round_id}").mkdir(parents=True, exist_ok=True)
    
    if round_id == 0: # do eval, no train for global round 0
        _, test_loss = server_eval_once(
            global_sd=global_sd, model_ctor=model_ctor, device=device, variance=None, cosine_similarity=None, worst_10=None, best_10=None,
            test_loader=test_loader, calib_loader=calib_loader, global_round=round_id,
            global_stage=f"g0", wb_logger=wb_logger, wb_logger_enable=True,
            precision_compression=precision_compression, precision_dtype=precision_dtype,
            server_logger_filename=server_logger_filename
        )
        last_loss = test_loss
        return global_sd, None, groups, need_intra, number_of_aggregation, last_loss, None
    
    timeline: List[Tuple[float, Dict[str, torch.Tensor]]] = [(0.0, sd_clone(global_sd))]
    
    if profile_cache is None:
        group_time_ms, groups, need_intra, running_time_ms = sim_cold_start_profile_group_time(
            ratio_map=ratio_map, delay_map=delay_map,
            timebase=sim_timebase, alpha=sim_alpha,
            wait_until_m=wait_until_m, server_logger_filename=server_logger_filename
        )
        fl_logger(f"[CS] Cold start finished.", server_logger_filename)
        fl_logger(f"[ROUND] [Global Round {round_id}] Begins.", server_logger_filename)
    else:
        group_time_ms = dict(profile_cache)
        running_time_ms = dict(running_time_cache)
        fl_logger(f"[ROUND] [Global Round {round_id}] Begins.", server_logger_filename)

    if save_path is not None:
        (save_path / "coldstart_group_time_ms.json").write_text(
            json.dumps({int(k): float(v) for k, v in group_time_ms.items()}, indent=2)
        )

    # feddrop, when use uncomment
    # p_map = compute_client_p_by_param_share(
    #     group_time_ms=group_time_ms,
    #     groups=groups,
    #     computing_time_ms=running_time_ms,
    #     alpha_conv=a_conv,
    #     alpha_fc=a_fc
    # )
    
    schedule, gantt_rows = build_schedule_by_time(need_intra, group_time_ms)
    if save_path is not None:
        with open(save_path / "timeline_gantt.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["group", "k", "start_ms", "end_ms"])
            w.writeheader()
            for row in gantt_rows:
                w.writerow(row)

    records: List[Dict[str, Any]] = []
    group_batch_max: List[Dict[str, Any]] = []

    submit_models: Dict[int, List[Dict[str, Any]]] = {p: [] for p in groups}
    end_times_at_threshold: Dict[int, float] = {}

    for end_ms, g, k in schedule:
        start_ms = end_ms - float(group_time_ms[g])
        base_t_start, base_sd_start = _timeline_pick_sd_at(timeline, start_ms)
        base = sd_clone(base_sd_start)

        fl_logger(f"[RT][start] r{round_id} g{g} k{k} "
                f"start_ms={start_ms:.2f} end_ms={end_ms:.2f} "
                f"train_base_snap_t={base_t_start:.2f} "
                f"latest_snap_t={timeline[-1][0]:.2f}", server_logger_filename)
        fl_logger(f"[RT][start] r{round_id} g{g} k{k} "
                f"start_ms={start_ms:.2f} end_ms={end_ms:.2f} "
                f"train_base_snap_t={base_t_start:.2f} "
                f"latest_snap_t={timeline[-1][0]:.2f}", client_logger_filename)
        
        batch_updates, batch_models = [], []
        for cid in groups[g]:
            # feddrop, when use uncomment
            # fl_logger(f"dropout_p for cid[{cid}]: {p_map[cid]}", client_logger_filename)
            new_model, delta, stats = local_train_once(
                # feddrop, when use uncomment
                # model_ctor=(lambda: model_ctor(dropout_p=p_map[cid])), base_sd=base, cid=cid,
                model_ctor=model_ctor, base_sd=base, cid=cid,
                train_loader=train_loaders[cid], device=device,
                total_global_round=T_rounds,
                global_round_count=round_id, epochs=client_epochs,
                client_use_optimizer=client_use_optimizer, client_lr=client_lr,
                client_lr_gamma=client_lr_gamma,
                precision_compression=precision_compression,
                precision_dtype=precision_dtype,
                client_logger_filename=client_logger_filename
            )
            
            test_acc, _ = server_eval_once(
                    global_sd=new_model, model_ctor=model_ctor, device=device, variance=None, cosine_similarity=None, worst_10=None, best_10=None,
                    test_loader=client_test_loaders[cid], calib_loader=None, global_round=round_id,
                    global_stage=f"g{g}-k{k}-c{cid}", wb_logger=wb_logger, wb_logger_enable=False,
                    precision_compression=precision_compression, precision_dtype=precision_dtype,
                    server_logger_filename=server_logger_filename
                )
            
            rec = {
                "cid": cid, "delta": delta, "loss": stats["loss"], "acc": test_acc,
                "n_eff": stats["n_eff"], "compute_ms": stats["compute_ms"],
            }
            batch_updates.append(rec)
            batch_models.append(new_model)

        for rec in batch_updates:
            records.append({
                "cid": int(rec["cid"]), "group": int(g), "k": int(k),
                "compute_ms": float(rec.get("compute_ms", 0.0))
            })
        max_ms = max(float(rec.get("compute_ms", 0.0)) for rec in batch_updates) if batch_updates else 0.0
        group_batch_max.append({"group": int(g), "k": int(k), "max_ms": max_ms})

        if k >= int(need_intra[g]):
            submit_models[g] = [{"cid": u["cid"], "model": m, "loss": u["loss"], "acc": u["acc"], "n_eff": u["n_eff"]}
                                for u, m in zip(batch_updates, batch_models)]
            end_times_at_threshold[g] = float(end_ms)
            fl_logger(f"[RT][submit] r{round_id} g{g} k{k} "
                    f"submit_at_end_ms={end_ms:.2f} "
                    f"train_base_snap_t={base_t_start:.2f}", server_logger_filename)
            fl_logger(f"[RT][submit] r{round_id} g{g} k{k} "
                    f"submit_at_end_ms={end_ms:.2f} "
                    f"train_base_snap_t={base_t_start:.2f}", client_logger_filename)

        else:
            base_t_end, base_sd_end = _timeline_pick_sd_at(timeline, end_ms)
            freq_map = {int(u["cid"]): float(k) for u in batch_updates}
            new_global, variance, cosine_similarity, worst_10, best_10 = legacy_aggregate_apply(
                model_ctor=model_ctor, base_sd=base_sd_end, batch_recs=batch_updates,
                lr=g_lr, q=q_fel, ratio_map=ratio_map, freq_map=freq_map, device=device, last_loss=last_loss,
                aggr_method=aggr_method, aggr_accum_fp32=aggr_accum_fp32,
                top_k=top_k, server_logger_filename=server_logger_filename
            )
            _timeline_append(timeline, end_ms, new_global)
            global_sd = new_global

            fl_logger(f"[AGG-INTRA] r{round_id} g{g} k{k} "
                    f"agg_at_end_ms={end_ms:.2f} "
                    f"agg_base_snap_t={base_t_end:.2f} "
                    f"new_snap_t={end_ms:.2f}", server_logger_filename)
            number_of_aggregation+=1

            _, test_loss = server_eval_once(
                global_sd=new_global, model_ctor=model_ctor, device=device, variance=variance, cosine_similarity=cosine_similarity, worst_10=worst_10, best_10=best_10,
                test_loader=test_loader, calib_loader=calib_loader, global_round=round_id,
                global_stage=f"g{g}-k{k}", wb_logger=wb_logger, wb_logger_enable=False,
                precision_compression=precision_compression, precision_dtype=precision_dtype,
                server_logger_filename=server_logger_filename
            )
            last_loss = test_loss

    T_inter = max(end_times_at_threshold.values()) if end_times_at_threshold else 0.0
    base_t_inter, base_inter = _timeline_pick_sd_at(timeline, T_inter)

    fl_logger(f"[AGG-INTER][begin] r{round_id} T_inter={T_inter:.2f} "
            f"inter_base_snap_t={base_t_inter:.2f}", server_logger_filename)

    inter_batch, freq_map_inter = [], {}
    for g in groups:
        k_final = int(need_intra[g])
        for item in submit_models[g]:
            cid = int(item["cid"])
            d = sd_sub(item["model"], base_inter)  # 相对 inter 基线
            inter_batch.append({"cid": cid, "delta": d, "loss": float(item["loss"]), "acc": float(item["acc"]), "n_eff": int(item["n_eff"])})
            freq_map_inter[cid] = float(k_final)

    new_global, variance, cosine_similarity, worst_10, best_10 = legacy_aggregate_apply(
        model_ctor=model_ctor, base_sd=base_inter, batch_recs=inter_batch,
        lr=g_lr, q=q_fel, ratio_map=ratio_map, freq_map=freq_map_inter, device=device, last_loss=last_loss,
        aggr_method=aggr_method, aggr_accum_fp32=aggr_accum_fp32,
        top_k=top_k, server_logger_filename=server_logger_filename
    )
    _timeline_append(timeline, T_inter, new_global)
    global_sd = new_global

    fl_logger(f"[AGG-INTER][end] r{round_id} T_inter={T_inter:.2f} "
            f"new_snap_t={T_inter:.2f}", server_logger_filename)
    number_of_aggregation+=1
    
    _, test_loss = server_eval_once(
        global_sd=global_sd,
        model_ctor=model_ctor,
        device=device,
        variance=variance,
        cosine_similarity=cosine_similarity, worst_10=worst_10, best_10=best_10,
        test_loader=test_loader,
        calib_loader=calib_loader,
        global_round=round_id,
        global_stage="inter",
        wb_logger=wb_logger,
        wb_logger_enable=True,
        precision_compression=precision_compression,
        precision_dtype=precision_dtype,
        server_logger_filename=server_logger_filename
    )
    last_loss = test_loss

    if save_path is not None:
        csv_path = save_path / f"G{round_id}" / "timings_per_client.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cid", "group", "k", "compute_ms"])
            w.writeheader()
            for r in records:
                w.writerow(r)

        jsonl_path = save_path / f"G{round_id}" / "timings_group_batch_max.jsonl"
        with open(jsonl_path, "w") as f:
            for r in group_batch_max:
                f.write(json.dumps(r) + "\n")

    if records:
        total_clients = len({r["cid"] for r in records})
        total_batches = len(group_batch_max)
        fl_logger(f"[Global Round {round_id}] timings: "
                  f"clients={total_clients}, group-batches={total_batches}, "
                  , server_logger_filename)
        fl_logger(f"[ROUND] finish global round {round_id}/{T_rounds}", server_logger_filename)

    return global_sd, group_time_ms, groups, need_intra, number_of_aggregation, last_loss, running_time_ms

def run_global(
    T: int,
    model_ctor: Callable[[], nn.Module],
    groups: Dict[int, List[int]],
    need_intra: Dict[int, int],
    train_loaders: Dict[int, DataLoader],
    client_test_loaders: Dict[int, DataLoader],
    test_loader: Dict[int, DataLoader],
    calib_loader: Dict[int, DataLoader],
    ratio_map: Dict[int, float],
    delay_map: Dict[int, float],
    device: torch.device,
    save_root: str | Path = "sim_runs/run",
    *,
    wb_logger=None,
    number_of_aggregation=None,
    last_loss=None,
    a_conv=None, 
    a_fc=None,
    sim_timebase=2048.0,
    sim_alpha=32.0,
    wait_until_m=1,
    g_lr=1.0,
    aggr_method=None,
    q_fel=1,
    top_k=None,
    client_epochs=2,
    client_use_optimizer=None,
    client_lr=None,
    client_lr_gamma=None,
    precision_compression=False,
    precision_dtype=None,
    aggr_accum_fp32=None,
    server_logger_filename:str=None,
    client_logger_filename:str=None,
):
    model0 = model_ctor()
    global_sd = sd_clone(model0.state_dict())

    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    cache = None
    for t in range(T+1):
        (save_root / f"G{t}").mkdir(parents=True, exist_ok=True)
        global_sd, cache, groups, need_intra, number_of_aggregation, last_loss, running_time_cache = run_one_global_round_real_timeline(
            T_rounds=T,
            round_id=t,
            global_sd=global_sd,
            groups=groups,
            need_intra=need_intra,
            model_ctor=model_ctor,
            train_loaders=train_loaders,
            client_test_loaders=client_test_loaders,
            test_loader=test_loader,
            calib_loader=calib_loader,
            device=device,
            ratio_map=ratio_map,
            delay_map=delay_map,
            save_dir=save_root,
            profile_cache=(cache if t > 1 else None),
            running_time_cache=(running_time_cache if t > 1 else None),
            wb_logger=wb_logger,
            number_of_aggregation=number_of_aggregation,
            last_loss=last_loss,
            a_conv=a_conv,
            a_fc=a_fc,
            sim_timebase=sim_timebase,
            sim_alpha=sim_alpha,
            wait_until_m=wait_until_m,
            g_lr=g_lr,
            aggr_method=aggr_method,
            q_fel=q_fel,
            top_k=top_k,
            client_epochs=client_epochs,
            client_use_optimizer=client_use_optimizer,
            client_lr=client_lr,
            client_lr_gamma=client_lr_gamma,
            precision_compression=precision_compression,
            precision_dtype=precision_dtype,
            aggr_accum_fp32=aggr_accum_fp32,
            server_logger_filename=server_logger_filename,
            client_logger_filename=client_logger_filename
        )

    torch.save({"global_model": global_sd}, save_root / "W_T.pt")
    fl_logger(f"[+] SYNC finished. Final global saved at {save_root / 'W_T.pt'}", server_logger_filename)
    return number_of_aggregation