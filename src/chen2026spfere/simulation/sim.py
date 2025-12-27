from __future__ import annotations
import datetime, argparse
import torch

from chen2026spfere.utils.set_seed import set_seed
from chen2026spfere.utils.low_dtype_casting import resolve_dtype, cast_non_norm_only
from chen2026spfere.utils.server_logger import fl_logger
from chen2026spfere.utils.model_map import MODEL_MAP
from chen2026spfere.simulation.sim_sd_op import sd_clone
from chen2026spfere.simulation.sim_device_data_gen import generate_devices_data
from chen2026spfere.simulation.sim_sync import run_global
from chen2026spfere.simulation.sim_load_data import build_loaders_with_legacy_fn, load_test_data
from chen2026spfere.simulation.sim_config import *
# feddrop, when use uncomment
# from sim_feddrop import count_conv_fc_params

# better logger
import wandb

# time stamp
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
server_logger_filename = f"simlog_{timestamp_str}/server_log.log"
client_logger_filename = f"simlog_{timestamp_str}/client_log.log"
wb_logger_name = f"simlog_{timestamp_str}"
SIM_EXTRA_DATA_SAVE_ROOT = f"simlog_{timestamp_str}/sim_runs"

# argparse
def str2bool(v):
    vl = v.lower()
    if vl in ("true", "false"):
        return vl == "true"
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def get_args():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("--GPU_ID", type=str, default=GPU_ID)
    parser.add_argument("--USE_MODEL", type=str, choices=["SimpleCNN", "ShuffleCNN", "EfficientCNN", "resnet20", "SimpleCNN_FM", "EfficientCNN_DP"], default=USE_MODEL)
    parser.add_argument("--DATASET", type=str, choices=["cifar10", "svhn", "stl_10", "eurosat_rgb", "f_mnist"], default=DATASET)
    #
    parser.add_argument("--TRAIN_ROUND", type=int, default=TRAIN_ROUND)
    parser.add_argument("--CLIENT_NUMBER", type=int, default=CLIENT_NUMBER)
    parser.add_argument("--WAIT_UNTIL_M", type=int, default=WAIT_UNTIL_M)
    #
    parser.add_argument("--G_LR", type=float, default=G_LR)
    parser.add_argument("--AGGR_METHOD", type=str, choices=["fedavg", "q_fel", "f_div", "f_mul", "f_add", "afl", "fair"], default=AGGR_METHOD)
    parser.add_argument("--Q_FEL", type=float, default=Q_FEL)
    parser.add_argument("--TOP_K", type=float, default=TOP_K)
    #
    parser.add_argument("--CLIENT_BATCH_SIZE", type=int, default=CLIENT_BATCH_SIZE)
    parser.add_argument("--CLIENT_EPOCHS", type=int, default=CLIENT_EPOCHS)
    parser.add_argument("--TRAIN_LOADER_MODE", type=str, choices=["avg", "dirichlet"], default=TRAIN_LOADER_MODE)
    parser.add_argument("--CLIENT_ALPHA_LOW", type=float, default=CLIENT_ALPHA_LOW)
    parser.add_argument("--CLIENT_ALPHA_HIGH", type=float, default=CLIENT_ALPHA_HIGH)
    #
    parser.add_argument("--CLIENT_USE_OPTIMIZER", type=str, default=CLIENT_USE_OPTIMIZER)
    parser.add_argument("--CLIENT_LR", type=float, default=CLIENT_LR)
    parser.add_argument("--CLIENT_LR_GAMMA", type=float, default=CLIENT_LR_GAMMA)
    #
    parser.add_argument("--SERVER_BATCH_SIZE", type=int, default=SERVER_BATCH_SIZE)
    parser.add_argument("--TEST_SEED", type=int, default=TEST_SEED)
    parser.add_argument("--TEST_LOADER_MODE", type=str, choices=["avg", "dirichlet"], default=TEST_LOADER_MODE)
    parser.add_argument("--TEST_ALPHA", type=float, default=TEST_ALPHA)
    #
    parser.add_argument("--PRECISION_COMPRESSION", type=str2bool, default=PRECISION_COMPRESSION)
    parser.add_argument("--PRECISION_DTYPE", type=str, default=PRECISION_DTYPE)
    parser.add_argument("--KEEP_NORM_FP32", type=str2bool, default=KEEP_NORM_FP32)
    parser.add_argument("--AGGR_ACCUM_FP32", type=str2bool, default=AGGR_ACCUM_FP32)
    #
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    wb_logger = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=WB_ENTITY,
        # Set the wandb project where this run will be logged.
        project=WB_PROJECT,
        name=wb_logger_name,
        # Track hyperparameters and run metadata.
        config={
            "GPU_ID": args.GPU_ID,
            "USE_MODEL": args.USE_MODEL,
            "DATASET": args.DATASET,
            #
            "TRAIN_ROUND": args.TRAIN_ROUND,
            "CLIENT_NUMBER": args.CLIENT_NUMBER,
            "WAIT_UNTIL_M": args.WAIT_UNTIL_M,
            #
            "G_LR": args.G_LR,
            "AGGR_METHOD": args.AGGR_METHOD,
            "Q_FEL": args.Q_FEL,
            "TOP_K": args.TOP_K,
            #
            "CLIENT_BATCH_SIZE": args.CLIENT_BATCH_SIZE,
            "CLIENT_EPOCHS": args.CLIENT_EPOCHS,
            "TRAIN_LOADER_MODE": args.TRAIN_LOADER_MODE,
            "CLIENT_ALPHA_LOW": args.CLIENT_ALPHA_LOW,
            "CLIENT_ALPHA_HIGH": args.CLIENT_ALPHA_HIGH,
            #
            "CLIENT_USE_OPTIMIZER": args.CLIENT_USE_OPTIMIZER,
            "CLIENT_LR": args.CLIENT_LR,
            "CLIENT_LR_GAMMA": args.CLIENT_LR_GAMMA,
            #
            "SERVER_BATCH_SIZE": args.SERVER_BATCH_SIZE,
            "TEST_SEED": args.TEST_SEED,
            "TEST_LOADER_MODE": args.TEST_LOADER_MODE,
            "TEST_ALPHA": args.TEST_ALPHA,
            #
            "PRECISION_COMPRESSION": args.PRECISION_COMPRESSION,
            "PRECISION_DTYPE": args.PRECISION_DTYPE,
            "KEEP_NORM_FP32": args.KEEP_NORM_FP32,
            "AGGR_ACCUM_FP32": args.AGGR_ACCUM_FP32,
        },
    )
    
    fl_logger(f"=============== Configs ===============", server_logger_filename)
    fl_logger(f"[+] GPU_ID = {args.GPU_ID}", server_logger_filename)
    fl_logger(f"[+] USE_MODEL = {args.USE_MODEL}", server_logger_filename)
    fl_logger(f"[+] DATASET = {args.DATASET}", server_logger_filename)
    fl_logger(f"---------------------------------------", server_logger_filename)
    fl_logger(f"[+] TRAIN_ROUND = {args.TRAIN_ROUND}", server_logger_filename)
    fl_logger(f"[+] CLIENT_NUMBER = {args.CLIENT_NUMBER}", server_logger_filename)
    fl_logger(f"[+] WAIT_UNTIL_M = {args.WAIT_UNTIL_M}", server_logger_filename)
    fl_logger(f"---------------------------------------", server_logger_filename)
    fl_logger(f"[+] G_LR = {args.G_LR}", server_logger_filename)
    fl_logger(f"[+] AGGR_METHOD = {args.AGGR_METHOD}", server_logger_filename)
    fl_logger(f"[+] Q_FEL = {args.Q_FEL}", server_logger_filename)
    fl_logger(f"[+] TOP_K = {args.TOP_K}", server_logger_filename)
    fl_logger(f"---------------------------------------", server_logger_filename)
    fl_logger(f"[+] CLIENT_BATCH_SIZE = {args.CLIENT_BATCH_SIZE}", server_logger_filename)
    fl_logger(f"[+] CLIENT_EPOCHS = {args.CLIENT_EPOCHS}", server_logger_filename)
    fl_logger(f"[+] TRAIN_LOADER_MODE = {args.TRAIN_LOADER_MODE}", server_logger_filename)
    fl_logger(f"[+] CLIENT_ALPHA_LOW = ({args.CLIENT_ALPHA_LOW}, {args.CLIENT_ALPHA_HIGH})", server_logger_filename)
    fl_logger(f"---------------------------------------", server_logger_filename)
    fl_logger(f"[+] CLIENT_USE_OPTIMIZER = {args.CLIENT_USE_OPTIMIZER}", server_logger_filename)
    fl_logger(f"[+] CLIENT_LR = {args.CLIENT_LR}", server_logger_filename)
    fl_logger(f"[+] CLIENT_LR_GAMMA = {args.CLIENT_LR_GAMMA}", server_logger_filename)
    fl_logger(f"---------------------------------------", server_logger_filename)
    fl_logger(f"[+] SERVER_BATCH_SIZE = {args.SERVER_BATCH_SIZE}", server_logger_filename)
    fl_logger(f"[+] TEST_SEED = {args.TEST_SEED}", server_logger_filename)
    fl_logger(f"[+] TEST_LOADER_MODE = {args.TEST_LOADER_MODE}", server_logger_filename)
    fl_logger(f"[+] TEST_ALPHA = {args.TEST_ALPHA}", server_logger_filename)
    fl_logger(f"---------------------------------------", server_logger_filename)
    fl_logger(f"[+] PRECISION_COMPRESSION = {args.PRECISION_COMPRESSION}", server_logger_filename)
    fl_logger(f"[+] PRECISION_DTYPE = {args.PRECISION_DTYPE}", server_logger_filename)
    fl_logger(f"[+] KEEP_NORM_FP32 = {args.KEEP_NORM_FP32}", server_logger_filename)
    fl_logger(f"[+] AGGR_ACCUM_FP32 = {args.AGGR_ACCUM_FP32}", server_logger_filename)
    fl_logger(f"---------------------------------------", server_logger_filename)
    fl_logger(f"[+] WB_PROJECT = {WB_PROJECT}", server_logger_filename)
    fl_logger(f"=============== Configs ===============", server_logger_filename)
    
    globals().update({
        "GPU_ID": args.GPU_ID,
        "USE_MODEL": args.USE_MODEL,
        "DATASET": args.DATASET,
        #
        "TRAIN_ROUND": args.TRAIN_ROUND,
        "CLIENT_NUMBER": args.CLIENT_NUMBER,
        "WAIT_UNTIL_M": args.WAIT_UNTIL_M,
        #
        "G_LR": args.G_LR,
        "AGGR_METHOD": args.AGGR_METHOD,
        "Q_FEL": args.Q_FEL,
        "TOP_K": args.TOP_K,
        #
        "CLIENT_BATCH_SIZE": args.CLIENT_BATCH_SIZE,
        "CLIENT_EPOCHS": args.CLIENT_EPOCHS,
        "TRAIN_LOADER_MODE": args.TRAIN_LOADER_MODE,
        "CLIENT_ALPHA_LOW": args.CLIENT_ALPHA_LOW,
        "CLIENT_ALPHA_HIGH": args.CLIENT_ALPHA_HIGH,
        #
        "CLIENT_USE_OPTIMIZER": args.CLIENT_USE_OPTIMIZER,
        "CLIENT_LR": args.CLIENT_LR,
        "CLIENT_LR_GAMMA": args.CLIENT_LR_GAMMA,
        #
        "SERVER_BATCH_SIZE": args.SERVER_BATCH_SIZE,
        "TEST_SEED": args.TEST_SEED,
        "TEST_LOADER_MODE": args.TEST_LOADER_MODE,
        "TEST_ALPHA": args.TEST_ALPHA,
        #
        "PRECISION_COMPRESSION": args.PRECISION_COMPRESSION,
        "PRECISION_DTYPE": args.PRECISION_DTYPE,
        "KEEP_NORM_FP32": args.KEEP_NORM_FP32,
        "AGGR_ACCUM_FP32": args.AGGR_ACCUM_FP32,
    })

    DEVICE = torch.device(
        "cpu" if GPU_ID == "cpu"
        else f"cuda:{GPU_ID}" if torch.cuda.is_available() 
            and GPU_ID.isdigit() 
            and int(GPU_ID) < torch.cuda.device_count()
        else "cuda:0" if torch.cuda.is_available()
        else "cpu"
    )
    
    devices_data_select_ext = generate_devices_data(
        n=CLIENT_NUMBER,
        ratio_range=(0.1, 0.5),
        alpha_range=(CLIENT_ALPHA_LOW, CLIENT_ALPHA_HIGH),
        delay_range=(0.5, 6),
        gen_seed=42
    )
    fl_logger(f"[INFO] cfg: {devices_data_select_ext}", server_logger_filename)
    
    set_seed(seed=42, use_cuda=True, logger_filename=server_logger_filename)
    model_cls = MODEL_MAP[args.USE_MODEL]
    if DATASET == "f_mnist":
        if USE_MODEL == "resnet20":
            init_model = model_cls(in_channels=1)
        elif USE_MODEL == "SimpleCNN_FM":
            init_model = model_cls()
        else:
            fl_logger(f"[ERROR] f_mnist dataset must use SimpleCNN_FM or resnet20", server_logger_filename)
            exit(0)
    else:
        init_model = model_cls()
    
    base_sd0 = sd_clone(init_model.state_dict())
    
    def model_ctor(dropout_p=0):
        if DATASET == "f_mnist":
            if USE_MODEL == "resnet20":
                m = model_cls(in_channels=1)
            elif USE_MODEL == "SimpleCNN_FM":
                m = model_cls()
            else:
                fl_logger(f"[ERROR] f_mnist dataset must use SimpleCNN_FM or resnet20", server_logger_filename)
                exit(0)
        else:
            if USE_MODEL == "EfficientCNN_DP":
                m = model_cls(dropout_p=dropout_p)
            else:
                m = model_cls()
        m.load_state_dict(sd_clone(base_sd0), strict=True)
        m.to(DEVICE)
        if PRECISION_COMPRESSION:
            low_dtype=resolve_dtype(PRECISION_DTYPE)
            if KEEP_NORM_FP32 and low_dtype != torch.float32:
                cast_non_norm_only(m, low_dtype)
            else:
                m.to(dtype=low_dtype)
        else:
            m.to(dtype=torch.float32) # no-op
        return m
    
    temp_model = model_ctor()
    a_conv = None
    a_fc = None
    # feddrop, when use uncomment
    # a_conv, a_fc = count_conv_fc_params(temp_model)

    client_train_loaders, client_test_loaders, ratio_map, delay_map = build_loaders_with_legacy_fn(
        cfgs=devices_data_select_ext, batch_size=CLIENT_BATCH_SIZE, base_seed=42,
        data_name=DATASET, train_loader_mode=TRAIN_LOADER_MODE,
        client_logger_filename=client_logger_filename
    )
    
    test_loader, calib_loader = load_test_data(data = DATASET, batch_size=SERVER_BATCH_SIZE, ratio=1.0, alpha=TEST_ALPHA, seed=TEST_SEED, mode=TEST_LOADER_MODE, server_logger_filename=server_logger_filename)

    groups = None
    need_intra = None
    number_of_aggregation = 0
    last_loss = None

    # set seed again to avoid any previous influence to training
    set_seed(seed=42, use_cuda=True, logger_filename=server_logger_filename) 

    number_of_aggregation = run_global(
        T=TRAIN_ROUND,
        model_ctor=model_ctor,
        groups=groups,
        need_intra=need_intra,
        train_loaders=client_train_loaders,
        client_test_loaders=client_test_loaders,
        test_loader=test_loader,
        calib_loader=calib_loader,
        ratio_map=ratio_map,
        delay_map=delay_map,
        device=DEVICE,
        save_root=SIM_EXTRA_DATA_SAVE_ROOT,
        wb_logger=wb_logger,
        number_of_aggregation=number_of_aggregation,
        last_loss=last_loss,
        a_conv=a_conv, 
        a_fc=a_fc,
        sim_timebase=SIM_COLD_TIME_BASE,
        sim_alpha=SIM_COLD_ALPHA,
        wait_until_m=WAIT_UNTIL_M,
        g_lr=G_LR,
        aggr_method=AGGR_METHOD,
        q_fel=Q_FEL,
        top_k=TOP_K,
        client_epochs=CLIENT_EPOCHS,
        client_use_optimizer=CLIENT_USE_OPTIMIZER,
        client_lr=CLIENT_LR,
        client_lr_gamma=CLIENT_LR_GAMMA,
        precision_compression=PRECISION_COMPRESSION,
        precision_dtype=PRECISION_DTYPE,
        aggr_accum_fp32=AGGR_ACCUM_FP32,
        server_logger_filename=server_logger_filename,
        client_logger_filename=client_logger_filename,
    )
    fl_logger(f"[+] Number of aggregation: {number_of_aggregation}", server_logger_filename)
    wb_logger.log({"number_of_aggregation": number_of_aggregation})
    wb_logger.finish()
