import random
import torch
import numpy as np
from chen2026spfere.utils.server_logger import fl_logger

def set_seed(seed=42, use_cuda=True, logger_filename=None):
    fl_logger(f"[INFO] set device seed: {seed}", logger_filename)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        fl_logger("[WARNING] Running without GPU, skipping CUDA-specific seed settings.", logger_filename)