from torch.utils.data import DataLoader
from typing import Dict, Tuple
from chen2026spfere.server.server_load_data import load_data as load_test_data # will be imported in the main sim file
from chen2026spfere.client.client_load_data import load_data as load_train_data

def build_loaders_with_legacy_fn( # will be imported in the main sim file
    cfgs: Dict[str, dict],
    batch_size: int = 128,
    base_seed: int = 42,
    data_name: str = "cifar10",
    train_loader_mode: str = "dirichlet",
    client_logger_filename=None
) -> Tuple[Dict[int, DataLoader], Dict[int, float]]:
    train_loaders: Dict[int, DataLoader] = {}
    test_loaders: Dict[int, DataLoader] = {}
    ratio_map: Dict[int, float] = {}
    delay_map: Dict[int, float] = {}
    for idx, (name, cfg) in enumerate(cfgs.items()):
        ratio = float(cfg["ratio"])
        alpha = float(cfg["alpha"])
        seed  = int(base_seed + idx)
        delay = float(cfg["delay"])
        train_loader, test_loader = load_train_data(data = data_name, batch_size=batch_size, ratio=ratio, alpha=alpha, seed=seed, mode=train_loader_mode, client_logger_filename=client_logger_filename)
        train_loaders[idx] = train_loader
        test_loaders[idx] = test_loader
        ratio_map[idx] = ratio
        delay_map[idx] = delay
    return train_loaders, test_loaders, ratio_map, delay_map