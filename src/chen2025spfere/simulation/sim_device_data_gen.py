import random

def generate_devices_data(n: int,
                          ratio_range=(0.1, 0.5),
                          alpha_range=(0.1, 10),
                          delay_range=(1.0, 5.0),
                          gen_seed: int = None):
    if gen_seed is not None:
        random.seed(gen_seed)

    devices_data = {}
    for i in range(n):
        cid = i
        devices_data[cid] = {
            "ratio": round(random.uniform(*ratio_range), 2),
            "alpha": round(random.uniform(*alpha_range), 2),
            "delay": round(random.uniform(*delay_range), 2)
        }
    return devices_data
