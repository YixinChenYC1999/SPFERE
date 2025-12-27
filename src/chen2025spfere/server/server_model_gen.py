import torch
from pathlib import Path

def save_checkpoint(model_dict, client_timeout, global_round, client_stop, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {  'model_state_dict': model_dict,
                    'client_timeout' : client_timeout,
                    'global_round' : global_round,
                    'client_stop': client_stop
                    # add more items in the future
                }
    torch.save(checkpoint, path)
