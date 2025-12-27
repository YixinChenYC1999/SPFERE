from __future__ import annotations
import socket, json, time, os
import random
import numpy as np
import torch

from chen2026spfere.utils.protocol_sock import send_packet, recv_packet, MSG_TYPE_MODEL, MSG_TYPE_INFO, MSG_TYPE_TERMINATE
from chen2026spfere.utils.model_map import MODEL_MAP
from chen2026spfere.utils.performance_test import test
from chen2026spfere.utils.client_logger import fl_logger
from chen2026spfere.client.client_config import CLIENT_INFO, SERVER_IP, SERVER_PORT, BATCH_SIZE, EPOCHS, SLEEP_TIME, USE_CUDA, TRAIN_RATIO, TRAIN_ALPHA
from chen2026spfere.client.client_load_data import load_data
from chen2026spfere.client.client_train_test import train
from chen2026spfere.client.client_bat import readVoltage, bus

USE_MODEL = "SimpleCNN"   # SimpleCNN ShuffleCNN EfficientCNN resnet20
                        # SimpleCNN_FM is only used when f_mnist is used
DATASET = "cifar10" # "cifar10" or "stl_10" or "eurosat_rgb" or "svhn" or "f_mnist"
TRAIN_LOADER_MODE = "dirichlet" # avg or dirichlet
# create a new instance
model_client = MODEL_MAP[USE_MODEL]()
USE_DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
model_client.to(USE_DEVICE)

device_seeds_prime = {
    'raspi_4_1': 2, 'raspi_4_2': 3, 'raspi_4_3': 5, 'raspi_4_4': 7,
    'raspi_5_1': 11, 'raspi_5_2': 13, 'raspi_5_3': 17, 'raspi_5_4': 19,
    'opi5_plus': 23, 'rockpi4c+': 29
}

def set_seed(seed=42, use_cuda=USE_CUDA):
    fl_logger(f"[+] set device seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        fl_logger("[!] Running without GPU, skipping CUDA-specific seed settings.")

def save_checkpoint(model_dict, last_loss, test_acc, ratio, interrupt_flag, train_elapsedtime, path, power_on=1, remain_V=0.0):
    checkpoint = {  'model_state_dict': model_dict,
                    'last_loss': last_loss,
                    'test_acc': test_acc,
                    'sample_ratio': ratio,
                    'interrupt_flag': interrupt_flag,
                    'train_elapsed_time': train_elapsedtime,
                    'power_on': power_on,
                    'remain_V': remain_V
                    # add more items in the future
                 }
    torch.save(checkpoint, path)

def perform_local_training(train_loader, global_round_count, train_timeout, interrupt_flag, seed=42):
    fl_logger(f"[+] Local Training and Testing Process...")
    fl_logger(f"[+] Global Round: {global_round_count}")
    
    start_time = time.time()
    total_round = 1 # remember to pass total round here
    t_loss, n_eff = train(global_round_count,
                          total_round,
                          start_time,
                          train_timeout,
                          interrupt_flag,
                          model_client,
                          USE_DEVICE,
                          train_loader,
                          EPOCHS
                          )
    fl_logger(f"[=] Local Training and Testing Finished.")
    return t_loss, n_eff

def connect_to_server(client_info, seed):
    train_loader, test_loader = load_data(data = DATASET, 
                                          batch_size = BATCH_SIZE, 
                                          ratio = TRAIN_RATIO, 
                                          alpha = TRAIN_ALPHA, 
                                          seed = seed, 
                                          mode = TRAIN_LOADER_MODE, 
                                          client_logger_filename = None
                                          )
    local_train_round = -1
    interrupt_flag = {'flag': 0}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        fl_logger("[+] Try to find the server...")
        sock.connect((SERVER_IP, SERVER_PORT))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        fl_logger("[=] Connected to the server.")
        # Send client identification information to the server
        info_payload = json.dumps(client_info).encode("utf-8")
        send_packet(sock, MSG_TYPE_INFO, info_payload, desc="Send Info")
        
        fl_logger("****************** Init")
        if(client_info["power-on"]==0):
            fl_logger("Voltage:%5.2fV" % readVoltage(bus))
        fl_logger("******************")

        while True:
            local_train_round += 1
            interrupt_flag['flag'] = 0
            fl_logger("******************")
            fl_logger(f"[+] Local round {local_train_round}. Try to receive global model from server...")
            if(client_info["power-on"]==0):
                fl_logger("Voltage:%5.2fV" % readVoltage(bus))
            fl_logger("******************")
            
            msg_type, payload = recv_packet(sock, desc="Recv fr server")
            if msg_type == MSG_TYPE_TERMINATE:
                reason = payload.decode("utf-8")
                fl_logger(f"[INFO] Server closed connection: {reason}. {client_info['client_id']} exiting cleanly at round {local_train_round}.")
                sock.close()
                return
            if msg_type == MSG_TYPE_MODEL:
                dirpath = os.path.dirname(f"client_model/received_global_my_round_{local_train_round}.pth")
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                with open(f"client_model/received_global_my_round_{local_train_round}.pth", "wb") as f:
                    f.write(payload)
                fl_logger(f"[=] Global model file received at client_model/received_global_my_round_{local_train_round}.pth.")
            checkpoint = torch.load(f"client_model/received_global_my_round_{local_train_round}.pth", map_location=USE_DEVICE)

            model_client.load_state_dict(checkpoint['model_state_dict'])
            train_timeout = checkpoint['client_timeout']
            global_round_count = checkpoint['global_round']
            fl_logger(f"[+] Global round {global_round_count} model loaded and set training timeout {train_timeout}.")
            train_start_time = time.time()
            last_loss, n_eff = perform_local_training(train_loader, global_round_count, train_timeout, interrupt_flag, seed = seed)
            train_end_time = time.time()
            test_acc, _ = test(model_client, USE_DEVICE, test_loader, None, None, None, None, None, None)
            train_elapsed_time = train_end_time - train_start_time
            
            global_weights = checkpoint['model_state_dict']
            local_weights = model_client.state_dict()
            model_update = {}
            for key in global_weights.keys():
                model_update[key] = local_weights[key] - global_weights[key]
            
            remain_V = 0.0
            if(client_info["power-on"]==0):
                remain_V = readVoltage(bus)
            save_checkpoint(model_update,
                            last_loss,
                            test_acc,
                            TRAIN_RATIO,
                            interrupt_flag['flag'],
                            train_elapsed_time,
                            f"client_model/client_cifar10_model_round_{local_train_round}.pth",
                            client_info["power-on"],
                            remain_V
                            )
            
            fl_logger(f"[+] New model of local round {local_train_round} saved.")

            fl_logger("[=] Start sending local model update to server...")
            # send back the model
            with open(f"client_model/client_cifar10_model_round_{local_train_round}.pth", "rb") as f:
                file_data = f.read()
                send_packet(sock, MSG_TYPE_MODEL, file_data, desc="Sent to server")
                fl_logger("[=] Local model update sent to the server.")

            fl_logger(f"[X] Local round {local_train_round} finished.")
            
            fl_logger("******************")
            fl_logger(f"This Round Train Using: {train_elapsed_time:.2f} second(s)")
            if(client_info["power-on"]==0):
                fl_logger("Voltage:%5.2fV" % readVoltage(bus))
            fl_logger("******************")
            # send back finished


def main():
    client_info = CLIENT_INFO
    seed = device_seeds_prime.get(client_info["client_id"], 42)
    set_seed(seed, USE_CUDA)
    fl_logger(f"[=] Client Info: {client_info}")
    
    connect_to_server(client_info, seed)

if __name__ == "__main__":
    main()
