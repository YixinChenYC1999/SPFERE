import socket, threading, os, argparse, datetime, json, time
import random
import numpy as np
import torch

from chen2026spfere.utils.protocol_sock import send_packet, recv_packet, MSG_TYPE_MODEL, MSG_TYPE_INFO, MSG_TYPE_TERMINATE
from chen2026spfere.utils.model_map import MODEL_MAP
from chen2026spfere.utils.performance_test import test, recalibrate_bn_stats
from chen2026spfere.utils.server_logger import fl_logger
from chen2026spfere.server.server_config import LISTEN_HOST, LISTEN_PORT, CLIENT_TIMEOUT, USE_CUDA
from chen2026spfere.server.server_model_gen import save_checkpoint
from chen2026spfere.server.server_load_data import load_data
from chen2026spfere.server.server_lstm import predict_voltage
from chen2026spfere.server.server_aggregation import server_aggregation

timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
server_logger_filename = f"server_log/log.log"
wb_logger_name = f"log_{timestamp_str}"

import wandb # better logger
WB_ENTITY = "change-to-your-wandb-entity" # Change to your wandb entity
WB_PROJECT = "spfere" # Change to your wandb project

### change the configs begins
GPU_ID = "0" # Assign GPU ID, e.g., GPU_ID = "1"   # use GPU 1
USE_MODEL = "SimpleCNN"   # SimpleCNN ShuffleCNN EfficientCNN resnet20
                        # SimpleCNN_FM is only used when f_mnist is used
DATASET = "cifar10" # "cifar10" or "stl_10" or "eurosat_rgb" or "svhn" or "f_mnist"
TRAIN_ROUND = 30
CLIENT_NUMBER = 10 # number of clients
WAIT_UNTIL_M = 5  # wait-until-M, only useful when M is less than client number

G_LR = 0.4 # global learning rate
AGGR_METHOD = "fedavg" # fedavg q_fel f_div f_mul f_add
Q_FEL = 1 # q value
TOP_K = None

CLIENT_BATCH_SIZE = 128 # batch size for training dataset at client
CLIENT_EPOCHS = 2 # local epoch
TRAIN_LOADER_MODE = "dirichlet" # avg or dirichlet
CLIENT_ALPHA_LOW = 0.1 # alpha in dirichlet distribution
CLIENT_ALPHA_HIGH = 5 # alpha range

CLIENT_USE_OPTIMIZER = "sgd" # client train optimizer, sgd or adam
CLIENT_LR = 0.01 # client train lr
CLIENT_LR_GAMMA = 1.0 # client train lr decay gamma

SERVER_BATCH_SIZE = 128 # batch size for testing dataset at server
TEST_SEED = 42
TEST_LOADER_MODE = "avg" # avg or dirichlet (checked - no issues)
TEST_ALPHA = 10 # only active when dirichlet is used

PRECISION_COMPRESSION = False   # True or False
PRECISION_DTYPE = "fp16"        # "fp16" | "bf16"
KEEP_NORM_FP32 = True
AGGR_ACCUM_FP32 = True
### change the configs ends

CNN_Model = MODEL_MAP[USE_MODEL]
USE_DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

test_loader, calib_loader = load_data(data = DATASET, batch_size = SERVER_BATCH_SIZE, ratio = 1, alpha = TEST_ALPHA, seed = TEST_SEED, mode = TEST_LOADER_MODE, server_logger_filename = None)

def set_seed(seed=42, use_cuda=USE_CUDA):
    fl_logger(f"[INFO] set device seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        fl_logger("[WARNING] Running without GPU, skipping CUDA-specific seed settings.")

import math

def _empty_bucket():
    return {"state_dicts": [], "last_loss": [], "test_acc": [],
            "ratio": [], "interrupt": [], "freq": []}

class FederatedServer:
    def __init__(self, host=LISTEN_HOST, port=LISTEN_PORT):
        self.host = host
        self.port = port
        self.clients = {}
        self.client_socks = []
        self.global_model = self.get_init_model()
        self.global_round_id = 0  # Global round identifier
        self.client_state_dicts = []
        self.client_test_acc = []
        self.client_last_loss = []
        self.client_ratio = []
        self.client_interrupt_flags = []
        self.client_freq = []
        self.finished_client_count = 0
        self.close_count = 0
        self.lock = threading.Lock()  # Protect shared resources
        self.last_test_acc = None
        self.last_test_loss = None
        
        self.coldstart_done = False
        self.RTT_avg_ms = {}
        self.RTT_count = {}
        self.groups = {}                # {g: [cid,...]}
        self.group_time_ms = {}         # {g: max_ms}
        self.need_intra = {}            # {g: K_g}
        self.cid2group = {}
        self.group_intra_done = {}
        self.group_waiting = {}
        self.group_bucket = {}
        self.inter_buffer = _empty_bucket()
        self.groups_ready_for_inter = set()

    def client_info_regist(self, client_info):
        with self.lock:
            self.clients[client_info["client_id"]] = client_info
    
    def get_init_model(self):
        """Model initialization"""
        ini_model = CNN_Model()
        ini_model.to(USE_DEVICE)
        model_path_ini = "server_model/global_model_0_ini.pth"
        save_checkpoint(ini_model.state_dict(), 0, 0, 0, model_path_ini)
        return ini_model
        
        
    def _build_groups_from_real_timing(self, wait_M):
        items = sorted(self.RTT_avg_ms.items(), key=lambda kv: (kv[1], kv[0]))
        groups, group_time_ms = {}, {}
        for g, start in enumerate(range(0, len(items), wait_M)):
            chunk = items[start:start+wait_M]  # [(cid, t_ms), ...]
            cid_list = [cid for cid, _ in chunk]
            groups[g] = cid_list
            group_time_ms[g] = max((t for _, t in chunk), default=0.0)

        last_g = max(groups.keys())
        last_base = max(group_time_ms[last_g], 1e-9)
        need_intra = {}
        for g, base in group_time_ms.items():
            ratio = last_base / max(base, 1e-9)
            need_intra[g] = max(int(math.floor(ratio + 1e-9)), 1)

        self.groups = groups
        self.group_time_ms = group_time_ms
        self.need_intra = need_intra
        self.coldstart_done = True
        fl_logger(f"[INFO] Cold start finished")
        fl_logger(f"[INFO] groups: {self.groups}")
        fl_logger(f"[INFO] need_intra: {self.need_intra}")
        
        self.cid2group = {cid: g for g, cids in self.groups.items() for cid in cids}
        self.group_intra_done = {g: 0 for g in self.groups}
        self.group_waiting = {g: set(cids) for g, cids in self.groups.items()}
        self.group_bucket = {g: _empty_bucket() for g in self.groups}
        self.groups_ready_for_inter = set()
    
    def perform_server_testing(self, global_round, global_stage, variance, cosine_similarity, worst_10, best_10, wb_logger=None, wb_logger_enable=False):
        """Perform server model testing here"""
        fl_logger(f"[INFO] Server Test Process...")
        self.global_model.to(USE_DEVICE)
        used = recalibrate_bn_stats(self.global_model, calib_loader, USE_DEVICE, max_batches=50)
        fl_logger(f"[BN] recalibrated with {used} samples.")
        test_acc, test_loss = test(self.global_model, USE_DEVICE, test_loader, global_round, global_stage, variance, cosine_similarity, worst_10, best_10, None, wb_logger, wb_logger_enable)
        self.last_test_acc = test_acc
        self.last_test_loss = test_loss
        fl_logger(f"[INFO] Server Test Finished")

    def handle_client(self, client_socket, address, wb_logger):
        """connection to client"""
        msg_type, payload = recv_packet(client_socket, desc=f"Recv client info")
        if msg_type == MSG_TYPE_INFO:
            client_info = json.loads(payload.decode("utf-8"))
            fl_logger(f"[INFO] Received client info: {client_info}")
        self.client_info_regist(client_info)
        client_id = client_info["client_id"]
        client_device_type = client_info["device_type"]
        client_remain_voltage = []
        client_stop = 0
        fl_logger(f"[INFO] Received info from client {client_id}", f"server_log/{client_id}_log.log")
        fl_logger(f"[INFO] New client connected: {client_id}", f"server_log/{client_id}_log.log")
        train_round = -1
        client_timeout = CLIENT_TIMEOUT

        fl_logger(f"[INFO] Client training round 0 [Cold Start], sending initial model to {client_id}", f"server_log/{client_id}_log.log")
        
        with open("server_model/global_model_0_ini.pth", "rb") as f:
            start = time.perf_counter()
            file_data = f.read()
            send_packet(client_socket, MSG_TYPE_MODEL, file_data, desc=f"Sent to {client_id}")
            fl_logger(f"[INFO] Sent model global_model_0_ini.pth to {client_id}", f"server_log/{client_id}_log.log")

        while True:
            """handle one round"""
            while True:
                if self.clients[client_id]["train-flag"] == 1: # if stat is ready
                    self.clients[client_id]["train-flag"] = -1 # change stat to train
                    fl_logger(f"[INFO] train-flag for client {client_id} changed to: {self.clients[client_id]['train-flag']}", f"server_log/{client_id}_log.log")
                    fl_logger(f"[INFO] threadlock status: released", f"server_log/{client_id}_log.log")
                    break
                fl_logger(f"[.] Waiting for train-flag==1 for client {client_id}, current flag: {self.clients[client_id]['train-flag']}", f"server_log/{client_id}_log.log")
                time.sleep(2)
            
            train_round+=1
            
            # stop for low volatge
            if client_stop == 1:
                break
            
            # stop for finishing
            if self.global_round_id > TRAIN_ROUND:
                message = f"Global Training End at Round {TRAIN_ROUND}".encode("utf-8")
                send_packet(client_socket, MSG_TYPE_TERMINATE, message, desc=f"Sent to {client_id}")
                break
            
            # Start of one round
            if train_round != 0: # not cold start
                fl_logger(f"[INFO] Client training round {train_round}, sending model to {client_id}", f"server_log/{client_id}_log.log")
                g = self.cid2group.get(client_id, None)
                if self.clients[client_id]['train-stage']==0: # global round 1, use the init model
                    global_dict_new_round = (torch.load(f"server_model/global_model_0_ini.pth", map_location=USE_DEVICE))['model_state_dict']
                    fl_logger(f"[INFO] loading server_model/global_model_0_ini.pth", f"server_log/{client_id}_log.log")
                elif self.clients[client_id]['train-stage']=="inter": # this is the case of a new round after inter aggregation, use the rule of round_{old round_id}_stage_inter
                    global_dict_new_round = torch.load(f"server_model/global_model_round_{self.global_round_id-1}_stage_inter.pth", map_location=USE_DEVICE)
                    fl_logger(f"[INFO] loading server_model/global_model_round_{self.global_round_id-1}_stage_inter.pth", f"server_log/{client_id}_log.log")
                else: # this is the case of a new stage after intra aggregation, use the rule of round_{old round_id}_group_{g}_stage_{old stage}
                    global_dict_new_round = torch.load(f"server_model/global_model_round_{self.global_round_id}_group_{g}_stage_{self.clients[client_id]['train-stage']}.pth", map_location=USE_DEVICE)
                    fl_logger(f"[INFO] loading server_model/global_model_round_{self.global_round_id}_group_{g}_stage_{self.clients[client_id]['train-stage']}.pth", f"server_log/{client_id}_log.log")
                save_checkpoint(global_dict_new_round, client_timeout, self.global_round_id, client_stop, f"server_model/{client_id}_sent_globalround_{self.global_round_id}_stage_{self.clients[client_id]['agg-count']}.pth") # use agg-count to correctly mark the current stage
                fl_logger(f"[INFO] saving to server_model/{client_id}_sent_globalround_{self.global_round_id}_stage_{self.clients[client_id]['agg-count']}.pth", f"server_log/{client_id}_log.log")
                
                with open(f"server_model/{client_id}_sent_globalround_{self.global_round_id}_stage_{self.clients[client_id]['agg-count']}.pth", "rb") as f:
                    start = time.perf_counter()
                    file_data = f.read()
                    send_packet(client_socket, MSG_TYPE_MODEL, file_data, desc=f"Sent to {client_id}")
                    fl_logger(f"[INFO] Sent model {client_id}_sent_globalround_{self.global_round_id}_stage_{self.clients[client_id]['agg-count']}.pth to {client_id}", f"server_log/{client_id}_log.log")
                
            fl_logger(f"[.] Waiting for client {client_id}...", f"server_log/{client_id}_log.log")
            msg_type, payload = recv_packet(client_socket, desc=f"Recv fr {client_id}")
            end = time.perf_counter()
            RTT_ms = 1000.*(end - start)
            if client_id not in self.RTT_avg_ms:
                self.RTT_avg_ms[client_id] = RTT_ms
                self.RTT_count[client_id] = 1
            else:
                self.RTT_count[client_id] += 1
                self.RTT_avg_ms[client_id] += (RTT_ms - self.RTT_avg_ms[client_id]) / self.RTT_count[client_id]
            
            if msg_type == MSG_TYPE_MODEL:
                with open(f"server_model/{client_id}_received_round_{train_round}.pth", "wb") as f:
                    f.write(payload)
                
            checkpoint = torch.load(f"server_model/{client_id}_received_round_{train_round}.pth", map_location=USE_DEVICE)
            last_loss = checkpoint['last_loss']
            c_ratio = checkpoint['sample_ratio']
            client_interrupt_flag = checkpoint['interrupt_flag']
            train_elapsed_time = checkpoint['train_elapsed_time']
            power_on = checkpoint['power_on']
            remain_V = checkpoint['remain_V']
            # if client_interrupt_flag != 1: # interrupt flag not used till now
            fl_logger(f"[INFO] Received model update from client {client_id} on {address}\n                      [INFO] Last Loss: {last_loss}\n                      [INFO] Client sample ratio: {c_ratio}\n                      [INFO] Power status: {'Plugged In' if power_on == 1 else 'Battery'}\n                      [INFO] Remain voltage: {remain_V:.2f} V\n                      [INFO] Train elapsed time: {train_elapsed_time:.2f} seconds\n                      [INFO] RTT_ms: {RTT_ms:.2f} ms\n                      [INFO] RTT_avg_ms: {self.RTT_avg_ms[client_id]:.2f} ms", f"server_log/{client_id}_log.log")

            # Aggregate the received model update
            with self.lock:
                g = self.cid2group.get(client_id, None)
                self.finished_client_count += 1 # only for cold start count
                self.clients[client_id]["train-flag"] = 0 # change stat to hold
                fl_logger(f"[INFO] train-flag for client {client_id} changed to: {self.clients[client_id]['train-flag']}", f"server_log/{client_id}_log.log")
                fl_logger(f"[INFO] threadlock status: locked", f"server_log/{client_id}_log.log")
                
                if not self.coldstart_done: # the case for cold start
                    if self.finished_client_count == CLIENT_NUMBER: # all clients finished one round training
                        self.perform_server_testing(self.global_round_id, None, None, None, None, None, wb_logger, True) # this is to record G0 acc info
                        self._build_groups_from_real_timing(wait_M=WAIT_UNTIL_M) # build the groups and need_intra
                        print(self.groups)
                        print(self.need_intra)
                        self.global_round_id=1 # change to global round 1
                        for loop_cid, info in self.clients.items(): # reset all clients stat for ready training
                            info["train-flag"] = 1 # change stat to ready
                            info["train-stage"] = 0 # the current stage of intra aggregation, only if 0 means CS finished and fetch the init model when train begins
                            info["agg-count"] = 1 # expected freq when the next aggregation happends, default as 1
                            fl_logger(f"[INFO] train-flag for client {loop_cid} changed to: {self.clients[loop_cid]['train-flag']}", f"server_log/{loop_cid}_log.log")
                            fl_logger(f"[INFO] threadlock status: released", f"server_log/{loop_cid}_log.log")
                else: # the case for normal training beginning from global round 1
                    # append all the results to the group bucket
                    self.group_bucket[g]["state_dicts"].append(checkpoint["model_state_dict"])
                    self.group_bucket[g]["last_loss"].append(checkpoint["last_loss"])
                    self.group_bucket[g]["test_acc"].append(checkpoint["test_acc"])
                    self.group_bucket[g]["ratio"].append(checkpoint["sample_ratio"])
                    self.group_bucket[g]["interrupt"].append(checkpoint["interrupt_flag"])
                    self.group_bucket[g]["freq"].append(self.clients[client_id]["agg-count"])
                    # mark this client as done for this group
                    self.group_waiting[g].remove(client_id)
                    if not self.group_waiting[g]: # if all clients in this group is done, make intra or inter aggr
                        self.group_intra_done[g]+=1 # should increase the current intra count by 1 before doing aggregation
                        if self.group_intra_done[g] < self.need_intra[g]: # intra case
                            self.global_model, variance, cosine_similarity, worst_10, best_10 = server_aggregation(
                                self.global_model,
                                self.group_bucket[g]["state_dicts"],
                                self.group_bucket[g]["last_loss"],
                                self.group_bucket[g]["test_acc"],
                                self.group_bucket[g]["ratio"],
                                self.group_bucket[g]["interrupt"],
                                self.group_bucket[g]["freq"],
                                Q_FEL, G_LR, server_logger_filename=None, mode=AGGR_METHOD,
                                aggr_accum_fp32=AGGR_ACCUM_FP32, top_k=TOP_K, last_loss=self.last_test_loss
                            )
                            self.group_bucket[g] = _empty_bucket() # clear this group bucket
                            self.group_waiting[g] = set(self.groups[g]) # reset the waiting stat for this group
                            # save the aggregated model with the rule of round_{self.global_round_id}_group_{g}_stage_{self.group_intra_done[g]}
                            torch.save(self.global_model.state_dict(), f"server_model/global_model_round_{self.global_round_id}_group_{g}_stage_{self.group_intra_done[g]}.pth")
                            self.perform_server_testing(self.global_round_id, f"g_{g}_k_{self.group_intra_done[g]}", variance, cosine_similarity, worst_10, best_10)
                            for loop_cid in self.groups[g]: # for the clients only in this group
                                info = self.clients[loop_cid]
                                info["train-flag"]  = 1 # update the stat to ready
                                info["train-stage"] = self.group_intra_done[g] # record the current finished stage to fetch the correct new model
                                info["agg-count"]   = self.group_intra_done[g]+1 # record the expected freq for next aggreagtion
                                fl_logger(f"[INFO] train-flag for client {loop_cid} changed to: {self.clients[loop_cid]['train-flag']}", f"server_log/{loop_cid}_log.log")
                                fl_logger(f"[INFO] threadlock status: released", f"server_log/{loop_cid}_log.log")
                        else: # inter case
                            # extend all the results from the group bucket to the inter buffer
                            self.inter_buffer["state_dicts"].extend(self.group_bucket[g]["state_dicts"])
                            self.inter_buffer["last_loss"].extend(self.group_bucket[g]["last_loss"])
                            self.inter_buffer["test_acc"].extend(self.group_bucket[g]["test_acc"])
                            self.inter_buffer["ratio"].extend(self.group_bucket[g]["ratio"])
                            self.inter_buffer["interrupt"].extend(self.group_bucket[g]["interrupt"])
                            self.inter_buffer["freq"].extend(self.group_bucket[g]["freq"])
                            self.groups_ready_for_inter.add(g) # mark group g as ready for inter aggregation
                            self.group_bucket[g] = _empty_bucket() # clear bucket for g
                            self.group_waiting[g] = set(self.groups[g]) # reset the waiting stat for this group
                            if len(self.groups_ready_for_inter) == len(self.groups): # all groups are ready, make inter
                                self.global_model, variance, cosine_similarity, worst_10, best_10 = server_aggregation(
                                    self.global_model,
                                    self.inter_buffer["state_dicts"],
                                    self.inter_buffer["last_loss"],
                                    self.inter_buffer["test_acc"],
                                    self.inter_buffer["ratio"],
                                    self.inter_buffer["interrupt"],
                                    self.inter_buffer["freq"],
                                    Q_FEL, G_LR, server_logger_filename=None,
                                    mode=AGGR_METHOD, aggr_accum_fp32=AGGR_ACCUM_FP32, top_k=TOP_K, last_loss=self.last_test_loss
                                )
                                self.inter_buffer = _empty_bucket() # clear the inter buffer
                                # save the aggregated model with the rule of round_{self.global_round_id}_stage_inter
                                torch.save(self.global_model.state_dict(), f"server_model/global_model_round_{self.global_round_id}_stage_inter.pth")
                                self.perform_server_testing(self.global_round_id, "inter", variance, cosine_similarity, worst_10, best_10, wb_logger, True) # perform test for inter and report to wandb
                                self.finished_client_count = 0 # a simple reset, but only useful for cold start, just to avoid overflow
                                self.global_round_id += 1 # as inter aggregation finished, increase the global round count here
                                self.group_intra_done = {g: 0 for g in self.groups} # reset the counter for group intra
                                self.groups_ready_for_inter = set() # reset the counter for inter aggregation
                                for loop_cid, info in self.clients.items(): # for all clients
                                        info["train-flag"] = 1 # update the stat to ready
                                        info["train-stage"] = "inter" # mark as inter to fetch the correct model for the next round
                                        info["agg-count"] = 1 # reset the expected freq to 1
                                        fl_logger(f"[INFO] train-flag for client {loop_cid} changed to: {self.clients[loop_cid]['train-flag']}", f"server_log/{loop_cid}_log.log")
                                        fl_logger(f"[INFO] threadlock status: released", f"server_log/{loop_cid}_log.log")
                                

            # only predict remain voltage when battery powered
            if power_on == 0:
                client_remain_voltage.append(remain_V)
                len_V = len(client_remain_voltage)
                if len_V > 5:
                    pred = predict_voltage(client_device_type, client_remain_voltage[-6:])
                    fl_logger(f"[INFO] Remain Voltage Prediction for {client_id}: {pred:.2f} V", f"server_log/{client_id}_log.log")
                    if pred < 2.9:
                        client_stop = 1 # not finished, just placeholder
                else:
                    fl_logger(f"[INFO] Need More Data .. Remain Voltage Prediction for {client_id}", f"server_log/{client_id}_log.log")
            client_timeout = 0
            # End of one round
        client_socket.close()
        self.client_socks = [c for c in self.client_socks if c["socket"] is not client_socket]
        fl_logger(f"[INFO] Connection closed: client {client_id} on {address}", f"server_log/{client_id}_log.log")
        with self.lock:
            self.close_count += 1
            fl_logger(f"[INFO] Close Count: {self.close_count}")
            if self.close_count == CLIENT_NUMBER:
                time.sleep(1)
                fl_logger(f"[INFO] All Connections Closed. Server is shutting down.")
                wb_logger.finish()
                os._exit(0)

    def start(self, wb_logger):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            s.bind((self.host, self.port))
            s.listen()
            fl_logger(f"[INFO] Server listening on {self.host}:{self.port}")
            try:
                while True:
                    client_sock, addr = s.accept()
                    client_info = {"socket": client_sock, "address": addr}
                    self.client_socks.append(client_info)
                    client_thread = threading.Thread(target=self.handle_client, args=(client_sock, addr, wb_logger))
                    client_thread.start()
            except KeyboardInterrupt:
                fl_logger("[INFO] Server is shutting down...")
            except Exception as e:
                fl_logger(f"[INFO] Server error: {e}")
            finally:
                for client in self.client_socks:
                    sock = client["socket"]
                    if sock.fileno() != -1:
                        sock.close()
                s.close()
                fl_logger("[INFO] Server shutdown completed")

def str2bool(v):
    vl = v.lower()
    if vl in ("true", "false"):
        return vl == "true"
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--USE_MODEL", type=str, choices=["SimpleCNN", "ShuffleCNN", "EfficientCNN", "resnet20", "SimpleCNN_FM", "EfficientCNN_DP"], default=USE_MODEL)
    parser.add_argument("--DATASET", type=str, choices=["cifar10", "svhn", "stl_10", "eurosat_rgb", "f_mnist"], default=DATASET)
    
    parser.add_argument("--TRAIN_ROUND", type=int, default=TRAIN_ROUND)
    parser.add_argument("--CLIENT_NUMBER", type=int, default=CLIENT_NUMBER)
    parser.add_argument("--WAIT_UNTIL_M", type=int, default=WAIT_UNTIL_M)
    
    parser.add_argument("--G_LR", type=float, default=G_LR)
    parser.add_argument("--AGGR_METHOD", type=str, choices=["fedavg", "q_fel", "f_div", "f_mul", "f_add", "afl", "fair"], default=AGGR_METHOD)
    parser.add_argument("--Q_FEL", type=float, default=Q_FEL)
    
    parser.add_argument("--TEST_SEED", type=int, default=TEST_SEED)
    parser.add_argument("--TEST_LOADER_MODE", type=str, choices=["avg", "dirichlet"], default=TEST_LOADER_MODE)
    parser.add_argument("--TEST_ALPHA", type=float, default=TEST_ALPHA)

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
            "USE_MODEL": args.USE_MODEL,
            "DATASET": args.DATASET,

            "TRAIN_ROUND": args.TRAIN_ROUND,
            "CLIENT_NUMBER": args.CLIENT_NUMBER,
            "WAIT_UNTIL_M": args.WAIT_UNTIL_M,

            "G_LR": args.G_LR,
            "AGGR_METHOD": args.AGGR_METHOD,
            "Q_FEL": args.Q_FEL,

            "TEST_SEED": args.TEST_SEED,
            "TEST_LOADER_MODE": args.TEST_LOADER_MODE,
            "TEST_ALPHA": args.TEST_ALPHA,
        },
    )
    
    fl_logger(f"=============== Configs ===============", server_logger_filename)
    fl_logger(f"[INFO] USE_MODEL = {args.USE_MODEL}", server_logger_filename)
    fl_logger(f"[INFO] DATASET = {args.DATASET}", server_logger_filename)
    fl_logger(f"---------------------------------------", server_logger_filename)
    fl_logger(f"[INFO] TRAIN_ROUND = {args.TRAIN_ROUND}", server_logger_filename)
    fl_logger(f"[INFO] CLIENT_NUMBER = {args.CLIENT_NUMBER}", server_logger_filename)
    fl_logger(f"[INFO] WAIT_UNTIL_M = {args.WAIT_UNTIL_M}", server_logger_filename)
    fl_logger(f"---------------------------------------", server_logger_filename)
    fl_logger(f"[INFO] G_LR = {args.G_LR}", server_logger_filename)
    fl_logger(f"[INFO] AGGR_METHOD = {args.AGGR_METHOD}", server_logger_filename)
    fl_logger(f"[INFO] Q_FEL = {args.Q_FEL}", server_logger_filename)
    fl_logger(f"---------------------------------------", server_logger_filename)
    fl_logger(f"[INFO] TEST_SEED = {args.TEST_SEED}", server_logger_filename)
    fl_logger(f"[INFO] TEST_LOADER_MODE = {args.TEST_LOADER_MODE}", server_logger_filename)
    fl_logger(f"[INFO] TEST_ALPHA = {args.TEST_ALPHA}", server_logger_filename)
    fl_logger(f"---------------------------------------", server_logger_filename)
    fl_logger(f"[INFO] WB_PROJECT = {WB_PROJECT}", server_logger_filename)
    fl_logger(f"=============== Configs ===============", server_logger_filename)
    
    globals().update({
        "USE_MODEL": args.USE_MODEL,
        "DATASET": args.DATASET,
        
        "TRAIN_ROUND": args.TRAIN_ROUND,
        "CLIENT_NUMBER": args.CLIENT_NUMBER,
        "WAIT_UNTIL_M": args.WAIT_UNTIL_M,
        
        "G_LR": args.G_LR,
        "AGGR_METHOD": args.AGGR_METHOD,
        "Q_FEL": args.Q_FEL,

        "TEST_SEED": args.TEST_SEED,
        "TEST_LOADER_MODE": args.TEST_LOADER_MODE,
        "TEST_ALPHA": args.TEST_ALPHA,
    })
    set_seed()
    fs = FederatedServer()
    fs.start(wb_logger)