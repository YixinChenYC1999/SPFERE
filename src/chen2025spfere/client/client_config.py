from .client_info import *
# Server ip address use ip a on server to see the ip, in the testing router server ip is fixed
SERVER_IP = "192.168.50.192"
# Connection port of server should be same as server
SERVER_PORT = 65432
# Batch size for training and testing
BATCH_SIZE = 128
# Number of epochs for training
EPOCHS = 2
# Used for different methods should be prime numbers to see significant differences
SLEEP_TIME = 0
# Either to use cuda or not, for edge devices always set to False
USE_CUDA = False

# data selection
DATA_SEL = True

devices_data_select = {
    'raspi_4_1': {'ratio': 0.1, 'alpha': 0.1},#1
    'raspi_4_2': {'ratio': 0.3, 'alpha': 0.5},#2
    'raspi_4_3': {'ratio': 0.3, 'alpha': 0.1},#2
    'raspi_4_4': {'ratio': 0.1, 'alpha': 0.5},#1
    'raspi_5_1': {'ratio': 0.8, 'alpha': 0.1},#2
    'raspi_5_2': {'ratio': 0.3, 'alpha': 0.5},#1
    'raspi_5_3': {'ratio': 0.3, 'alpha': 0.1},#1
    'raspi_5_4': {'ratio': 0.8, 'alpha': 0.5},#2
    'opi5_plus': {'ratio': 0.3, 'alpha': 10},#1
    'rockpi4c+': {'ratio': 0.3, 'alpha': 10}#2
}

# Dataset ratio
TRAIN_RATIO = 0.1
if DATA_SEL:
    client_id = CLIENT_INFO.get("client_id")
    if client_id in devices_data_select:
        TRAIN_RATIO = devices_data_select[client_id].get("ratio", 0.1)
    
TRAIN_ALPHA = 1
if DATA_SEL:
    client_id = CLIENT_INFO.get("client_id")
    if client_id in devices_data_select:
        TRAIN_ALPHA = devices_data_select[client_id].get("alpha", 1)