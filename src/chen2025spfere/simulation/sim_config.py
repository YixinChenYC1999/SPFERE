WB_ENTITY = "change-to-your-wandb-entity" # Change to your wandb entity
WB_PROJECT = "spfere" # Change to your wandb project

SIM_COLD_TIME_BASE = 2048.0
SIM_COLD_ALPHA = 32.0

GPU_ID = "0" # Assign GPU ID, e.g., GPU_ID = "1" means use GPU 1
USE_MODEL = "SimpleCNN"   # SimpleCNN BetterCNN ShuffleCNN EfficientCNN resnet20
                        # SimpleCNN_FM is only used when f_mnist is used
DATASET = "cifar10" # "cifar10" or "stl_10" or "eurosat_rgb" or "svhn" or "f_mnist"
TRAIN_ROUND = 40
CLIENT_NUMBER = 20 # number of clients
WAIT_UNTIL_M = 5  # wait-until-M, only useful when sync and M is less than client number

G_LR = 1 # global learning rate
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
TEST_ALPHA = 0.1 # only active when dirichlet is used

PRECISION_COMPRESSION = False   # True or False
PRECISION_DTYPE = "fp16"        # "fp16" | "bf16"
KEEP_NORM_FP32 = True           # True or False
AGGR_ACCUM_FP32 = True          # True or False