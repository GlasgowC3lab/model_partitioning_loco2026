from datetime import datetime
# Experiment configuration (defined for each experiment run)
EXPERIMENT_START = str(datetime.now())
EXPERIMENT_ID = EXPERIMENT_START.replace(":", "-").replace(" ", "_")
NUM_ROUNDS = 50
NUM_CLIENTS = 1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
MODEL = "ResNet50"
DATASET = "CIFAR10"
SEED = 42
GPU_CLIENTS = "L40S"
GPU_SERVER = "H100"

# General configs
PORT = 5050
SAVE_PATH_MODELS = f"./models/{EXPERIMENT_ID}/"
SAVE_PATH_CHECKPOINT = f"{SAVE_PATH_MODELS}checkpoint_"
DATA_PATH = "data"
RESULTS_PATH = f"./results/{EXPERIMENT_ID}/"
MEASUREMENTS_PATH = f"{RESULTS_PATH}results.csv"
TRAINING_SUMMARY_PATH = f"{RESULTS_PATH}training_summary.csv"
SAVE_EXPERIMENT_CONFIG_PATH = f"{RESULTS_PATH}experiment_config.json"
PERFORMANCE_PATH = f"{RESULTS_PATH}test_performance.csv"
