# Defines the main training process
from server import Server
from config_run import *
import torch
from utils import save_experiment_config, save_performance_to_csv
from datetime import datetime
import sys
import socket

hostname = sys.argv[1]
if hostname == "127.0.0.1":
    host = hostname
else:
    host = socket.gethostbyname(hostname)

# PyTorch reproducibility
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

for round in range(NUM_ROUNDS):
    if round == 0:
        server = Server(
            ip_address=host,
            port=PORT,
            num_training_rounds=NUM_ROUNDS,
            learning_rate=LEARNING_RATE,
            momentum=MOMENTUM,
            model_name=MODEL
        )
        server.send_models()
        
    server.train_round(round)
    server.aggregate_round()
    server.test()
    print(f"[STRATEGY] Round {round+1} finished")
    round += 1

server.disconnect()
experiment_end = str(datetime.now())
save_experiment_config(experiment_end=experiment_end)
save_performance_to_csv(server.test_accuracy, server.test_average_loss, PERFORMANCE_PATH)
