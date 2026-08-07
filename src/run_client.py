from client import Client
from config_run import PORT, SEED
import sys
import socket
import torch

hostname = sys.argv[1]
if hostname == "127.0.0.1":
    host = hostname
else:
    host = socket.gethostbyname(hostname)

# PyTorch reproducibility
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


if len(sys.argv) > 2:
    partition_point = int(sys.argv[2])
else:
    partition_point = None

client = Client(host, PORT, partition_point)
