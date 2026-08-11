# Could Model Partitioning Make Federated Learning More Sustainable?
Official repository for the paper "Could Model Partitioning Make Federated Learning More Sustainable?", to be presented at LOCO 2026.
## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
You should download CIFAR10 before your first run with a new installation:
```bash
python src/download_dataset.py CIFAR10
```
## Usage
Training hyperparameters and configurations can be defined in `src/config_run.py`.

For local simulations, you can specify the number of clients and their partition points in `run_simulation.sh` before running the script.

Alternatively, clients and server can be run manually on different devices.  
First, start the training process and server:
```bash
python src/strategy.py [server_address]
```
Then, after waiting briefly to allow the server socket to start listening, start each client with:
```bash
python src/run_client.py [server_address] [partition_point]
```

`run_experiment_template.sh` further contains the script template used to run all experiments.
## Citation
To be added
