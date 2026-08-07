import csv
import json
from config_run import *

def save_measurements_to_csv(measurements: dict[str, any], path: str) -> None:
    fieldnames = measurements[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in measurements:
            writer.writerow(record)

def save_performance_to_csv(accuracies: list[float], losses: list[float], path: str) -> None:
    """Saves accuracy and average loss per round"""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "test_accuracy", "test_average_loss"])
        writer.writeheader()

        for i in range(len(accuracies)):
            writer.writerow({
                "round": i, 
                "test_accuracy": accuracies[i], 
                "test_average_loss": losses[i]
            })

def save_experiment_config(experiment_end: str, optimizer: str ="SGD") -> None:
    config = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_start": EXPERIMENT_START,
        "experiment_end": experiment_end,
        "num_rounds": NUM_ROUNDS,
        "num_clients": NUM_CLIENTS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": optimizer,
        "model": MODEL,
        "dataset": DATASET,
        "gpu_server": GPU_SERVER,
        "gpu_clients": GPU_CLIENTS
    }

    json_str = json.dumps(config, indent=4)
    with open(SAVE_EXPERIMENT_CONFIG_PATH, "w") as f:
        f.write(json_str)

