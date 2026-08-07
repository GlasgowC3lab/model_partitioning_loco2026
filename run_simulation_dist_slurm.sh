#!/bin/bash -l

############# SLURM SETTINGS #############
#SBATCH --job-name=resnet50_0_1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=0-08:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH hetjob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
  

############# LOADING MODULES (optional) #############
module load miniforge3

############# MY CODE #############
conda activate py3.12

run_server() {
    nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 1 > "resnet50_0_1_power_server.csv" &
    local POWER_PID=$!

    python3 src/strategy.py "$@"
    local SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"

    kill $POWER_PID
    wait $POWER_PID 2>/dev/null
}

run_clients() {
    NUM_CLIENTS=1
    PARTITION_POINTS=(0)

    nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 1 > "resnet50_0_1_power_client.csv" &
    local POWER_PID=$!

    for ((i=0; i<"$NUM_CLIENTS";i++)); do
        python3 src/run_client.py "$@" "${PARTITION_POINTS[$i]}" &
        local CLIENT_PID=$!
        echo "Client PID: ${CLIENT_PID}"
    done

    wait $CLIENT_PID
    kill $POWER_PID
    wait $POWER_PID 2>/dev/null
}

export -f run_server
export -f run_clients

SERVER_HOSTNAME="$(scontrol show hostnames $SLURM_NODELIST | head -n 1)"

srun --het-group=0 bash -c "run_server $SERVER_HOSTNAME" &

sleep 20

srun --het-group=1 bash -c "run_clients $SERVER_HOSTNAME"
wait

echo "Simulation finished"
