#!/bin/bash
NUM_CLIENTS=1
PARTITION_POINTS=(4)

# Start server
nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 1 > power.csv &
POWER_PID=$!

python3 src/strategy.py "127.0.0.1" &
SERVER_PID=$!

sleep 20

# Start clients
for ((i=0; i<"$NUM_CLIENTS";i++)); do
    python3 src/run_client.py "127.0.0.1" "${PARTITION_POINTS[$i]}" &
done

wait $SERVER_PID

kill $POWER_PID
echo "Simulation finished"
