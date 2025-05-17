#!/bin/bash

. .venv/bin/activate

# Configuration
MASTER_IP_LOCAL=127.0.0.1
MASTER_PORT_LOCAL=29500
MASTER_IP_REMOTE=38.99.105.118
MASTER_PORT_REMOTE=21071
NODE_RANK=0
NUM_NODES=2

# Launch mode: 'single' or 'multi'
LAUNCH_MODE=${1:-single}

# Build torchrun command
TORCHRUN_CMD="torchrun"

if [ "$LAUNCH_MODE" = "single" ]; then
  echo "Launching in single-node mode with auto GPU detection..."
  TORCHRUN_CMD="$TORCHRUN_CMD --nnodes=1 --nproc_per_node=auto"
else
  echo "Launching in multi-node mode (NODE_RANK=$NODE_RANK)..."
  TORCHRUN_CMD="$TORCHRUN_CMD --nnodes=$NUM_NODES --nproc_per_node=auto --node_rank=$NODE_RANK"
  
  # Set master address based on node rank
  if [ "$NODE_RANK" -eq 0 ]; then
    MASTER_IP=$MASTER_IP_LOCAL
    MASTER_PORT=$MASTER_PORT_LOCAL
  else
    MASTER_IP=$MASTER_IP_REMOTE
    MASTER_PORT=$MASTER_PORT_REMOTE
  fi
  
  TORCHRUN_CMD="$TORCHRUN_CMD --master_addr=$MASTER_IP --master_port=$MASTER_PORT"
fi

# Add script path
TORCHRUN_CMD="$TORCHRUN_CMD src/eli/train/__main__.py"

# Display configuration
echo "Running with configuration:"
echo "  Mode: $LAUNCH_MODE"
echo "  Command: $TORCHRUN_CMD"

# Clear log file
> log.txt

# Execute command and log output
echo "Logging output to log.txt"
eval "$TORCHRUN_CMD" >> log.txt 2>&1