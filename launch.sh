MASTER_IP_LOCAL=127.0.0.1
MASTER_PORT_LOCAL=29500

MASTER_IP_REMOTE=38.99.105.118
MASTER_PORT_REMOTE=21071

NODE_RANK=0

NUM_NODES=2
GPUS_PER_NODE=1

# New argument for launch mode: 'single' or 'multi'
# Defaults to 'multi' if $1 is not provided
LAUNCH_MODE=${1:-multi}

if [ "$LAUNCH_MODE" = "single" ]; then
  # Single-node configuration
  echo "Launching in single-node mode..."
  CMD_NNODES="--nnodes=1"
  CMD_NPROC_PER_NODE="" # Omit for auto nproc_per_node in torchrun
  CMD_MASTER_ADDR="--master_addr=$MASTER_IP_LOCAL"
  CMD_MASTER_PORT="--master_port=$MASTER_PORT_LOCAL"
  CMD_NODE_RANK="--node_rank=0"
else
  # Multi-node configuration
  echo "Launching in multi-node mode (NODE_RANK=$NODE_RANK)..."
  CMD_NNODES="--nnodes=$NUM_NODES"
  CMD_NPROC_PER_NODE="--nproc_per_node=$GPUS_PER_NODE"

  # Set master IP and port based on node rank for multi-node
  if [ "$NODE_RANK" -eq 0 ]; then
    CURRENT_MASTER_IP=$MASTER_IP_LOCAL
    CURRENT_MASTER_PORT=$MASTER_PORT_LOCAL
  else
    CURRENT_MASTER_IP=$MASTER_IP_REMOTE
    CURRENT_MASTER_PORT=$MASTER_PORT_REMOTE
  fi
  CMD_MASTER_ADDR="--master_addr=$CURRENT_MASTER_IP"
  CMD_MASTER_PORT="--master_port=$CURRENT_MASTER_PORT"
  CMD_NODE_RANK="--node_rank=$NODE_RANK"
fi

# Launch the training
echo "Running torchrun with the following configuration:"
echo "  NNODES: ${CMD_NNODES#*=} (from $CMD_NNODES)"
echo "  NODE_RANK: ${CMD_NODE_RANK#*=} (from $CMD_NODE_RANK)"
if [ -n "$CMD_NPROC_PER_NODE" ]; then
  echo "  NPROC_PER_NODE: ${CMD_NPROC_PER_NODE#*=} (from $CMD_NPROC_PER_NODE)"
else
  echo "  NPROC_PER_NODE: auto (argument omitted)"
fi
echo "  MASTER_ADDR: ${CMD_MASTER_ADDR#*=} (from $CMD_MASTER_ADDR)"
echo "  MASTER_PORT: ${CMD_MASTER_PORT#*=} (from $CMD_MASTER_PORT)"

torchrun \
  $CMD_NNODES \
  $CMD_NODE_RANK \
  $CMD_NPROC_PER_NODE \
  $CMD_MASTER_ADDR \
  $CMD_MASTER_PORT \
  src/eli/train/__main__.py