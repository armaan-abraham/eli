MASTER_IP_LOCAL=127.0.0.1
MASTER_PORT_LOCAL=29500

MASTER_IP_REMOTE=38.99.105.118
MASTER_PORT_REMOTE=21071

NODE_RANK=0

NUM_NODES=2             
GPUS_PER_NODE=1

# Set master IP and port based on node rank
if [ "$NODE_RANK" -eq 0 ]; then
  MASTER_IP=$MASTER_IP_LOCAL
  MASTER_PORT=$MASTER_PORT_LOCAL
else
  MASTER_IP=$MASTER_IP_REMOTE
  MASTER_PORT=$MASTER_PORT_REMOTE
fi

# Launch the training
torchrun \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$GPUS_PER_NODE \
  --master_addr=$MASTER_IP \
  --master_port=$MASTER_PORT \
  src/eli/train/__main__.py