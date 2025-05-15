MASTER_IP=198.145.126.237  
NUM_NODES=2             
GPUS_PER_NODE=8         
NODE_RANK=0             
MASTER_PORT=35481

# Launch the training
torchrun \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$GPUS_PER_NODE \
  --master_addr=$MASTER_IP \
  --master_port=$MASTER_PORT \
  src/eli/train/__main__.py