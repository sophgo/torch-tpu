#!/bin/bash
#torchrun --nproc_per_node 2 --nnodes 1 broadcast_object_list.py

python -m torch.distributed.launch \
        --nnodes 1 \
        --nproc_per_node 2 \
        broadcast_object_list.py

#per process
#RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=6000 python broadcast_object_list.py