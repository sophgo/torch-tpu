#!/bin/bash
#torchrun --nproc_per_node 2 --nnodes 1 all_reduce.py

python -m torch.distributed.launch \
        --nnodes 1 \
        --nproc_per_node 2 \
        all_to_all.py

#per process
#RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=6000 gdb --args python all_to_all.py