#!/bin/bash
#torchrun --nproc_per_node 2 --nnodes 1 kv_store.py

python -m torch.distributed.launch \
        --nnodes 1 \
        --nproc_per_node 2 \
        kv_store.py