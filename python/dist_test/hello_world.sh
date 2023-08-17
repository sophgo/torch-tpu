#!/bin/bash
#torchrun --nproc_per_node 2 --nnodes 1 test.py

python -m torch.distributed.launch \
        --nnodes 1 \
        --nproc_per_node 2 \
        hello_world.py