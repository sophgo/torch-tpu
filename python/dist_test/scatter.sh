#!/bin/bash

python -m torch.distributed.launch \
        --nnodes 1 \
        --nproc_per_node 8 \
        scatter.py

torchrun --nproc_per_node 8 --nnodes 1 scatter.py 2>&1 | tee 1.log