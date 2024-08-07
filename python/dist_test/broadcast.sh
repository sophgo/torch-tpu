#!/bin/bash

# python -m torch.distributed.launch \
#         --nnodes 1 \
#         --nproc_per_node 8 \
#         broadcast.py 2>&1 | tee 1.log

torchrun --nproc_per_node 8 --nnodes 1 broadcast.py  2>&1 | tee 1.log