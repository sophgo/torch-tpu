#!/bin/bash

# python -m torch.distributed.launch \
#         --nnodes=1 \
#         --nproc_per_node=8 \
#         test_send.py 2>&1 | tee 0.log

torchrun --nproc_per_node 2 --nnodes 1 test_send.py 2>&1 | tee 1.log