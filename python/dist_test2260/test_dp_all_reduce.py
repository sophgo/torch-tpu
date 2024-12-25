import time
import torch
import torch.distributed as dist
import logging
import os
TPU = "tpu"
import torch_tpu

def init_logger():
    role = "master" 
    sccl_logger = logging.getLogger('sccl_logger')
    sccl_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("./pytorch-dist.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(f'{role.ljust(6)} ==> %(asctime)8s %(levelname)5s %(message)4s'))
    sccl_logger.addHandler(file_handler)

init_logger()
logger = logging.getLogger('sccl_logger')

# get RANK and WORLD_SIZE from env
RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
GROUP_PER_NODE = int(os.getenv("GROUP_PER_NODE", "1"))

LOCAL_WORLD_SIZE = WORLD_SIZE // GROUP_PER_NODE
LOCAL_RANK = RANK % LOCAL_WORLD_SIZE  

GROUP_IDX = (RANK)//LOCAL_WORLD_SIZE
os.environ["TPU_GROUP_IDX"] = str(GROUP_IDX)

options = torch_tpu.ProcessGroupSCCLOptions()
chip_map = [6,7,4,5]
chip_map = chip_map[GROUP_IDX * LOCAL_WORLD_SIZE: (GROUP_IDX+1)* LOCAL_WORLD_SIZE]
options.chip_map = chip_map
torch_tpu.tpu.set_device(chip_map[LOCAL_RANK])
os.environ["USE_CHIP"] =  ",".join(str(chip) for chip in chip_map)

logger.info(f'distribute initilizing........{RANK}/{WORLD_SIZE}')
torch.distributed.init_process_group(
    backend="sccl",
    world_size=LOCAL_WORLD_SIZE,
    rank=LOCAL_RANK,
    pg_options=options,
)


def case1():
    logger.info(f"LOCAL_RANK: {LOCAL_RANK}, LOCAL_WORLD_SIZE:{LOCAL_WORLD_SIZE}, RANK: {RANK}, WORLD_SIZE:{WORLD_SIZE},")
    
    device = torch.device(f"{TPU}:{chip_map[int(RANK)]}")
    tensor = torch.ones(4).to(device)
    torch.distributed.all_reduce(tensor, op=dist.ReduceOp.SUM)

    results = tensor.cpu()
    logger.info("rank: {}, results: {}".format(RANK, results))

if __name__ == "__main__":
    case1()

#  torchrun --nproc_per_node 4 --nnodes 1 my_t.py 