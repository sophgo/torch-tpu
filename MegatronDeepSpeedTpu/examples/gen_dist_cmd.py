import argparse
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=str, choices=["bert", "gpt", "llama"])
    parser.add_argument("--master_addr", type=str, default=None)
    parser.add_argument("--master_port", type=int, default=6000)
    # parser.add_argument("--local_size", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--device", type=str, choices=["tpu", "cuda"], default="tpu")
    parser.add_argument("--byobu", type=str, default="default")
    parser.add_argument("--cmodel_dist_addrs", type=str, default="172.28.143.14,172.28.143.8,172.28.143.9,172.28.143.36")
    parser.add_argument("--extra_flags", type=str, default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    assert args.world_size % args.num_nodes == 0, "world size must be divisible by num nodes"
    real_local_size = args.world_size // args.num_nodes
    LOCAL_RANK={"cuda": 0, "tpu": "$i"}[args.device] # f"$((i % {args.local_size}))" # we treat all local size as 1 to avoid confusion
    REAL_LOCAL_RANK=f"$((i % {real_local_size}))" # for logging and CUDA_VISIBLE_DEVICES
    CMODEL_DIST_ADDRS=args.cmodel_dist_addrs
    if args.num_nodes > 1:
        ip_list = CMODEL_DIST_ADDRS.split(",")
        assert len(ip_list) <= args.num_nodes, "number of cmodel dist addrs must be equal to num nodes"
        CMODEL_DIST_ADDRS = ",".join(ip_list[:args.num_nodes])
    elif args.num_nodes == 1:
        CMODEL_DIST_ADDRS = ""
    ENABLE_FRAMEWORK_DEBUGGER=1
    DS_DISABLE_TPU=1
    CUDA_VISIBLE_DEVICES=REAL_LOCAL_RANK
    RANK="$i"
    OMPI_COMM_WORLD_RANK=RANK
    WORLD_SIZE=args.world_size
    OMPI_COMM_WORLD_SIZE=WORLD_SIZE
    LOCAL_SIZE={"cuda": 1, "tpu": WORLD_SIZE}[args.device] # args.local_size
    MASTER_ADDR=args.master_addr
    if MASTER_ADDR is None:
        MASTER_ADDR=ip_list[0] if args.num_nodes > 1 else "127.0.0.1"
    MASTER_PORT=args.master_port + int(args.device == "tpu")
    
    byobu = ""
    if args.byobu == "default":
        byobu = {"tpu": "tpu-train:DeepSpeed",
                 "cuda": "tpu-train:deepspeed"}[args.device]
    else:
        byobu = args.byobu
    assert ":" in byobu, "byobu target must be in the form of 'session:window'"
    
    flags = f"{DS_DISABLE_TPU=} {CUDA_VISIBLE_DEVICES=} {ENABLE_FRAMEWORK_DEBUGGER=}" if args.device == 'cuda' else f"{CMODEL_DIST_ADDRS=}"
    flags += " " + args.extra_flags
    dist_env = f"{RANK=} {LOCAL_RANK=} {WORLD_SIZE=} {LOCAL_SIZE=} {REAL_LOCAL_RANK=} {MASTER_ADDR=} {MASTER_PORT=}"
    dist_env += f" {OMPI_COMM_WORLD_RANK=} {OMPI_COMM_WORLD_SIZE=}" if args.device == 'tpu' else ""
    case_cmd = {
        "bert": f"python -u train_bert_ds.py --local_rank={LOCAL_RANK} --checkpoint_dir experiment_deepspeed",
        "gpt": "bash run_distributed_gpt_train.sh",
        "llama": "bash run_distributed_llama2_train.sh"
    }[args.case]
    
    gen_each_server_cmd = f"""for i in `seq 0 {args.world_size - 1}`; do if [ $((i % {real_local_size})) -eq 0 ]; then echo ----- Command to run on server$((i / {real_local_size} + 1)): -----; fi; echo "{flags} {dist_env} {case_cmd} &"; done"""
    os.system(gen_each_server_cmd)
    
    cmd = f"""for i in `seq 0 {args.world_size - 1}`; do ssh server$((i / {real_local_size} + 1)) "byobu send-keys -t {byobu} \\"{flags} {dist_env} {case_cmd} & \\" Enter"; done"""
    print("----- Remote Byobu command: -----")
    print(cmd)
    if args.execute:
        os.system(cmd)
        print("Byobu Command executed.")
