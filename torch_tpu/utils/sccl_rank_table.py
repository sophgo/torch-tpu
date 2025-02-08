import os
import json
import sys
import argparse
import ast

def get_device_id(device_id: str)->list:
    try:
        device_id = ast.literal_eval(device_id)
    except (ValueError, SyntaxError) as e:
        print("Invalid input format. Please provide a valid nested list.")
        print(f"Error: {e}")
        sys.exit(1)
    return device_id

def get_device_ip(device_ip: str)->list:
    try:
        device_ip = ast.literal_eval(device_ip)
        if isinstance(device_ip, list) and all(isinstance(sublist, list) for sublist in device_ip):
            print("IP Addresses:", device_ip)
        else:
            raise ValueError("Input must be a nested list.")
    except (ValueError, SyntaxError) as e:
        print("Invalid input format. Please provide a valid nested list.")
        print(f"Error: {e}")
        sys.exit(1)
    return device_ip

def gen_sccl_json(device_id: list, emulator: bool = False, device_ip: list = [["127.0.0.1"] * 8]):
    """
    generate a TPU resource configuration file.

    Example:

    #single-machine multi-card config
    {
        "version":"1.0", // Template version information, currently must be "1.0"
        "node_list":[
            {
                "device_list":[ // Device list in Host Server
                    {
                        "device_id":"0", // Device ID
                        "device_ip_emulator":"127.0.0.1", // The real network card IP of the device host when emulator is True
                        "local_rank":"0", // The rank ID, rank_id starts from 0
                        "rank":"0"
                    },
                    {
                        "device_id":"7",
                        "device_ip_emulator":"127.0.0.1",
                        "local_rank":"1",
                        "rank":"1"
                    }
                ]
            }
        ]
    }
    """

    version = "1.0"
    path = os.environ.get("RANK_TABLE_FILE") if os.environ.get("RANK_TABLE_FILE") else 'sccl_rank_table.json'
    device_id = get_device_id(device_id)
    if device_ip != [["127.0.0.1"] * 8]:
        device_ip = get_device_ip(device_ip)
    assert len(device_id) <= len(device_ip)

    node_list, device_list = [], []
    sum = 0

    for i in range(len(device_id)):
        assert len(device_id[i]) <= len(device_ip[i]), f"the num of device_id and the num of device_ip is mismatch at index {i}"
        device_ip_key = "device_ip_emulator" if emulator else "device_ip"
        device_list = [{"device_id": device_id[i][j], device_ip_key: device_ip[i][j], "local_rank": f"{j}", "rank": f"{sum+j}"} for j in range(len(device_id[i]))]
        sum += len(device_id[i])
        node_list.append({"device_list": device_list})

    sccl_config = {"version": version, "node_list": node_list}
    print(sccl_config)

    # write json
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(sccl_config, f, indent=4, separators=(',', ':'))

def gen_sccl_rank_table():
    # only support single-machine multi-card now
    parser = argparse.ArgumentParser(description='Gen sccl rank table')
    parser.add_argument('--device_id', type=str, help='device id list')
    parser.add_argument('--emulator', type=bool, help='option for using multi nodes to emulator single-machine multi-card',
                        default=False)
    parser.add_argument('--device_ip', type=str, help='device ip list', default=[["127.0.0.1"] * 8])
    args = parser.parse_args()

    gen_sccl_json(args.device_id, args.emulator, args.device_ip)

# python3 torch_tpu/tpu/gen_sccl_rank_table.py --device_id "[['0', '1', '2', '3', '4', '5', '6', '7']]" --device_ip "[['127.0.0.1', '127.0.0.1', '127.0.0.1', '127.0.0.1', '127.0.0.1', '127.0.0.1', '127.0.0.1', '127.0.0.1']]"
# tpu_gen_sccl_rank_table --device_id "[['0', '1', '2', '3', '4', '5', '6', '7']]" --device_ip "[['127.0.0.1', '127.0.0.1', '127.0.0.1', '127.0.0.1', '127.0.0.1', '127.0.0.1', '127.0.0.1', '127.0.0.1']]"