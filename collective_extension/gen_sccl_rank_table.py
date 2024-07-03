import os
import json


def gen_sccl_json(server_ip: list, device_id: list, device_ip: list, path: str):
    """
    generate a TPU resource configuration file.

    Example:
    #single-machine multi-card config

    {
        "status":"completed", // Available flag, completed means available
        "version":"1.0", // Template version information, currently must be "1.0"
        "server_count":1, // The number of host servers participating in the training.
        "server_list":[
            {
                "server_ip":"10.0.0.10", // Host Server ID, expressed as an IP string in dotted decimal notation
                "device":[ // Device list in Host Server
                    {
                        "device_id":"0", // Device ID
                        "device_ip":"192.1.27.6", // The real network card IP of the device
                        "rank_id":"0" // The rank ID, rank_id starts from 0
                    },
                    {
                        "device_id":"7",
                        "device_ip":"192.1.27.6",
                        "rank_id":"1"
                    }
                ],
                "host_nic_ip":"reserve"
            }
        ]
    }
    """
    server_count = 1 # int
    version = "1.0"
    host_nic_ip = "reserve"
    assert server_count == len(server_ip)
    assert len(device_id) == len(device_ip) == server_count

    server_list, device_list = [], []
    sum = 0
    for i in range(len(device_id)):
        assert len(device_id[i]) == len(device_ip[i]), f"the num of device_id and the num of device_ip is mismatch at index {i}"
        device_list = [{"device_id": device_id[i][j], "device_ip": device_ip[i][j], "rank_id": f"{sum+j}"} for j in range(len(device_id[i]))]
        sum += len(device_id[i])
        server_list.append({"server_ip": server_ip[i], "device": device_list, "host_nic_ip": host_nic_ip})

    sccl_config = {"status":"completed", "version": version, "server_count":server_count, "server_list": server_list}
    print(sccl_config)

    # write json
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(sccl_config, f, indent=4, separators=(',', ':'))

if __name__ == "__main__":
    # only support single-machine multi-card
    # export RANK_TABLE_FILE='sccl_rank_table.json'

    server_ip = ["172.17.0.11"]
    device_id = [["0", "1", "2", "3", "4", "5", "6", "7"]]
    device_ip = [["172.17.0.11"] * 8]
    path = os.environ.get("RANK_TABLE_FILE") if os.environ.get("RANK_TABLE_FILE") else 'sccl_rank_table.json'
    gen_sccl_json(server_ip, device_id, device_ip, path)
