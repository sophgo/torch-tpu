分布式通信
============

本章以 ``all_reduce`` 为例，介绍如何使用 torch.distributed 包调用 SCCL (SOPHGO Collective Communication Library) 
进行集合通信。

本节默认使用者已经按照第2章说明完成torch-tpu环境配置安装。

分布式通信简介
----------

SOPHGO Collective Communication Library (SCCL) 是一种高性能 TPU-to-TPU 通信库，专为大规模并行计算环境设计。
它提供单机多卡间的通信功能，旨在提供高效、可扩展的数据传输解决方案，以支持复杂的并行计算任务。

Torch.distributed 包为多个计算设备的 PyTorch 提供多进程并行通信原语，可以跨进程和跨集群进行并行化计算。
Torch.distributed 支持 SCCL 内置后端，SCCL 通信后端能够充分利用处理器间的通信带宽，提升分布式训推性能。

.. list-table:: SCCL 后端通信函数支持列表

  * - 设备
    - TPU
  * - send
    - ✖
  * - recv
    - ✖
  * - broadcast
    - ✓
  * - all_reduce
    - ✓
  * - reduce
    - ✓
  * - all_gather
    - ✓
  * - gather
    - ✓
  * - scatter
    - ✓
  * - reduce_scatter
    - ✖
  * - all_to_all
    - ✓
  * - barrier
    - ✓

更多集合通信函数说明请参考https://pytorch.org/docs/stable/distributed.html#collective-functions。

分布式通信流程
----------

分布式通信流程流程如下：

1. 生成设备资源配置文件
2. 初始化SCCL后端
3. 调用集合通信函数

（1）准备设备资源配置文件:

以下为本地8个 TPU 设备生成配置文件的命令:

.. code-block:: shell

    # 设置生成配置文路径
    export RANK_TABLE_FILE=rank_table_file_path 
    tpu_gen_sccl_rank_table --device_id "[[0, 1, 2, 3, 4, 5, 6, 7]]"

其中 device_id 为 TPU 设备在计算节点上的序列号和 rank 的对应列表，如 ``device_id[0][0]`` 表示第一个节点的 rank 0对应的 TPU 设备ID。

（2）``all_reduce`` 原语示例代码:

.. code-block:: python

    import torch
    import torch.distributed as dist
    import os
    import torch_tpu
    TPU = "tpu"

    # get rank and world_size from env
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))

    # get chip_map
    if torch_tpu.tpu.is_rank_table_valid():
        chip_map = torch_tpu.tpu.read_rank_table()

    # init device
    torch_tpu.tpu.set_device(chip_map[rank])
    device = torch.device(f"{TPU}:{chip_map[rank]}")

    # init backend
    options = torch_tpu.ProcessGroupSCCLOptions()
    options.chip_map = chip_map
    dist.init_process_group(
        backend="sccl", 
        rank=rank, 
        world_size=world_size, 
        pg_options=options)

    # init tensor
    tensor_len = 4
    tensor = torch.ones(tensor_len).float()
    print("rank: {}, {}".format(rank, tensor))

    # converting a CPU Tensor with pinned memory to a TPU Tensor
    if torch_tpu.tpu.current_device() == device.index:
        tensor = tensor.to(device)

    # all_reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # print results
    results = tensor.cpu()
    print("rank: {}, results: {}".format(rank, results))

（3）启动多进程执行示例代码:

.. code-block:: shell

    torchrun --nproc_per_node 8 --nnodes 1 test_all_reduce_file_path

得到如下结果(8个 rank 结果都为 ``[8., 8., 8., 8.]``):

.. code-block:: shell

    rank: 6, results: tensor([8., 8., 8., 8.])
    rank: 2, results: tensor([8., 8., 8., 8.])
    rank: 0, results: tensor([8., 8., 8., 8.])
    rank: 5, results: tensor([8., 8., 8., 8.])
    rank: 4, results: tensor([8., 8., 8., 8.])
    rank: 3, results: tensor([8., 8., 8., 8.])
    rank: 1, results: tensor([8., 8., 8., 8.])
    rank: 7, results: tensor([8., 8., 8., 8.])

torchrun 具体参数说明和调用示例请参考https://pytorch.org/docs/stable/elastic/run.html。

（4）设备资源配置文件模板：

.. code-block:: json

    {
        "version":"1.0",
        "node_list":[
            {
                "device_list":[
                    {
                        "device_id":"0",
                        "rank":"0"
                    },
                    {
                        "device_id":"1",
                        "rank":"1"
                    }
                ]
            }
        ]
    }

（5）rank table 文件主要参数说明:

.. list-table:: rank table 文件说明
   :widths: 20 12 50
   :header-rows: 1

   * - 配置项
     - 必选？
     - 配置说明
   * - version
     - 是
     - rank table模板版本信息，当前仅支持配置为1.0。
   * - node_list
     - 是
     - 本次参与并行计算的节点列表
   * - device_list
     - 是
     - 本次参与并行计算的节点上的设备列表
   * - device_id
     - 是
     - tpu设备ID， 即 TPU 设备在计算节点上的序列号。取值范围：[0，实际device数量-1]
   * - rank
     - 是
     - 进程的全局排名。取值范围：[0，总device数量-1]

（6）init_process_group 主要参数说明:

.. list-table:: init_process_group 参数功能
   :widths: 20 10 50
   :header-rows: 1

   * - 参数名
     - 必选？
     - 说明
   * - backend
     - 是
     - 指定使用的后端，有效值为 ``sccl``。
   * - world_size
     - 是
     - 参与作业的总进程数。
   * - rank
     - 是
     - 进程的全局排名。取值范围：[0, world_size-1]。
   * - pg_options
     - 是
     - 进程组选项。SCCL 后端支持 ProcessGroupSCCL.Options 选项传入 chip_map 信息，用以进程和特定设备的对应。