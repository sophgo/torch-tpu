from dataclasses import dataclass, field
import torch

from typing import Optional


def get_dtypesize(dtype: torch.dtype) -> int:
    try:
        return torch.tensor(0, dtype=dtype).element_size()
    except Exception as e:
        breakpoint()
        return 4


KB = 1 << 10
MB = KB << 10
GB = MB << 10

T = 1 << 12
GHz = 1e9


@dataclass
class EstimationConfig:
    mac_utilization: float = 0.7
    ddr_utilization: float = 0.7


@dataclass
class ArchConfig:
    est: EstimationConfig = field(default_factory=EstimationConfig)

    @dataclass
    class Tiu:
        cube_m: int = 32  # 16
        cube_k: int = 64
        cube_n: int = 4
        macs_per_cycle = cube_m * cube_k * cube_n
        eu_num: int = 16  # 512
        lane_num: int = 32  # 16
        align_bytes: int = 64  # 32

    @dataclass
    class DDR:
        dram: float = 80.0 * GB
        dram_bw: float = 4000e9
        dma_bw: float = 128
        intra_bw: float = 500.0e9
        inter_bw: float = 40.0e9
        sram_size: int = 4 * (1 << 20)

    tiu: Tiu = field(default_factory=Tiu)
    ddr: DDR = field(default_factory=DDR)

    ddr_bw = 546
    core_num: int = 32
    flops: float = 256.0 * T
    discount_rate: float = 1.0
    tpu_gdma_overlap_rate: float = 0.8

    @property
    def flops_us(self) -> float:
        return self.flops / 1e9

    @property
    def dram_bw_us(self) -> float:
        return self.ddr.dram_bw / 1e9

    def dtyped_flops(self, dtype: torch.dtype) -> float:
        itemsize = get_dtypesize(dtype)
        if itemsize == 1:
            return self.flops
        if itemsize == 2:
            return self.flops // 2

        if itemsize == 4:
            if str(dtype) == "tf32":
                return self.flops // 4
            return self.flops // 16

    def dtyped_eunum(self, dtype: torch.dtype) -> int:
        itemsize = get_dtypesize(dtype)
        if itemsize == 1:
            return self.eu_num
        if itemsize == 2:
            return self.eu_num // 2
        if itemsize == 4:
            return self.eu_num // 16


SG2260Arch = ArchConfig(
    core_num=8, tiu=ArchConfig.Tiu(eu_num=16), ddr=ArchConfig.DDR(dram_bw=546)
)


SG2260eArch = ArchConfig(
    core_num=4, tiu=ArchConfig.Tiu(eu_num=16), ddr=ArchConfig.DDR(dram_bw=546)
)


@dataclass
class ReflectConfig:
    # call origin function
    call_func: bool = True
    # disable cmodel atomic
    disable_cmodel_atomic: bool = False  # TODO

    @dataclass
    class StoreCmdConfig:
        # use store_cmd
        store_cmd: bool = False
        # use store_cmd_inst
        store_cmd_base_path = "./"

    @dataclass
    class OpInferConfig:
        # use OpInfer to infer tensor
        tensor_infer: bool = True
        # use OpInfer to infer time
        time_infer: bool = True
        # find graind level of time infer
        time_infer_level: int = 0

    # use store_cmd
    store_cmd: StoreCmdConfig = field(default_factory=StoreCmdConfig)

    op_infer: Optional[OpInferConfig] = field(default_factory=OpInferConfig)


@dataclass
class ClusterConfig:
    """用于描述节点之间连接情况"""

    pass


@dataclass
class GlobalConfig:
    arch: ArchConfig
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    reflect: ReflectConfig = field(default_factory=ReflectConfig)


__global_config = GlobalConfig(
    arch=ArchConfig(),
    reflect=ReflectConfig(),
)  # default config


def get_global_config():
    return __global_config


def set_global_config(config: GlobalConfig):
    global __global_config
    __global_config = config
