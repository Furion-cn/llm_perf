import argparse
from enum import Enum
from contextlib import contextmanager
import math
from typing import Optional


class DisaggregationMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class GPUOp():
    def __init__(self, discount_rate: float):
        self.discount_rate = discount_rate

    def get_discount_rate(self):
        return self.discount_rate


def n_pow2_range(n: int):
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n = n+1
    return n


class GroupGemm(GPUOp):
    def __init__(self, m_per_group: int):
        super().__init__(1)
        self.m_per_group = m_per_group

    def get_discount_rate(self):
        flops_discounts = {
            1: 0.05,
            2: 0.05,
            4: 0.05,
            8: 0.05,
            16: 0.08,
            32: 0.1,
            64: 0.2,
            128: 0.35,
            256: 0.4,
            512: 0.6,
            1024: 0.7,
            2048: 0.7,
            4096: 0.7,
            8192: 0.7,
            16384: 0.7,
            32768: 0.7,
            65536: 0.7
        }
        return flops_discounts[n_pow2_range(int(self.m_per_group))]


class AllReduce(GPUOp):
    def __init__(self, device_num: int, size: int):
        super().__init__(1)
        self.device_num = device_num
        self.size = size

    def get_discount_rate(self):
        if self.device_num == 1:
            return 1

        assert self.device_num <= 8, "only for intra-node allreduce"

        discount_rate = 0.85 * self.device_num / (self.device_num - 1) / 2

        v = 0.00045
        if self.size < 512 * 1024:
            v *= 2 ** (math.log2(self.size / 1024))
        elif self.size < 1024 * 1024:
            v = 0.3
        elif self.size < 2*1024 * 1024:
            v = 0.35
        elif self.size < 4*1024 * 1024:
            v = 0.5
        elif self.size < 8*1024 * 1024:
            v = 0.67
        elif self.size < 16*1024 * 1024:
            v = 0.73
        elif self.size < 32*1024 * 1024:
            v = 0.88
        elif self.size < 64*1024 * 1024:
            v = 0.93
        elif self.size < 128*1024 * 1024:
            v = 0.95
        elif self.size < 256*1024 * 1024:
            v = 0.97
        elif self.size < 512*1024 * 1024:
            v = 0.98
        else:
            v = 1

        return discount_rate * v


class AllToAll(GPUOp):
    def __init__(self, device_num: int):
        super().__init__(1)
        self.device_num = device_num

    def get_discount_rate(self):
        m = n_pow2_range(int(self.device_num))
        if m == 1:
            return 1
        elif m < 8:
            return 0.95
        elif m == 8:
            return 0.92
        elif m == 16:
            return 0.86
        elif m == 32:
            return 0.82
        elif m == 64:
            return 0.80
        else:
            return 0.78


class DType(Enum):
    FP16 = "fp16"
    FP8 = "fp8"


class GPUInfo():
    def __init__(self, name: str, sm, comm_sm, flops_map: dict[DType, float], mem, mem_bw, nvlink_bw, pcie_bw, discount_rate):
        self.name = name
        self.sm = sm
        self.comm_sm = comm_sm  # 用于通信的SM数量
        self.flops_map = flops_map
        self.mem = mem
        self.mem_bw = mem_bw
        self.nvlink_bw = nvlink_bw
        self.pcie_bw = pcie_bw
        self.discount_rate = discount_rate  # 整体性能按峰值性能折扣

    def get_flops(self, op: Optional[GPUOp] = None, dtype: Optional[DType] = None):
        if op is None:
            op = GPUOp(self.discount_rate)
        if dtype is None:
            dtype = DType.FP16
        if dtype not in self.flops_map:
            raise ValueError(f"Invalid dtype: {dtype}")
        return self.flops_map[dtype] * op.get_discount_rate() * (self.sm - self.comm_sm) / self.sm

    def get_mem_bw(self):
        return self.mem_bw * self.discount_rate

    def support_dtype(self, dtype: DType):
        return dtype in self.flops_map

    def get_pcie_bw(self, op: Optional[GPUOp] = None):
        if op is None:
            op = GPUOp(self.discount_rate)
        return self.pcie_bw * op.get_discount_rate()

    def get_mem_size(self):
        return self.mem - 4 - 1.5 # rdma network buffer and cuda runner cache

    def get_nvlink_bw(self, op: Optional[GPUOp] = None):
        if op is None:
            op = GPUOp(self.discount_rate)
        return self.nvlink_bw * op.get_discount_rate()


# A800
A800_PREFILL = GPUInfo(name="A800", sm=108, comm_sm=10,
                       # not support fp8
                       flops_map={DType.FP16: 499.2},
                       mem=80, mem_bw=2*1024,
                       nvlink_bw=200, pcie_bw=50,
                       discount_rate=0.78)
A800_DECODE = GPUInfo(name="A800", sm=108, comm_sm=0,
                      # not support fp8
                      flops_map={DType.FP16: 499.2},
                      mem=80, mem_bw=2*1024,
                      nvlink_bw=200, pcie_bw=50,
                      discount_rate=0.78)

# H20
H20_PREFILL = GPUInfo(name="H20", sm=78, comm_sm=10,
                      flops_map={DType.FP16: 118.4, DType.FP8: 236.8},
                      mem=96, mem_bw=4*1024,
                      nvlink_bw=450, pcie_bw=50,
                      discount_rate=0.85)

H20_DECODE = GPUInfo(name="H20", sm=78, comm_sm=0,
                     flops_map={DType.FP16: 118.4, DType.FP8: 236.8},
                     mem=96, mem_bw=4*1024,
                     nvlink_bw=450, pcie_bw=50,
                     discount_rate=0.85)

# H800
H800_PREFILL = GPUInfo(name="H800", sm=132, comm_sm=20,
                       flops_map={DType.FP16: 791.6, DType.FP8: 1583.2},
                       mem=80, mem_bw=3.35*1024,
                       nvlink_bw=200, pcie_bw=50,
                       discount_rate=0.85)


H800_DECODE = GPUInfo(name="H800", sm=132, comm_sm=0,
                      flops_map={DType.FP16: 791.6, DType.FP8: 1583.2},
                      mem=80, mem_bw=3.35*1024,
                      nvlink_bw=200, pcie_bw=50,
                      discount_rate=0.85)

# H200
H200_PREFILL = GPUInfo(name="H200", sm=132, comm_sm=20,
                       flops_map={DType.FP16: 791.6, DType.FP8: 1583.2},
                       mem=141, mem_bw=4.8*1024,
                       nvlink_bw=450, pcie_bw=50,
                       discount_rate=0.85)

H200_DECODE = GPUInfo(name="H200", sm=132, comm_sm=0,
                      flops_map={DType.FP16: 791.6, DType.FP8: 1583.2},
                      mem=141, mem_bw=4.8*1024,
                      nvlink_bw=450, pcie_bw=50,
                      discount_rate=0.85)


GPU_INFO_MAP = {
    "A800": {
        DisaggregationMode.PREFILL: A800_PREFILL,
        DisaggregationMode.DECODE: A800_DECODE
    },
    "H20": {
        DisaggregationMode.PREFILL: H20_PREFILL,
        DisaggregationMode.DECODE: H20_DECODE
    },
    "H800": {
        DisaggregationMode.PREFILL: H800_PREFILL,
        DisaggregationMode.DECODE: H800_DECODE
    },
    "H200": {
        DisaggregationMode.PREFILL: H200_PREFILL,
        DisaggregationMode.DECODE: H200_DECODE
    }
}


def get_gpu_info(gpu_type: str, disaggregation_mode: DisaggregationMode) -> GPUInfo:
    if gpu_type not in GPU_INFO_MAP:
        raise ValueError(f"Invalid GPU type: {gpu_type}")
    if disaggregation_mode not in GPU_INFO_MAP[gpu_type]:
        raise ValueError(f"Invalid disaggregation mode: {disaggregation_mode}")
    return GPU_INFO_MAP[gpu_type][disaggregation_mode]


class ServerArgs():
    def __init__(self,
                 tp: int,
                 dp: int,
                 pp: int,
                 ep: int,
                 nnodes: int,
                 batch_size: int,
                 input_seq_len: int,
                 output_seq_len: int,
                 kv_cache_hit_rate: float,
                 dispatch_node: int,
                 overlap: bool,
                 disaggregation_mode: DisaggregationMode,
                 mem_fraction_static: float,
                 ):
        self.tp = tp
        self.dp = dp
        self.pp = pp
        self.ep = ep
        self.nnodes = nnodes
        self.world_size = tp * dp * pp
        self.batch_size = batch_size
        self.input_seq_len = input_seq_len  # ISL
        self.output_seq_len = output_seq_len  # OSL
        self.kv_cache_hit_rate = kv_cache_hit_rate
        self.dispatch_node = dispatch_node
        self.overlap = overlap
        self.disaggregation_mode: DisaggregationMode = DisaggregationMode(
            disaggregation_mode)
        self.mem_fraction_static = mem_fraction_static

        assert self.world_size % 8 == 0, f"world_size must be divisible by 8, but got {self.world_size}"

        if self.nnodes <= 0:
            self.nnodes = self.world_size // 8

        assert self.ep <= self.world_size, f"ep must be less than world_size, but got {self.ep} and {self.world_size}"
        assert self.ep > 0, f"ep must be greater than 0, but got {self.ep}"
        assert self.kv_cache_hit_rate >= 0 and self.kv_cache_hit_rate <= 1, f"kv_cache_hit_rate must be between 0 and 1, but got {self.kv_cache_hit_rate}"

    def get_nnodes(self):
        return self.tp * self.dp * self.pp / 8

    def get_world_size(self):
        return self.tp * self.dp * self.pp

    def is_decode_mode(self):
        return self.disaggregation_mode == DisaggregationMode.DECODE

    def is_prefill_mode(self):
        return self.disaggregation_mode == DisaggregationMode.PREFILL


class Throughput():
    def __init__(self, gpu_info: GPUInfo, args: ServerArgs, throughput: float, elapse_time: float, detail: dict[str, float]):
        self.gpu_info = gpu_info
        self.args = args
        self.throughput = throughput
        self.elapse_time = elapse_time
        self.detail = detail

    def get_gpu_info(self):
        return self.gpu_info

    def get_args(self):
        return self.args

    def get_throughput(self):
        return self.throughput

    def get_elapse_time(self):
        return self.elapse_time

    def get_detail(self):
        return self.detail
