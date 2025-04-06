import argparse
from enum import Enum
from contextlib import contextmanager


class DisaggregationMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class GPUInfo():
    def __init__(self, name: str, sm, comm_sm, fp16_flops, fp8_flops, mem, mem_bw, nvlink_bw, pcie_bw, discount_rate):
        self.name = name
        self.sm = sm
        self.comm_sm = comm_sm  # 用于通信的SM数量
        self.fp16_flops = fp16_flops
        self.fp8_flops = fp8_flops
        self.mem = mem
        self.mem_bw = mem_bw
        self.nvlink_bw = nvlink_bw
        self.pcie_bw = pcie_bw
        self.discount_rate = discount_rate  # 整体性能按峰值性能折扣

    def get_fp16_flops(self):
        return self.fp16_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp8_flops(self):
        return self.fp8_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_mem_bw(self):
        return self.mem_bw * self.discount_rate

    def get_nvlink_bw(self):
        return self.nvlink_bw * self.discount_rate

    def get_pcie_bw(self):
        return self.pcie_bw * self.discount_rate

    def get_mem_size(self):
        return self.mem

# A800
A800_PREFILL = GPUInfo(name="A800", sm=108, comm_sm=10,
                      fp16_flops=499.2, fp8_flops=499.2, # not support fp8
                      mem=80, mem_bw=2*1024,
                      nvlink_bw=200, pcie_bw=50,
                      discount_rate=0.85)
A800_DECODE = GPUInfo(name="A800", sm=108, comm_sm=0,
                      fp16_flops=499.2, fp8_flops=499.2, # not support fp8
                      mem=80, mem_bw=2*1024,
                      nvlink_bw=200, pcie_bw=50,
                      discount_rate=0.85)

# H20
H20_PREFILL = GPUInfo(name="H20", sm=78, comm_sm=10,
                      fp16_flops=118.4, fp8_flops=236.8,
                      mem=96, mem_bw=4*1024,
                      nvlink_bw=450, pcie_bw=50,
                      discount_rate=0.85)

H20_DECODE = GPUInfo(name="H20", sm=78, comm_sm=0,
                     fp16_flops=118.4, fp8_flops=236.8,
                     mem=96, mem_bw=4*1024,
                     nvlink_bw=450, pcie_bw=50,
                     discount_rate=0.85)

# H800
H800_PREFILL = GPUInfo(name="H800", sm=132, comm_sm=20,
                       fp16_flops=791.6, fp8_flops=1583.2,
                       mem=80, mem_bw=3.35*1024,
                       nvlink_bw=200, pcie_bw=50,
                       discount_rate=0.85)


H800_DECODE = GPUInfo(name="H800", sm=132, comm_sm=0,
                      fp16_flops=791.6, fp8_flops=1583.2,
                      mem=80, mem_bw=3.35*1024,
                      nvlink_bw=200, pcie_bw=50,
                      discount_rate=0.85)

# H200
H200_PREFILL = GPUInfo(name="H200", sm=132, comm_sm=20,
                       fp16_flops=791.6, fp8_flops=1583.2,
                       mem=141, mem_bw=4.8*1024,
                       nvlink_bw=450, pcie_bw=50,
                       discount_rate=0.85)

H200_DECODE = GPUInfo(name="H200", sm=132, comm_sm=0,
                      fp16_flops=791.6, fp8_flops=1583.2,
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
    def __init__(self, args: argparse.Namespace):
        self.tp = args.tp
        self.dp = args.dp
        self.pp = args.pp
        self.ep = args.ep
        self.world_size = self.tp * self.dp * self.pp
        self.nnodes = args.nnodes
        self.batch_size = args.batch_size_per_device
        self.input_seq_len = args.input_seq_len  # ISL
        self.output_seq_len = args.output_seq_len  # OSL
        self.kv_cache_hit_rate = args.kv_cache_hit_rate
        self.dispatch_node = args.dispatch_node
        self.overlap = args.overlap
        self.disaggregation_mode: DisaggregationMode = DisaggregationMode(
            args.disaggregation_mode)
        self.mem_fraction_static = args.mem_fraction_static

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
    def __init__(self, gpu_info: GPUInfo, args: ServerArgs, throughput: float, detail: dict[str, float]):
        self.gpu_info = gpu_info
        self.args = args
        self.throughput = throughput
        self.detail = detail

    def get_gpu_info(self):
        return self.gpu_info

    def get_args(self):
        return self.args

    def get_throughput(self):
        return self.throughput

    def get_detail(self):
        return self.detail


@contextmanager
def disaggregation_mode(args: ServerArgs):
    if args.disaggregation_mode == DisaggregationMode.PREFILL:
        output_seq_len = args.output_seq_len
        args.output_seq_len = 1

    try:
        yield
    finally:
        if args.disaggregation_mode == DisaggregationMode.PREFILL:
            args.output_seq_len = output_seq_len
