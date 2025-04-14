from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
from structs.structs import GPUInfo, ServerArgs, Throughput, DisaggregationMode, DType, AllReduce, GroupGemm, AllToAll


class Model:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    def get_mla_flops(self, q_len, kv_len, kv_cache_hit_rate: float):
        q_down_proj = q_len * self.dim * self.q_lora_rank
        q_up_proj = q_len * self.q_lora_rank * self.n_heads * \
            (self.qk_nope_head_dim + self.qk_rope_head_dim)
        kv_down_proj = kv_len * self.dim * \
            (self.kv_lora_rank + self.qk_rope_head_dim)
        k_up_proj = kv_len * self.kv_lora_rank * \
            self.n_heads * self.qk_nope_head_dim
        v_up_proj = kv_len * self.kv_lora_rank * self.n_heads * self.v_head_dim

        kv_down_proj = kv_down_proj * (1 - kv_cache_hit_rate)
        gemm_sum = q_down_proj + q_up_proj + kv_down_proj + k_up_proj + v_up_proj

        mha = self.n_heads * (q_len * self.qk_rope_head_dim * kv_len
                              + q_len * self.qk_nope_head_dim * kv_len
                              + q_len * kv_len * self.v_head_dim)
        wo = q_len * self.n_heads * self.v_head_dim * self.dim
        attn_sum = mha + wo

        # return flops by 2* Sum(MACs)
        GEMM_FP8_FLOPS = gemm_sum * 2/1e9
        ATTN_FP16_FLOPS = attn_sum * 2/1e9

        return GEMM_FP8_FLOPS + ATTN_FP16_FLOPS, GEMM_FP8_FLOPS, ATTN_FP16_FLOPS

    def get_mla_mat_absob_flops(self, q_len, kv_len, kv_cache_hit_rate: float):
        q_down_proj = q_len * self.dim * self.q_lora_rank
        q_rope_up_proj = q_len * self.q_lora_rank * \
            self.n_heads * self.qk_rope_head_dim
        q_absorb = q_len * self.n_heads * (self.q_lora_rank * self.qk_nope_head_dim
                                           + self.qk_nope_head_dim * self.kv_lora_rank)

        kv_down_proj = kv_len * self.dim * \
            (self.kv_lora_rank + self.qk_rope_head_dim)
        kv_down_proj = kv_down_proj * (1 - kv_cache_hit_rate)
        gemm_sum = q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj

        mqa = self.n_heads * (q_len * self.qk_rope_head_dim * kv_len
                              + q_len * self.kv_lora_rank * kv_len
                              + q_len * kv_len * self.kv_lora_rank)

        attn_up_proj = q_len * self.n_heads * self.v_head_dim * self.kv_lora_rank
        o_proj = q_len * self.n_heads * self.v_head_dim * self.dim
        attn_sum = mqa + attn_up_proj + o_proj

        gemm_sum = gemm_sum * 2/1e9
        attn_sum = attn_sum * 2/1e9

        return gemm_sum + attn_sum, gemm_sum, attn_sum

    def get_moe_expert_flops(self, seq_len):
        return 3 * seq_len * self.dim * self.moe_inter_dim * 2/1e9

    def get_densemlp_flops(self, seq_len):
        return 3 * seq_len * self.dim * self.inter_dim * 2 / 1e9

    def get_mla_weight_load_time(self, gpu: GPUInfo):
        q_down_proj = self.dim * self.q_lora_rank  # wq_a
        q_up_proj = self.q_lora_rank * self.n_heads * \
            (self.qk_nope_head_dim + self.qk_rope_head_dim)  # wq_b
        kv_down_proj = self.dim * \
            (self.kv_lora_rank + self.qk_rope_head_dim)  # wkv_a
        k_up_proj = self.kv_lora_rank * self.n_heads * self.qk_nope_head_dim  # w_uk
        v_up_proj = self.kv_lora_rank * self.n_heads * self.v_head_dim  # w_uv
        wo = self.n_heads * self.v_head_dim * self.dim  # wo
        return (q_down_proj + q_up_proj + k_up_proj + kv_down_proj + v_up_proj + wo) / 1024/1024/1024 / gpu.get_mem_bw() * 1000

    def get_mla_elapse_time(self, gpu: GPUInfo, tp: int, dp: int, bs: int, seq_len: int, kv_cache_hit_rate: float, disaggregation_mode: DisaggregationMode):
        if disaggregation_mode == DisaggregationMode.DECODE:
            all_reduce_comm_size = bs / dp * self.dim * 2
            _, gemm_fp8_flops, attn_fp16_flops = self.get_mla_mat_absob_flops(
                1, seq_len, 1)
        else:
            all_reduce_comm_size = seq_len * bs / dp * self.dim * 2
            _, gemm_fp8_flops, attn_fp16_flops = self.get_mla_flops(
                seq_len, seq_len, kv_cache_hit_rate)
        gemm_fp8_time = gemm_fp8_flops / \
            gpu.get_flops(dtype=DType.FP8)/0.7 * bs / \
            dp  # 0.7 based on FlashMLA result on H800
        attn_fp16_time = attn_fp16_flops / \
            gpu.get_flops(dtype=DType.FP16)/0.7 * bs/dp
        compute_elapsed_time = gemm_fp8_time + attn_fp16_time
        if tp > 1:
            ar_elapsed_time = all_reduce_comm_size / \
                gpu.get_nvlink_bw(
                    AllReduce(tp, all_reduce_comm_size)) / 1024/1024 + 0.015  # static latency 0.015 ms
        else:
            ar_elapsed_time = 0
        mla_kernel_static_time = 0.05
        kv_cache_load_time = self.get_kv_cache_load_time(
            gpu, dp, bs, seq_len)
        weight_load_time = self.get_mla_weight_load_time(gpu)
        return compute_elapsed_time + ar_elapsed_time + kv_cache_load_time + weight_load_time + mla_kernel_static_time, ar_elapsed_time, kv_cache_load_time, weight_load_time

    def get_dense_mlp_elapse_time(self, gpu: GPUInfo, seq_len: int):
        gemm_fp8_flops = self.get_densemlp_flops(seq_len)
        gemm_fp8_time = gemm_fp8_flops / gpu.get_flops(dtype=DType.FP8)
        return gemm_fp8_time

    def get_moe_expert_elapse_time(self, gpu: GPUInfo, ep: int, seq_len: int, overlap: bool):
        if overlap:
            seq_len = seq_len / 2
        load_model_time = self.get_moe_expert_size() / 1024/1024/1024 / \
            gpu.get_mem_bw() * 1000
        shared_flops = self.get_moe_expert_flops(seq_len)
        shared_time = shared_flops / gpu.get_flops(op=GroupGemm(seq_len), dtype=DType.FP8) + \
            load_model_time * self.n_shared_experts
        routed_flops = self.get_moe_expert_flops(
            seq_len * self.n_activated_experts)
        expert_num_per_device = math.ceil(self.n_routed_experts / ep)
        routed_time = routed_flops / gpu.get_flops(op=GroupGemm(seq_len * self.n_activated_experts / expert_num_per_device), dtype=DType.FP8) + \
            load_model_time * expert_num_per_device
        if overlap:
            return 2 * shared_time, 2 * routed_time
        return shared_time, routed_time

    def get_alltoall_elapse_time(self, gpu: GPUInfo, ep: int, dispatch_size: int):
        dispatch_size = dispatch_size * self.dim / 1024/1024
        combine_size = 2 * dispatch_size  # fp16
        comm_bw = gpu.get_pcie_bw(op=AllToAll(ep))
        dispatch_time = dispatch_size / comm_bw
        combine_time = combine_size / comm_bw
        return dispatch_time, combine_time

    def get_kv_cache_load_time(self, gpu: GPUInfo, dp: int, bs: int, seq_len: int):
        return bs/dp * seq_len * (
            self.kv_lora_rank + self.qk_rope_head_dim) / 1024/1024/1024 / gpu.get_mem_bw() * 1000

    def get_prefill_elapse_time(self,
                                gpu: GPUInfo,
                                tp: int,
                                dp: int,
                                ep: int,
                                bs: int,
                                input_seq_len: int,
                                kv_cache_hit_rate: float,
                                world_size: int,
                                dispatch_node: int,
                                overlap: bool):
        mla, mla_ar, _, _ = self.get_mla_elapse_time(
            gpu, tp, dp, bs, input_seq_len, kv_cache_hit_rate, DisaggregationMode.PREFILL)
        dense_mlp = self.get_dense_mlp_elapse_time(
            gpu, bs*input_seq_len/dp)
        shared, routed = self.get_moe_expert_elapse_time(
            gpu, ep, dp * input_seq_len / world_size, overlap)
        dispatch, combine = self.get_alltoall_elapse_time(
            gpu, ep, (dispatch_node - 1) * (8/tp) * input_seq_len)
        if overlap:
            elapsed_time = self.n_layers * mla + self.n_dense_layers * dense_mlp + \
                (self.n_layers - self.n_dense_layers) * (shared + routed)
        else:
            elapsed_time = self.n_layers * mla + self.n_dense_layers * dense_mlp + \
                (self.n_layers - self.n_dense_layers) * \
                (shared + routed + dispatch + combine)
        return elapsed_time, dense_mlp, mla, mla_ar, shared, routed, dispatch, combine

    def get_decode_elapse_time(self,
                               gpu: GPUInfo,
                               tp: int,
                               dp: int,
                               ep: int,
                               bs: int,
                               input_seq_len: int,
                               output_seq_len: int,
                               kv_cache_hit_rate: float,
                               world_size: int,
                               overlap: bool):
        mla, mla_ar, mla_kv_cache_load_time, mla_weight_load_time = self.get_mla_elapse_time(
            gpu, tp, dp, bs, input_seq_len + output_seq_len/2, kv_cache_hit_rate, DisaggregationMode.DECODE)
        mlp = self.get_dense_mlp_elapse_time(gpu, bs/dp)
        shared, routed = self.get_moe_expert_elapse_time(
            gpu, ep, bs / world_size, overlap)
        dispatch, combine = self.get_alltoall_elapse_time(
            gpu, ep, bs / world_size * self.n_activated_experts)
        if overlap:
            elapsed_time = self.n_dense_layers * (mlp + mla) + (
                self.n_layers - self.n_dense_layers) * (max(mla + shared + routed, dispatch + combine))
        else:
            elapsed_time = self.n_layers * mla + self.n_dense_layers * mlp + (
                self.n_layers - self.n_dense_layers) * (shared + routed + dispatch + combine)
        return elapsed_time, mlp, mla, mla_ar, mla_kv_cache_load_time, mla_weight_load_time, shared, routed, dispatch, combine

    def get_prefill_throughput(self, gpu: GPUInfo, args: ServerArgs) -> list[Throughput]:
        elapse_time, mlp, mla, mla_ar, shared, routed, dispatch, combine = self.get_prefill_elapse_time(
            gpu, tp=args.tp, dp=args.dp, bs=args.batch_size, input_seq_len=args.input_seq_len, kv_cache_hit_rate=args.kv_cache_hit_rate, world_size=args.world_size, dispatch_node=args.dispatch_node, overlap=args.overlap)
        return Throughput(
            gpu,
            args,
            args.dp * args.input_seq_len * 1000 / elapse_time / args.get_nnodes(),
            elapse_time,
            {
                "BatchSize": args.batch_size,
                "DenseMLP(ms)": mlp,
                "MLA(ms)": mla,
                "MLA_AllReduce(ms)": mla_ar,
                "Shared Expert(ms)": shared,
                "Routed Expert(ms)": routed,
                "Dispatch(ms)": dispatch,
                "Combine(ms)": combine,
                "ElapseTime(ms)": elapse_time
            }
        )

    def get_decode_throughput(self, gpu: GPUInfo, args: ServerArgs) -> list[Throughput]:
        elapse_time, mlp, mla, mla_ar, mla_kv_cache_load_time, mla_weight_load_time, shared, routed, dispatch, combine = self.get_decode_elapse_time(
            gpu, tp=args.tp, dp=args.dp, ep=args.ep, bs=args.batch_size, input_seq_len=args.input_seq_len, output_seq_len=args.output_seq_len, kv_cache_hit_rate=args.kv_cache_hit_rate, world_size=args.world_size, overlap=args.overlap)
        dtype_mem_size = 1 if gpu.support_dtype(DType.FP8) else 2
        return Throughput(
            gpu,
            args,
            1000 / elapse_time * args.batch_size,
            elapse_time,
            {
                "BatchSize": args.batch_size,
                "DeviceModelSize(GB)": self.get_device_model_size(args.tp, args.ep) * dtype_mem_size / 1024/1024/1024,
                "DenseMLP(ms)": self.n_dense_layers * mlp,
                "MLA(ms)": self.n_layers * mla,
                "MLA_AllReduce(ms)": self.n_layers * mla_ar,
                "MLA KV Cache Load(ms)": self.n_layers * mla_kv_cache_load_time,
                "MLA Weight Load(ms)": self.n_layers * mla_weight_load_time,
                "Shared Expert(ms)": (
                    self.n_layers - self.n_dense_layers) * shared,
                "Routed Expert(ms)": (
                    self.n_layers - self.n_dense_layers) * routed,
                "Dispatch(ms)": (
                    self.n_layers - self.n_dense_layers) * dispatch,
                "Combine(ms)": (
                    self.n_layers - self.n_dense_layers) * combine,
                "ElapseTime(TPOT, ms)": elapse_time,
            }
        )

    def get_embeding_size(self):
        return self.vocab_size * self.dim

    def get_mla_size(self):
        return (
            self.dim * self.q_lora_rank + self.q_lora_rank + self.q_lora_rank * self.n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim) +
            self.dim * (self.kv_lora_rank + self.qk_rope_head_dim) + self.kv_lora_rank + self.kv_lora_rank * self.n_heads * (self.qk_nope_head_dim + self.qk_nope_head_dim) +
            self.dim * self.n_heads * self.v_head_dim +
            self.dim * 2
        )

    def get_dense_mlp_size(self):
        return (
            self.dim * self.moe_inter_dim * 3
        ) * 9

    def get_moe_expert_size(self):
        return (
            self.dim * self.moe_inter_dim * 3
        )

    def get_moe_gate_size(self):
        return (
            self.dim * self.n_routed_experts + self.n_routed_experts
        )

    def get_output_layer_size(self):
        return self.dim * self.vocab_size + self.dim

    def get_model_size(self):
        return self.get_embeding_size() + self.n_dense_layers * (
            self.get_mla_size() + self.get_dense_mlp_size()
        ) + (self.n_layers - self.n_dense_layers) * (
            self.get_mla_size() + self.get_moe_expert_size() * (self.n_shared_experts +
                                                                self.n_routed_experts) + self.get_moe_gate_size()
        ) + self.get_output_layer_size()

    # 单个设备上的模型大小
    def get_device_model_size(self, tp: int, ep: int):
        expert_num = self.n_shared_experts + \
            math.ceil(self.n_routed_experts / ep)
        dense_model_size = self.n_layers * (
            self.get_mla_size()
        ) / tp + self.n_dense_layers * (
            self.get_dense_mlp_size()
        ) / tp
        moe_model_size = (self.n_layers - self.n_dense_layers) * \
            (self.get_moe_expert_size() * expert_num + self.get_moe_gate_size())
        embeding_size = self.get_embeding_size() / tp
        output_layer_size = self.get_output_layer_size()
        return dense_model_size + moe_model_size + embeding_size + output_layer_size

    def get_kv_cache_size(self, input_seq_len: int, output_seq_len: int, kv_cache_hit_rate: float):
        return (input_seq_len * (1 - kv_cache_hit_rate) + output_seq_len) * (self.kv_lora_rank + self.qk_rope_head_dim) * self.n_layers

    def get_max_bs(self,
                   gpu: GPUInfo,
                   tp: int,
                   dp: int,
                   ep: int,
                   input_seq_len: int,
                   output_seq_len: int,
                   kv_cache_hit_rate: float,
                   mem_fraction_static: float):
        dtype_mem_size = 1 if gpu.support_dtype(DType.FP8) else 2
        return math.floor((gpu.get_mem_size() * 1024 * 1024 * 1024 * mem_fraction_static - self.get_device_model_size(tp, ep) * dtype_mem_size) / (self.get_kv_cache_size(input_seq_len, output_seq_len, kv_cache_hit_rate) * dtype_mem_size) * dp)

    def get_throughput(self, gpu: GPUInfo, args: ServerArgs) -> Throughput:
        if args.disaggregation_mode == DisaggregationMode.PREFILL:
            return self.get_prefill_throughput(gpu, args)
        elif args.disaggregation_mode == DisaggregationMode.DECODE:
            return self.get_decode_throughput(gpu, args)
        else:
            raise ValueError(
                f"Invalid disaggregation mode: {args.disaggregation_mode}")


class DenseMLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, dtype=torch.bfloat16)
        self.w2 = nn.Linear(inter_dim, dim, dtype=torch.bfloat16)
        self.w3 = nn.Linear(dim, inter_dim, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
