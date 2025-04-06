from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
from structs.structs import GPUInfo, ServerArgs, Throughput, DisaggregationMode, disaggregation_mode


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
        q_down_proj = q_len * self.dim * self.q_lora_rank  # wq_a
        q_up_proj = q_len * self.q_lora_rank * self.n_heads * \
            (self.qk_nope_head_dim + self.qk_rope_head_dim)  # wq_b
        kv_down_proj = kv_len * self.dim * \
            (self.kv_lora_rank + self.qk_rope_head_dim)  # wkv_a
        k_up_proj = kv_len * self.kv_lora_rank * \
            self.n_heads * self.qk_nope_head_dim  # w_uk
        v_up_proj = kv_len * self.kv_lora_rank * self.n_heads * self.v_head_dim  # w_uv

        kv_down_proj = kv_down_proj * (1 - kv_cache_hit_rate)
        gemm_sum = q_down_proj + q_up_proj + kv_down_proj + k_up_proj + v_up_proj

        # 把它看成一个标准的args.n_heads的MHA
        mha = self.n_heads * (q_len * self.qk_rope_head_dim * kv_len  # QK_score_rope
                              + q_len * self.qk_nope_head_dim * kv_len  # QK_score_nope
                              + q_len * kv_len * self.v_head_dim)  # ScoreV
        wo = q_len * self.n_heads * self.v_head_dim * self.dim  # wo
        attn_sum = mha + wo

        # return flops by 2* Sum(MACs)
        GEMM_FP8_FLOPS = gemm_sum * 2/1e9
        ATTN_FP16_FLOPS = attn_sum * 2/1e9

        return GEMM_FP8_FLOPS + ATTN_FP16_FLOPS, GEMM_FP8_FLOPS, ATTN_FP16_FLOPS

    def get_mla_mat_absob_flops(self, q_len, kv_len, kv_cache_hit_rate: float):
        q_down_proj = q_len * self.dim * self.q_lora_rank  # wq_a
        q_rope_up_proj = q_len * self.q_lora_rank * \
            self.n_heads * self.qk_rope_head_dim  # wq_b_rope
        q_absorb = q_len * self.n_heads * (self.q_lora_rank * self.qk_nope_head_dim  # wq_b_nope
                                           + self.qk_nope_head_dim * self.kv_lora_rank)  # w_uk

        kv_down_proj = kv_len * self.dim * \
            (self.kv_lora_rank + self.qk_rope_head_dim)  # wkv_a
        kv_down_proj = kv_down_proj * \
            (1 - kv_cache_hit_rate)  # KV-Cache命中率修正
        gemm_sum = q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj

        # 把它看成一个标准的args.n_heads的MQA
        mqa = self.n_heads * (q_len * self.qk_rope_head_dim * kv_len  # Score_rope
                              + q_len * self.kv_lora_rank * kv_len  # Score_nope
                              + q_len * kv_len * self.kv_lora_rank)  # Score V

        attn_up_proj = q_len * self.n_heads * self.v_head_dim * self.kv_lora_rank
        o_proj = q_len * self.n_heads * self.v_head_dim * self.dim
        attn_sum = mqa + attn_up_proj + o_proj

        # return flops by 2* Sum(MACs)
        gemm_sum = gemm_sum * 2/1e9
        attn_sum = attn_sum * 2/1e9

        return gemm_sum + attn_sum, gemm_sum, attn_sum

    def get_moe_expert_flops(self, seq_len):
        return 3 * seq_len * self.dim * self.moe_inter_dim * 2/1e9

    def get_densemlp_flops(self, seq_len):
        return 3 * seq_len * self.dim * self.inter_dim * 2 / 1e9

    def get_prefill_mla_elapse_time(self, gpu: GPUInfo, args: ServerArgs, tp: int):
        _, gemm_fp8_flops, attn_fp16_flops = self.get_mla_flops(
            args.input_seq_len, args.input_seq_len, args.kv_cache_hit_rate)
        gemm_fp8_time = gemm_fp8_flops / gpu.get_fp8_flops()
        attn_fp16_time = attn_fp16_flops / gpu.get_fp16_flops()
        if tp > 1:
            all_reduce_comm_size = args.input_seq_len * \
                self.dim * 2 / 1024/1024  # fp16 take 2Bytes
            ar_elapsed_time = all_reduce_comm_size / gpu.get_nvlink_bw()
            return (gemm_fp8_time + attn_fp16_time)/tp + ar_elapsed_time
        return gemm_fp8_time + attn_fp16_time

    def get_decode_mla_elapse_time(self, gpu: GPUInfo, args: ServerArgs, tp: int, bs: int):
        _, gemm_fp8_flops, attn_fp16_flops = self.get_mla_mat_absob_flops(
            1, args.input_seq_len, 1)
        gemm_fp8_time = gemm_fp8_flops / gpu.get_fp8_flops() * bs
        attn_fp16_time = attn_fp16_flops / gpu.get_fp16_flops() * bs
        if tp > 1:
            all_reduce_comm_size = args.input_seq_len * \
                self.dim * 2 / 1024/1024  # fp16 take 2Bytes
            ar_elapsed_time = all_reduce_comm_size / gpu.get_nvlink_bw()
            return (gemm_fp8_time + attn_fp16_time)/tp + ar_elapsed_time
        return gemm_fp8_time + attn_fp16_time

    def get_dense_mlp_elapse_time(self, gpu: GPUInfo, args: ServerArgs, seq_len: int):
        gemm_fp8_flops = self.get_densemlp_flops(seq_len)
        gemm_fp8_time = gemm_fp8_flops / gpu.get_fp8_flops()
        return gemm_fp8_time

    def get_moe_expert_elapse_time(self, gpu: GPUInfo, args: ServerArgs, seq_len: int):
        if seq_len <= 32:
            group_gemm_discount_rate = 0.2
        elif seq_len <= 64:
            group_gemm_discount_rate = 0.3
        elif seq_len <= 128:
            group_gemm_discount_rate = 0.5
        elif seq_len <= 256:
            group_gemm_discount_rate = 0.7
        elif seq_len <= 512:
            group_gemm_discount_rate = 0.8
        elif seq_len <= 1024:
            group_gemm_discount_rate = 0.9
        else:
            group_gemm_discount_rate = 1

        load_model_time = self.get_moe_expert_size() / 1024/1024/1024 / \
            gpu.get_mem_bw() * 1000
        shared_flops = self.get_moe_expert_flops(seq_len)
        shared_time = shared_flops / gpu.get_fp8_flops() / group_gemm_discount_rate + load_model_time
        routed_flops = self.get_moe_expert_flops(
            seq_len * self.n_activated_experts)
        routed_time = routed_flops / gpu.get_fp8_flops() / group_gemm_discount_rate + load_model_time

        return shared_time, routed_time

    def get_prefill_alltoall_elapse_time(self, gpu: GPUInfo, args: ServerArgs, dispatch_size: int):
        dispatch_size = dispatch_size * self.dim / 1024/1024
        combine_size = 2 * dispatch_size  # fp16
        comm_bw = gpu.get_pcie_bw()
        dispatch_time = dispatch_size / comm_bw
        combine_time = combine_size / comm_bw
        return dispatch_time, combine_time

    def get_prefill_elapse_time(self, gpu: GPUInfo, args: ServerArgs):
        dense_mla = self.get_prefill_mla_elapse_time(gpu, args, 1)
        tp_mla = self.get_prefill_mla_elapse_time(gpu,  args, args.tp)
        dense_mlp = self.get_dense_mlp_elapse_time(
            gpu, args, args.input_seq_len)
        shared, routed = self.get_moe_expert_elapse_time(
            gpu, args,  args.dp * args.input_seq_len / args.world_size)
        dispatch, combine = self.get_prefill_alltoall_elapse_time(gpu, args, (args.dispatch_node - 1) * (8/args.tp) *
                                                                  args.input_seq_len)
        return dense_mla, dense_mlp, tp_mla, shared, routed, dispatch, combine

    def get_decode_elapse_time(self, gpu: GPUInfo, args: ServerArgs, batch_size: int):
        dense_mla = self.get_decode_mla_elapse_time(
            gpu, args, 1, batch_size)
        tp_mla = self.get_decode_mla_elapse_time(
            gpu, args, args.tp, batch_size)
        kv_cache_load_time = batch_size * (args.input_seq_len) * (
            self.kv_lora_rank + self.qk_rope_head_dim) / 1024/1024/1024 / gpu.get_mem_bw() * 1000
        dense_mlp = self.get_dense_mlp_elapse_time(gpu, args, batch_size)
        shared, routed = self.get_moe_expert_elapse_time(
            gpu, args, batch_size)
        dispatch, combine = self.get_prefill_alltoall_elapse_time(
            gpu, args, batch_size * self.n_activated_experts)
        return dense_mla, dense_mlp, tp_mla, kv_cache_load_time, shared, routed, dispatch, combine

    def get_prefill_elapse_time_sum(self, gpu: GPUInfo, args: ServerArgs):
        dense_mla, dense_mlp, tp_mla, shared, routed, dispatch, combine = self.get_prefill_elapse_time(
            gpu, args)
        if args.overlap:
            return self.n_dense_layers * (dense_mla + dense_mlp) + (self.n_layers - self.n_dense_layers) * (tp_mla + shared + routed)
        else:
            return self.n_dense_layers * (dense_mla + dense_mlp) + (self.n_layers - self.n_dense_layers) * (tp_mla + shared + routed + dispatch + combine)

    def get_decode_elapse_time_sum(self, gpu: GPUInfo, args: ServerArgs, batch_size: int):
        dense_mla, dense_mlp, tp_mla, kv_cache_load_time, shared, routed, dispatch, combine = self.get_decode_elapse_time(
            gpu, args, batch_size)
        if args.overlap:
            return self.n_dense_layers * (dense_mla + kv_cache_load_time + dense_mlp) + (self.n_layers - self.n_dense_layers) * (tp_mla + kv_cache_load_time+shared + routed)
        else:
            return self.n_dense_layers * (dense_mla + kv_cache_load_time + dense_mlp) + (self.n_layers - self.n_dense_layers) * (tp_mla + kv_cache_load_time + shared + routed + dispatch + combine)

    def get_prefill_throughput(self, gpu_info_list: list[GPUInfo], args: ServerArgs) -> list[Throughput]:
        throughputs = []
        for gpu_info in gpu_info_list:
            dense_mla, dense_mlp, tp_mla, shared, routed, dispatch, combine = self.get_prefill_elapse_time(
                gpu_info, args)
            elapse_time = self.get_prefill_elapse_time_sum(
                gpu_info, args)
            throughputs.append(Throughput(
                gpu_info,
                args,
                args.dp * args.input_seq_len * 1000 / elapse_time / args.get_nnodes(),
                elapse_time,
                {
                    "MaxBatchSize": self.get_max_bs(gpu=gpu_info, args=args),
                    "DenseMLA(ms)": dense_mla,
                    "DenseMLP(ms)": dense_mlp,
                    "TP_MLA(ms)": tp_mla,
                    "Shared Expert(ms)": shared,
                    "Routed Expert(ms)": routed,
                    "Dispatch(ms)": dispatch,
                    "Combine(ms)": combine,
                    "ElapseTime(ms)": elapse_time
                }
            ))
        return throughputs

    def get_decode_throughput(self, gpu_info_list: list[GPUInfo], args: ServerArgs) -> list[Throughput]:
        throughputs = []
        for gpu_info in gpu_info_list:
            device_max_bs = self.get_max_bs(gpu=gpu_info, args=args)
            batch_size = args.batch_size if args.batch_size <= device_max_bs else device_max_bs
            dense_mla, dense_mlp, tp_mla, kv_cache_load_time, shared, routed, dispatch, combine = self.get_decode_elapse_time(
                gpu_info, args, batch_size)
            elapse_time = self.get_decode_elapse_time_sum(gpu_info, args, batch_size)
            throughputs.append(Throughput(
                gpu_info,
                args,
                1000 / elapse_time * batch_size,
                elapse_time,
                {
                    "MaxBatchSize": device_max_bs,
                    "DenseMLA(ms)": dense_mla,
                    "DenseMLP(ms)": dense_mlp,
                    "TP_MLA(ms)": tp_mla,
                    "KV Cache Load(ms)": kv_cache_load_time,
                    "Shared Expert(ms)": shared,
                    "Routed Expert(ms)": routed,
                    "Dispatch(ms)": dispatch,
                    "Combine(ms)": combine,
                    "ElapseTime(TPOT, ms)": elapse_time,
                }
            ))
        return throughputs

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
    def get_device_model_size(self, args: ServerArgs):
        expert_num = self.n_shared_experts + self.n_routed_experts / args.ep
        dense_model_size = self.n_layers * (
            self.get_mla_size()
        ) / args.tp + self.n_dense_layers * (
            self.get_dense_mlp_size()
        ) / args.tp
        moe_model_size = (self.n_layers - self.n_dense_layers) * \
            (self.get_moe_expert_size() * expert_num + self.get_moe_gate_size())
        embeding_size = self.get_embeding_size() / args.tp
        output_layer_size = self.get_output_layer_size()
        return dense_model_size + moe_model_size + embeding_size + output_layer_size

    def get_max_bs(self, gpu: GPUInfo, args: ServerArgs):
        with disaggregation_mode(args):
            # print(gpu.name,gpu.get_fp8_mem_size())
            kv_cache_size = (args.input_seq_len * (1 - args.kv_cache_hit_rate) +
                             args.output_seq_len) * (self.kv_lora_rank + self.qk_rope_head_dim) * self.n_layers * gpu.get_fp8_mem_size()
            device_model_size = self.get_device_model_size(args) * gpu.get_fp8_mem_size()
            return math.floor((gpu.get_mem_size() * 1024 * 1024 * 1024 * args.mem_fraction_static - device_model_size) / kv_cache_size / args.tp)

    def get_throughput(self, gpu_info_list: list[GPUInfo], args: ServerArgs) -> list[Throughput]:
        if args.disaggregation_mode == DisaggregationMode.PREFILL:
            return self.get_prefill_throughput(gpu_info_list, args)
        elif args.disaggregation_mode == DisaggregationMode.DECODE:
            return self.get_decode_throughput(gpu_info_list, args)
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
