from models.deepseek_v3.model import Model
from models.deepseek_v3.model import DenseMLP
from ptflops import get_model_complexity_info
from structs.structs import ServerArgs, get_gpu_info, DisaggregationMode, H800_DECODE
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek_v3")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--dp", type=int, default=8)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=32)
    parser.add_argument("--nnodes", type=int, default=0)
    parser.add_argument("--batch-size-per-device", type=int, default=1)
    parser.add_argument("--input-seq-len", type=int, default=4383)
    parser.add_argument("--output-seq-len", type=int, default=1210)
    parser.add_argument("--kv-cache-hit-rate", type=float, default=0.563)
    parser.add_argument("--dispatch-node", type=int, default=4)
    parser.add_argument("--overlap", action="store_true", default=True)
    parser.add_argument("--gpu-type", type=str, nargs='+', default=["H800"])
    parser.add_argument("--disaggregation-mode", type=str,
                        default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--mem-fraction-static", type=float, default=0.9)

    args = parser.parse_args()
    server_args = ServerArgs(args)

    gpu_info_list = []
    for gpu_type in args.gpu_type:
        gpu_info_list.append(get_gpu_info(
            gpu_type, server_args.disaggregation_mode))

    m = Model()

    # m.print_prefill_time_sum(gpu_info_list, server_args)
    throughputs = m.get_throughput(gpu_info_list, server_args)
    detail_keys = list(throughputs[0].get_detail().keys())
    columns = ["GPU"] + detail_keys + ["Throughput(tok/s)"]
    df = pd.DataFrame(columns=columns)
    for throughput in throughputs:
        detail = throughput.get_detail()
        row = [throughput.get_gpu_info().name]
        # 按照detail_keys的顺序添加值，确保与列名匹配
        for key in detail_keys:
            row.append(detail[key])
        row.append(throughput.get_throughput())
        df.loc[len(df)] = row
    df = df.set_index('GPU').T
    env = f"# Environment:\n\
#  NNodes: {server_args.nnodes} (gpus: 8*{server_args.nnodes}={server_args.world_size})\n\
#  ISL/OSL: {server_args.input_seq_len}/{server_args.output_seq_len}\n\
#  TP/DP/PP: {server_args.tp}/{server_args.dp}/{server_args.pp}\n\
#  EP: {server_args.ep}\n\
#  KVCacheHitRate: {server_args.kv_cache_hit_rate}"
    result = f"# { 'Prefill' if server_args.disaggregation_mode == DisaggregationMode.PREFILL else 'Decode'} { '(Overlap)' if server_args.overlap else '' }) 吞吐/单卡: "
    markdown_output = f"{env}\n{result}\n{df.to_markdown(floatfmt='.3f')}"
    print(markdown_output)
