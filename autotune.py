import pandas as pd
import argparse
from models.deepseek_v3.model import Model
from structs.structs import ServerArgs, get_gpu_info, DisaggregationMode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek_v3")
    parser.add_argument("--nnodes", type=int, default=4)
    parser.add_argument("--input-seq-len", type=int, default=4383)
    parser.add_argument("--output-seq-len", type=int, default=1210)
    parser.add_argument("--kv-cache-hit-rate", type=float, default=0.563)
    parser.add_argument("--dispatch-node", type=int, default=4)
    parser.add_argument("--overlap", action="store_true", default=False)
    parser.add_argument("--gpu-type", type=str, nargs='+', default=["H800"])
    parser.add_argument("--disaggregation-mode", type=str,
                        default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--mem-fraction-static", type=float, default=0.92)
    parser.add_argument("--tpot-threshold", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()

    gpu_info_list = []
    for gpu_type in args.gpu_type:
        gpu_info_list.append(get_gpu_info(
            gpu_type, DisaggregationMode(args.disaggregation_mode)))

    m = Model()

    tp_list = [1, 2, 4, 8]
    throughputs = []
    for gpu in gpu_info_list:
        for tp in tp_list:
            if args.batch_size > 0:
                max_bs = args.batch_size
            else:
                max_bs = m.get_max_bs(gpu=gpu, tp=tp, dp=args.nnodes * 8 // tp, ep=args.nnodes * 8, input_seq_len=args.input_seq_len,
                                      output_seq_len=args.output_seq_len, kv_cache_hit_rate=args.kv_cache_hit_rate, mem_fraction_static=args.mem_fraction_static)
               
            if max_bs <= 0:
                print(
                    f"No valid batch size found for gpu={gpu.name} tp={tp}, dp={args.nnodes * 8 // tp}, ep={args.nnodes * 8}")
                continue

            for bs in range(max_bs, 0, -1):
                server_args = ServerArgs(
                    tp, args.nnodes * 8 // tp, 1, args.nnodes * 8,
                    args.nnodes, bs, args.input_seq_len, args.output_seq_len,
                    args.kv_cache_hit_rate, args.dispatch_node, args.overlap, DisaggregationMode(args.disaggregation_mode), args.mem_fraction_static)

                throughput = m.get_throughput(gpu, server_args)

                if DisaggregationMode(args.disaggregation_mode) == DisaggregationMode.DECODE:
                    if throughput.get_elapse_time() > args.tpot_threshold:
                        print(
                            f"TPOT({throughput.get_elapse_time():.3f}ms) > 50ms, skip gpu={throughput.get_gpu_info().name} tp={tp}, dp={args.nnodes * 8 // tp}, ep={args.nnodes * 8}, batch_size={throughput.detail['BatchSize']}")
                        continue
                throughputs.append(throughput)
                break

    if len(throughputs) == 0:
        print("There is no valid throughputs under the current conditions")
        exit(1)

    # sort by gpu and throughput
    throughputs.sort(key=lambda x: (x.get_gpu_info().name, x.get_throughput()))

    detail_keys = list(throughputs[0].get_detail().keys())
    columns = ["GPU", "TP", "DP", "EP"] + detail_keys + \
        ["ThroughputPerNode(tok/s)"] + ["Throughput(tok/s)"]
    df = pd.DataFrame(columns=columns)
    for throughput in throughputs:
        detail = throughput.get_detail()
        row = [throughput.get_gpu_info().name]
        args = throughput.get_args()
        row.append(args.tp)
        row.append(args.dp)
        row.append(args.ep)
        for key in detail_keys:
            row.append(detail[key])
        row.append(throughput.get_throughput() / args.nnodes)
        row.append(throughput.get_throughput())
        df.loc[len(df)] = row

    df = df.set_index('GPU').T
    print(df.to_markdown(floatfmt='.3f'))
