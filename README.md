## Usage

The main interface is through `throughput_analysis.py`, which accepts various parameters to customize the analysis:

```bash
# Basic usage with default parameters
python throughput_analysis.py

# Custom configuration example
python throughput_analysis.py --tp 4 --dp 8 --ep 32 --gpu-type H800 H20 --disaggregation-mode prefill --overlap
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model type to analyze | `deepseek_v3` |
| `--tp` | Tensor Parallelism degree | `8` |
| `--dp` | Data Parallelism degree | `1` |
| `--pp` | Pipeline Parallelism degree | `1` |
| `--ep` | Expert Parallelism degree | `1` |
| `--input-seq-len` | Input sequence length | `4383` |
| `--output-seq-len` | Output sequence length | `1210` |
| `--batch-size-per-device` | Batch size per device | `1` |
| `--kv-cache-hit-rate` | KV cache hit rate | `0.563` |
| `--gpu-type` | GPU types to analyze (space-separated) | `H800` |
| `--disaggregation-mode` | Mode of analysis: prefill or decode | `prefill` |
| `--overlap` | Enable computation-communication overlap | `False` |
| `--dispatch-node` | Number of dispatch nodes for MoE | `4` |
| `--mem-fraction-static` | Static memory fraction | `0.9` |

## Example Output

Running the tool produces a detailed analysis of model performance, including:

```
# Environment:
#  NNodes: 1 (gpus: 8*1=8)
#  ISL/OSL: 4383/1210
#  TP/DP/PP: 8/1/1
#  EP: 1
#  KVCacheHitRate: 0.563

# Prefill 吞吐/单卡: 

|                 | H800     |
|:----------------|:---------|
| DenseMLA        | 0.123    |
| DenseMLP        | 0.456    |
| TP_MLA          | 0.789    |
| Shared Expert   | 0.012    |
| Routed Expert   | 0.345    |
| Dispatch        | 0.678    |
| Combine         | 0.901    |
| Throughput(tok/s) | 1234.567 |
```

## Advanced Usage

### Comparing Multiple GPU Types

```bash
python throughput_analysis.py --gpu-type H800 H20 --tp 4 --dp 8
```

### Analyzing Decode Mode

```bash
python throughput_analysis.py --disaggregation-mode decode --output-seq-len 512
```

### Evaluating With Communication-Computation Overlap

```bash
python throughput_analysis.py --overlap
```