# read the benchmark result from *.jsonl
# and print the result in a table

import json
import argparse
import pandas as pd
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    # read the benchmark result from *.jsonl using regex
    import os

    # Get all .jsonl files in the current directory
    jsonl_files = [f for f in os.listdir(args.path) if f.endswith('.jsonl')]
    results = []
    for file in jsonl_files:
        with open(f'{args.path}/{file}', 'r') as f:
            # read the last line
            result = json.loads(f.readlines()[-1])
            results.append(result)

    # sort the results by max_concurrency
    results.sort(key=lambda x: x['max_concurrency'])

    # convert the results to a pandas dataframe
    columns = ['BatchSize', 'InputSeqLen', 'OutputSeqLen',
               'InputThroughput(tok/s)', 'OutputThroughput(tok/s)', 'E2ELatency(ms)', 'TTFT(ms)', 'TPOT(ms)']
    df = pd.DataFrame(columns=columns)
    for result in results:
        df.loc[len(df)] = [result['max_concurrency'], result['total_input_tokens'] / result['completed'], result['total_output_tokens'] / result['completed'],
                            result['input_throughput'], result['output_throughput'], result['median_e2e_latency_ms'], result['median_ttft_ms'], result['median_tpot_ms']]
    df = df.set_index('BatchSize').T
    markdown_output = df.to_markdown(floatfmt='.3f')
    print(markdown_output)
