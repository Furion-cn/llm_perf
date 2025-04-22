import argparse
import pandas as pd
import os

TOTAL_EXPERTS = 256
TOTAL_LAYERS = 61

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--from-device", type=int)
    parser.add_argument("--to-device", type=int)
    args = parser.parse_args()

    if args.from_device is None and args.to_device is None:
        print("Please specify the from and to device id")
        exit(1)

    if args.from_device is not None and args.to_device is not None:
        print(f"Only one of from_device and to_device should be specified")
        exit(1)

    expert_num_per_device = TOTAL_EXPERTS // args.ep_size
    expert_device_map = {expert_id: expert_id //
                         expert_num_per_device for expert_id in range(TOTAL_EXPERTS)}
    tokens_distributed = [
        [[0 for _ in range(TOTAL_LAYERS)] for _ in range(args.ep_size)] for _ in range(args.ep_size)
    ]
    csv_files = [f for f in os.listdir(args.path) if f.endswith('.csv')]
    for csv_file in csv_files:
        current_device_id = int(csv_file.split('rank')[1].split('_')[0])
        df = pd.read_csv(os.path.join(args.path, csv_file))
        for index, row in df.iterrows():
            expert_id = row['expert_id']
            layer_id = row['layer_id']
            if layer_id == 'UNKNOWN' or isinstance(layer_id, str):
                continue
            device_id = expert_device_map[expert_id]
            tokens_distributed[current_device_id][device_id][layer_id] += row['count']
    # print the result
    columns = ['DEVICE']+['Layer' + str(i) for i in range(TOTAL_LAYERS)]
    df = pd.DataFrame(columns=columns)
    for device_id in range(args.ep_size):
        if args.from_device is not None:
            df.loc[len(df)] = [device_id] + tokens_distributed[args.from_device][device_id]
        else:
            df.loc[len(df)] = [device_id] + tokens_distributed[device_id][args.to_device]
    df = df.set_index('DEVICE').T
    print(df.to_markdown(floatfmt='.3f'))
