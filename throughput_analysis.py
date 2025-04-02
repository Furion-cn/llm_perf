from models.deepseek_v3.model import MLA, ModelArgs
from ptflops import get_model_complexity_info

def get_mla_flops():
    args = ModelArgs()
    m = MLA(args)
    num_tokens = 1

    mla_flops, mla_params = get_model_complexity_info(m, (num_tokens,args.dim),as_strings=True,print_per_layer_stat=True)
    return mla_flops, mla_params

if __name__ == "__main__":
    mla_flops, mla_params = get_mla_flops()
    print(mla_flops, mla_params)