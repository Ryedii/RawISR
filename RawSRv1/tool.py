import numpy as np
import torch
import argparse
from RawSRv1.model import Model
import json


def load_rawf(rawf):
    raw = np.load(rawf)
    raw_img = raw["raw"]
    raw_max = raw["max_val"]
    return (raw_img / raw_max).astype(np.float32), raw_max


def save_rawf(rawf, raw_img, raw_max):
    raw_img = (raw_img * raw_max).astype(np.uint16)
    np.savez(rawf, raw=raw_img, max_val=raw_max)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file."
    )
    return parser.parse_args()


def count_para():
    args = parse_args()
    with (open(args.config, 'r') as f):
        config = json.load(f)
        model = Model(config).to(torch.device(config['device']))
        # model = RawUpsampleModel(config)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':
    count_para()

    # raw_list = os.listdir(para.VAL_PRED_PATH)
    # for f in raw_list:
    #     print(f)
    #     x = np.load(os.path.join(para.VAL_PRED_PATH, f))
    #     print(x)
    #     print(type(x))
    #     print(x['max_val'])

    # raw_list = os.listdir(para.VAL_IN_PATH)
    # for f in raw_list:
    #     x = np.load(os.path.join(para.VAL_IN_PATH, f))
    #     print(x['max_val'])
