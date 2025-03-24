from RawSRv1 import para
import os
import numpy as np
import torch
from tqdm import tqdm
from RawSRv1.train import Predictor
import argparse
import json


def load_rawf(rawf):
    raw = np.load(rawf)
    raw_img = raw["raw"]
    raw_max = raw["max_val"]
    return (raw_img / raw_max).astype(np.float32), raw_max


def upsample_raw(raw):
    raw = torch.from_numpy(raw).float().permute(2, 0, 1).unsqueeze(0)
    upsampled_raw = torch.nn.functional.interpolate(raw, scale_factor=2, mode='nearest')
    upsampled_raw = upsampled_raw.squeeze(0).permute(1, 2, 0).numpy()
    return upsampled_raw


def parse_args():
    parser = argparse.ArgumentParser(description="Please use a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
        raw_list = os.listdir(para.VAL_IN_PATH)
        os.makedirs(para.UPSAMPLED_RAW_DIR)
        predictor = Predictor(config)
        for f in tqdm(raw_list):
            raw_img, raw_max = load_rawf(os.path.join(para.VAL_IN_PATH, f))
            # up_raw = upsample_raw(raw_img, raw_max)
            up_raw = predictor.predict(raw_img)
            np.savez(os.path.join(para.UPSAMPLED_RAW_DIR, f), raw=(up_raw * raw_max).astype(np.uint16), max_val=raw_max)
