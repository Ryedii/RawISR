import os
import para
import tool
import numpy as np
import random
from tqdm import tqdm
from degrade.degradations import simple_deg_simulation as degrade


if __name__ == '__main__':
    raw_list = os.listdir(para.RAW_DIR)
    raw_list = random.sample(raw_list, len(raw_list))

    kernels = np.load(para.DEGRADE_KERNELS_PATH, allow_pickle=True)

    val_freq = 1 // para.VAL_R
    data_cnt = 0

    os.makedirs(os.path.join(para.TRAIN_DATASET_DIR, 'hr'))
    os.makedirs(os.path.join(para.TRAIN_DATASET_DIR, 'lr'))
    os.makedirs(os.path.join(para.VALIDATE_DATASET_DIR, 'hr'))
    os.makedirs(os.path.join(para.VALIDATE_DATASET_DIR, 'lr'))

    raw_pbar = tqdm(raw_list)
    for f in raw_pbar:
        hr_raw, raw_max = tool.load_rawf(os.path.join(para.RAW_DIR, f))
        h, w, _ = hr_raw.shape
        hc = h // (h // 1024)
        wc = w // (w // 1024)
        for i in range(h // 1024):
            for j in range(w // 1024):
                # [hc * i, hc * (i + 1)), [wc * j, wc * (j + 1))
                x = hc * i + np.random.randint(hc - 1024)
                y = wc * j + np.random.randint(wc - 1024)
                crop_hr_raw = hr_raw[x:x + 1024, y:y + 1024, :]
                crop_lr_raw = degrade(crop_hr_raw, kernels)

                data_cnt += 1
                if data_cnt % val_freq == 0:
                    np.save(str(os.path.join(para.VALIDATE_DATASET_DIR, 'hr', str(data_cnt) + '.npy')), crop_hr_raw)
                    np.save(str(os.path.join(para.VALIDATE_DATASET_DIR, 'lr', str(data_cnt) + '.npy')), crop_lr_raw)
                else:
                    np.save(str(os.path.join(para.TRAIN_DATASET_DIR, 'hr', str(data_cnt) + '.npy')), crop_hr_raw)
                    np.save(str(os.path.join(para.TRAIN_DATASET_DIR, 'lr', str(data_cnt) + '.npy')), crop_lr_raw)
        raw_pbar.set_description_str(f'Processing {f}')
