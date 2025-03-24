import os
from datetime import datetime

SCALE = 2
LR_SIZE = 512

VAL_R = 0.01

TIME = datetime.now().strftime("%m%d-%H%M%S")

DATASET_DIR = '/'

RAW_DIR = os.path.join(DATASET_DIR, 'train_raws')
DOWNSAMPLED_RAW_DIR = os.path.join(DATASET_DIR, 'n25_rawsr', 'dw' + TIME)
TRAIN_DATASET_DIR = os.path.join(DOWNSAMPLED_RAW_DIR, 'train')
VALIDATE_DATASET_DIR = os.path.join(DOWNSAMPLED_RAW_DIR, 'validate')
DEGRADE_KERNELS_PATH = os.path.join(DATASET_DIR, 'n25_rawsr', 'kernels.npy')

VAL_IN_PATH = os.path.join(DATASET_DIR, 'n25_rawsr', 'val_in')
VAL_PRED_PATH = os.path.join(DATASET_DIR, 'n25_rawsr', 'val_pred')
UPSAMPLED_RAW_DIR = os.path.join(DATASET_DIR, 'n25_rawsr', 'up' + TIME)
