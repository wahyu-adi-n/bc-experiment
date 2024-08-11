# normalization.py
N_CHANNELS = 3

# datasets.py
CSV_PATH = './artifact/mysplit.csv'
DATASET_ROOT = './dataset/BreaKHis_v1/'

# mysplit.py
FOLD_CSV_DIR = './artifact/mysplit.csv'
FOLD_STAT_DIR = './artifact/tumor_table.csv'

# main.py
SEED = 123
STEP_SIZE = 1
GAMMA = 0.8

# activation.py
SAVE_AFS_PLOTTING = './artifact/afs_plot'

# train_eval.py
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")