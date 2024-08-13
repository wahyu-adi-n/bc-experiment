# normalization.py
N_CHANNELS = 3

# datasets.py
CSV_PATH = './artifact/mysplit.csv'
DATASET_ROOT = './dataset/BreaKHis_v1/'
BINARY_LABEL_DICT  = {'B':0, 'M':1}
SUBTYPE_LABEL_DICT = {'A':0, 'F':1, 'PT':2, 'TA':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}
MAGNIFICATION_DICT = {'40':0, '100':1, '200':2, '400':3}
LABEL_DICTS = {
                'binary':BINARY_LABEL_DICT,
                'subtype':SUBTYPE_LABEL_DICT,
                'magnification':MAGNIFICATION_DICT
              }
INV_LABEL_DICTS = {
                    'binary': {v: k for k, v in BINARY_LABEL_DICT.items()},
                    'subtype':{v: k for k, v in SUBTYPE_LABEL_DICT.items()},
                    'magnification':{v: k for k, v in MAGNIFICATION_DICT.items()}
                  }

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