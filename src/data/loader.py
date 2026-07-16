import os
import pandas as pd
from src.config import config as cfg


def load_data():
    if os.path.exists(f'{cfg.COMPETITION_DIR}/train.csv'):
        DATA_PATH_TRAIN = f'{cfg.COMPETITION_DIR}/train.csv'
        DATA_PATH_TEST  = f'{cfg.COMPETITION_DIR}/test.csv'
    else:
        DATA_PATH_TRAIN = f'{cfg.INPUT_DIR}/train.csv'
        DATA_PATH_TEST  = f'{cfg.INPUT_DIR}/test.csv'
    train = pd.read_csv(DATA_PATH_TRAIN)
    test  = pd.read_csv(DATA_PATH_TEST)
    return train, test