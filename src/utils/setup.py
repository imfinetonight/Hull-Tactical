import numpy as np
import pandas as pd
import warnings
import os
import sys
import random

from src.config import config as cfg


def setup_environment():
    
    os.environ['PYTHONHASHSEED'] = str(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

    os.makedirs(f'{cfg.OUT_DIR}', exist_ok=True)

    sys.path.append(os.getcwd())

    