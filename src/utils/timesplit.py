import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=5, purge=0):
        self.n_splits = n_splits
        self.purge    = purge

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        for i in range(1, self.n_splits + 1):
            # test index
            test_start = i * test_size
            test_end   = min(test_start + test_size, n_samples)
            
            # train index (purge対応)
            train_end = max(0, test_start - self.purge)

            train_idx = np.arange(0, train_end)
            test_idx  = np.arange(test_start, test_end)

            yield train_idx, test_idx