import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from scipy.stats import spearmanr

from src.config.config import cat_params, lgb_params, n_splits, purge
from src.utils.timesplit import PurgedTimeSeriesSplit


def catboost_cv(df_train, features, target='forward_returns', params=cat_params, n_splits=n_splits, purge=purge):
    cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge=purge)
    oof = np.zeros(len(df_train))
    scores = []
    for fold, (idx_tr, idx_va) in enumerate(cv.split(df_train)):
        x_tr, x_va = df_train.iloc[idx_tr][features], df_train.iloc[idx_va][features]
        y_tr, y_va = df_train.iloc[idx_tr][target], df_train.iloc[idx_va][target]
        model = CatBoostRegressor(**params, early_stopping_rounds=100)
        model.fit(x_tr, y_tr, eval_set=(x_va, y_va), use_best_model=True)
        pred = model.predict(x_va)
        oof[idx_va] = pred
        score = spearmanr(y_va, pred)[0]
        scores.append(score)
        print(f'\nFold {fold} | Spearman: {score:.4f} | best_iter: {model.get_best_iteration()}')
    oof_score = spearmanr(df_train[target], oof)[0]
    print(f'Mean Spearman: {np.mean(scores):.5f}')
    print(f'OOF Spearman: {oof_score:.5f}')
    return oof



def lightgbm_cv(df_train, features, target='forward_returns', params=lgb_params, n_splits=n_splits, purge=purge):
    cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge=purge)
    oof = np.zeros(len(df_train))
    scores = []
    print('LightGBM Baseline CV')
    for fold, (idx_tr, idx_va) in enumerate(cv.split(df_train)):
        x_tr, x_va = df_train.iloc[idx_tr][features], df_train.iloc[idx_va][features]
        y_tr, y_va = df_train.iloc[idx_tr][target], df_train.iloc[idx_va][target]
        model = LGBMRegressor(**params)
        model.fit(x_tr, y_tr, eval_set=(x_va, y_va), callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        pred = model.predict(x_va, num_iteration=model.best_iteration_)
        oof[idx_va] = pred
        score = spearmanr(y_va, pred)[0]
        scores.append(score)
        print(f'\nFold {fold} | Spearman: {score:.4f} | best_iter: {model.best_iteration_}')
    oof_score = spearmanr(df_train[target], oof)[0]
    print(f'Mean Spearman: {np.mean(scores):.5f}')
    print(f'OOF Spearman: {oof_score:.5f}')
    return oof