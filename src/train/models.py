import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from src.features.features import add_anon_features, feature_extraction
from src.validation.cv import catboost_cv, lightgbm_cv
from src.config import config as cfg


def train_catboost(n_components_list=cfg.cat_components_list):
    cat_pipeline = Pipeline([
       ('model', CatBoostRegressor(**cfg.cat_params))
    ])
    df_train, y_train, anon_cols, rank_cols = feature_extraction()
    for n_components in n_components_list:
        pca_pipeline = joblib.load(f'{cfg.OUT_DIR}/pca_pipeline_for_cat_n{n_components}.pkl')
        x_train = add_anon_features(df_train, pca_pipeline, anon_cols, rank_cols)
        feature_cols = x_train.columns.tolist()
        cat_pipeline.fit(x_train, y_train)
        joblib.dump({'cat_pipeline': cat_pipeline, 'cat_feature_cols': feature_cols}, f'{cfg.OUT_DIR}/cat_pipeline_n{n_components}.pkl')
        print(f"✅saved 'cat_pipeline_n{n_components}.pkl'")


def train_lightgbm(n_components_list=cfg.lgb_components_list):
    lgb_pipeline = Pipeline([
        ('model', LGBMRegressor(**cfg.lgb_params))
    ])
    df_train, y_train, anon_cols, rank_cols = feature_extraction()
    for n_components in n_components_list:
        pca_pipeline = joblib.load(f'{cfg.OUT_DIR}/pca_pipeline_for_lgb_n{n_components}.pkl')
        x_train = add_anon_features(df_train, pca_pipeline, anon_cols, rank_cols)
        feature_cols = x_train.columns.tolist()
        lgb_pipeline.fit(x_train, y_train)
        joblib.dump({'lgb_pipeline': lgb_pipeline, 'lgb_feature_cols': feature_cols}, f'{cfg.OUT_DIR}/lgb_pipeline_n{n_components}.pkl')
        print(f"✅saved 'lgb_pipeline_n{n_components}.pkl'")


def ridge_stacking(n_components_cat=cfg.cat_components_list, n_components_lgb=cfg.lgb_components_list):
    oof_predictions = {}; keys = []
    df_train, y_train, anon_cols, rank_cols = feature_extraction()

    for n_components in n_components_cat:
        pca_pipeline = joblib.load(f'{cfg.OUT_DIR}/pca_pipeline_for_cat_n{n_components}.pkl')
        x_train = add_anon_features(df_train, pca_pipeline, anon_cols, rank_cols)
        feature_cols = x_train.columns.tolist()
        df_tmp = x_train.copy()
        df_tmp['forward_returns'] = y_train.reset_index(drop=True)
        current_oof = catboost_cv(df_tmp, feature_cols)
        key = f'oof_cat{n_components}'
        oof_predictions[key] = current_oof
        keys.append(key)
    for n_components in n_components_lgb:
        pca_pipeline = joblib.load(f'{cfg.OUT_DIR}/pca_pipeline_for_lgb_n{n_components}.pkl')
        x_train = add_anon_features(df_train, pca_pipeline, anon_cols, rank_cols)
        feature_cols = x_train.columns.tolist()
        df_tmp = x_train.copy()
        df_tmp['forward_returns'] = y_train.reset_index(drop=True)
        current_oof = lightgbm_cv(df_tmp, feature_cols)
        key = f'oof_lgb{n_components}'
        oof_predictions[key] = current_oof
        keys.append(key)

    X = np.column_stack([oof_predictions[k] for k in keys])
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X)
    y = np.asarray(y_train).ravel()
    y_rank = np.sign(y) * (rankdata(np.abs(y)) ** cfg.ridge_config['beta'])
    ridge = Ridge(alpha=cfg.ridge_config['alpha'], fit_intercept=True)
    ridge.fit(X_std, y_rank)
    joblib.dump({'stack_scaler': scaler, 'ridge_for_weights': ridge}, f'{cfg.OUT_DIR}/ridge_stack.pkl')
    print("✅saved 'ridge_stack.pkl'")



if __name__ == "__main__":
    ridge_stacking()