import joblib
import numpy as np
from scipy.stats import spearmanr

from src.config import config as cfg
from src.features.features import add_anon_features
from src.validation.cv import catboost_cv, lightgbm_cv
from src.features.features import feature_extraction



def validation_catboost(n_components_cat=cfg.cat_components_list):
    oof_predictions = {}
    df_train, y_train, anon_cols, rank_cols = feature_extraction()
    print('CatBoost Baseline CV')

    for n_components in n_components_cat:
        pca_pipeline = joblib.load(f'{cfg.OUT_DIR}/pca_pipeline_for_cat_n{n_components}.pkl')
        x_train = add_anon_features(df_train, pca_pipeline, anon_cols, rank_cols)
        feature_cols = x_train.columns.tolist()
        df_tmp = x_train.copy()
        df_tmp['forward_returns'] = y_train.reset_index(drop=True)
        current_oof = catboost_cv(df_tmp, feature_cols)
        oof_predictions[f'oof_n{n_components}'] = current_oof
    
    # 相関係数
    def p_sp(tgt):
        print(f"\n{'-'*5} < 相関（対{tgt}）> {'-'*5}")
        for k, v in oof_predictions.items():
            corr = spearmanr(oof_predictions[f'oof_{tgt}'], v)[0]
            print(k, corr)
    p_sp(tgt='n36')
    
    # アンサンブル（単純平均）
    mean_ensemble_oof = np.mean(list(oof_predictions.values()), axis=0)
    print(f"\n{'-'*5} < アンサンブル（単純平均）結果 > {'-'*5}")
    print(f'Ensemble_OOF_Spearman: {spearmanr(df_tmp["forward_returns"], mean_ensemble_oof)[0]}')

    return oof_predictions


def validation_lightgbm(n_components_lgb=cfg.lgb_components_list):
    oof_predictions = {}
    df_train, y_train, anon_cols, rank_cols = feature_extraction()
    print('LightGBM Baseline CV')

    for n_components in n_components_lgb:
        pca_pipeline = joblib.load(f'{cfg.OUT_DIR}/pca_pipeline_for_lgb_n{n_components}.pkl')
        x_train = add_anon_features(df_train, pca_pipeline, anon_cols, rank_cols)
        feature_cols = x_train.columns.tolist()
        df_tmp = x_train.copy()
        df_tmp['forward_returns'] = y_train.reset_index(drop=True)
        current_oof = lightgbm_cv(df_tmp, feature_cols)
        oof_predictions[f'oof_n{n_components}'] = current_oof
    
    # 相関係数
    def p_sp(tgt):
        print(f"\n{'-'*5} < 相関（対{tgt}）> {'-'*5}")
        for k, v in oof_predictions.items():
            corr = spearmanr(oof_predictions[f'oof_{tgt}'], v)[0]
            print(k, corr)
    p_sp(tgt='n83')
    
    # アンサンブル（単純平均）
    mean_ensemble_oof = np.mean(list(oof_predictions.values()), axis=0)
    print(f"\n{'-'*5} < アンサンブル（単純平均）結果 > {'-'*5}")
    print(f'Ensemble_OOF_Spearman: {spearmanr(df_tmp["forward_returns"], mean_ensemble_oof)[0]}')

    return oof_predictions