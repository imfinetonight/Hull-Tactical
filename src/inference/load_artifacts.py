import os
import glob
import re
import polars as pl
import joblib

from src.config import config as cfg


def _get_number(filepath):
    match = re.search(r'_n(\d+)\.pkl$', filepath)
    return int(match.group(1)) if match else 0


def load_models(flag):
    if flag == 'cat':
        # CatBoost
        files = sorted(glob.glob(f'{cfg.OUT_DIR}/cat_pipeline_n*.pkl'), key=_get_number)
        pipeline_bundle_list = [joblib.load(f) for f in files]
        pipeline_list = [b['cat_pipeline'] for b in pipeline_bundle_list]
        feature_cols_list = [b['cat_feature_cols'] for b in pipeline_bundle_list]
        pca_files = sorted(glob.glob(f'{cfg.OUT_DIR}/pca_pipeline_for_cat_n*.pkl'), key=_get_number)
        pca_pipeline_list = [joblib.load(f) for f in pca_files]
    elif flag == 'lgb':
        # LightGBM
        files = sorted(glob.glob(f'{cfg.OUT_DIR}/lgb_pipeline_n*.pkl'), key=_get_number)
        pipeline_bundle_list = [joblib.load(f) for f in files]
        pipeline_list = [b['lgb_pipeline'] for b in pipeline_bundle_list]
        feature_cols_list = [b['lgb_feature_cols'] for b in pipeline_bundle_list]
        pca_files = sorted(glob.glob(f'{cfg.OUT_DIR}/pca_pipeline_for_lgb_n*.pkl'), key=_get_number)
        pca_pipeline_list = [joblib.load(f) for f in pca_files]
    return list(zip(pipeline_list, feature_cols_list, pca_pipeline_list))


def load_macro_pipeline():
    macro_pca_pipeline = joblib.load(f'{cfg.OUT_DIR}/macro_pca_pipeline.pkl')
    return macro_pca_pipeline


def load_kmeans_pipeline():
    kmeans_pipeline = joblib.load(f'{cfg.OUT_DIR}/kmeans_pipeline.pkl')
    return kmeans_pipeline


def load_col_list():
    col_list_bundle = joblib.load(f'{cfg.OUT_DIR}/col_list.pkl')
    anon_cols = col_list_bundle['anon_cols']
    rank_cols = col_list_bundle['rank_cols']
    return anon_cols, rank_cols


def load_ridge_stack():
    ridge_stack_bundle = joblib.load(f'{cfg.OUT_DIR}/ridge_stack.pkl')
    stack_scaler = ridge_stack_bundle['stack_scaler']
    ridge_for_weights = ridge_stack_bundle['ridge_for_weights']
    return stack_scaler, ridge_for_weights


def load_parquet():
    tail = pl.read_parquet(f'{cfg.OUT_DIR}/tail.parquet')
    preds_sim = pl.read_parquet(f'{cfg.OUT_DIR}/preds_sim.parquet')
    return tail, preds_sim
