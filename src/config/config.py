# ─── Base ───────────────────────────────
SEED = 42
INPUT_DIR = 'data'
OUT_DIR = 'outputs'
COMPETITION_DIR = '/kaggle/input/hull-tactical-market-prediction'


# ─── PCA ───────────────────────────────
cat_components_list = [36,30,31,32]
lgb_components_list = [83]
keys = [f'oof_n{n}' for n in cat_components_list] + [f'oof_lgb{n}' for n in lgb_components_list]


# ─── KMeans ───────────────────────────────
n_clusters = 6
n_init = 20


# ─── CrossValidation ────────────────────────
n_splits = 5
purge = 5


# ─── CatBoost ───────────────────────────────
cat_params = {
    'loss_function': 'RMSE',
    'depth': 5,
    'learning_rate': 0.02,
    'l2_leaf_reg': 7,
    'random_strength': 0.8,
    'bagging_temperature': 0.8,
    'bootstrap_type': 'Bayesian',
    'iterations': 300,
    'random_seed': 42,
    'verbose': False
}


# ─── LightGBM ───────────────────────────────
lgb_params = {
    'objective': 'regression',
    'metric': 'RMSE',
    'num_leaves': 31,
    'learning_rate': 0.02,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'max_depth': -1,
    'n_estimators': 300,
    'random_state': 42,
    'verbose': -1
}


# ─── Ridge Stacking ───────────────────────────────
ridge_config = {
    'alpha': 12.5,
    'beta' : 0.55,
}


# ─── Position Size Prameters ──────────────────────────────────
# cluster scale
CLUSTER_SCALE = {
    0: 1.0,
    1: 0.8,
    2: 1.25,
    3: 0.6,
    4: 1.15,
    5: 0.9
}

# vol cap
VOL_CAP = {
    0: 1.8,
    1: 1.4,
    2: 1.0
}

BUFFER = 336