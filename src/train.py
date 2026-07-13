# ─── Libraries ───
import numpy as np
import pandas as pd
import polars as pl
import joblib

import os
os.makedirs('outputs', exist_ok=True)

import random
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import mean_squared_error


# ─── Evaluation functions ───
# Spearman
def spearman_corr(forward_returns, preds):
    corr, _ = spearmanr(forward_returns, preds)
    return corr if not np.isnan(corr) else 0.0

# Direction Accuracy
def direction_accuracy(forward_returns, preds):
    sign_true = np.sign(forward_returns)
    sign_pred = np.sign(preds)
    return (sign_true == sign_pred).mean()

# RMSE
def rmse(forward_returns, preds):
    return np.sqrt(mean_squared_error(forward_returns, preds))

# model evaluation
def evaluate_model(forward_returns, preds):
    return {
        'spearman_corr': spearman_corr(forward_returns, preds),
        'direction_accuracy': direction_accuracy(forward_returns, preds),
        'rmse': rmse(forward_returns, preds),
    }

# Competition evaluation
def calc_sharpe_ratio(df, submission):
    solution = df[['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']].reset_index(drop=True)
    solution['position'] = submission
    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()
    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)
    if market_volatility == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol
    return_gap = max(0,(market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,)
    return_penalty = 1 + (return_gap**2) / 100
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    print(f'Sharp Ratio：{min(float(adjusted_sharpe), 1_000_000)}')


# ─── Purged TimeSeriesSplit Class ───
class PurgedTimeSeriesSplit(BaseCrossValidator):
    
    def __init__(self, n_splits=5, purge=0):
        self.n_splits = n_splits
        self.purge = purge

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):

            # ----------------------
            # test index
            # ----------------------
            test_start = i * test_size
            test_end = min(test_start + test_size, n_samples)

            # ----------------------
            # train index (purge対応)
            # ----------------------
            train_end = max(0, test_start - self.purge)

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx


# ─── Load Data───
if os.path.exists('/kaggle/input/hull-tactical-market-prediction/train.csv'):
    DATA_PATH_TRAIN = '/kaggle/input/hull-tactical-market-prediction/train.csv'
    DATA_PATH_TEST = '/kaggle/input/hull-tactical-market-prediction/test.csv'
else:
    DATA_PATH_TRAIN = 'data/train.csv'
    DATA_PATH_TEST = 'data/test.csv'

data = pd.read_csv(DATA_PATH_TRAIN)
sample = pd.read_csv(DATA_PATH_TEST)

# ─── Train To Test ───
ttt = data.copy()
ttt['lagged_forward_returns'] = ttt['forward_returns'].shift(1)
ttt['lagged_risk_free_rate'] = ttt['risk_free_rate'].shift(1)
ttt['lagged_market_forward_excess_returns'] = ttt['market_forward_excess_returns'].shift(1)
ttt = ttt.drop(columns=['forward_returns', 'risk_free_rate', 'market_forward_excess_returns'])
ttt.insert(loc=95, column='is_scored', value=False)

# train (テストデータ部分除外）
train = ttt.head(-180).copy()
train = train.drop(columns=['is_scored'])

# ｔest (擬似テストデータ)
test_sim = ttt.tail(180).copy().reset_index(drop=True)
true_forward_returns = data.tail(180)['forward_returns']

# 特徴量作成用 （create_features の最過去日準拠）
tail = pl.DataFrame(ttt.copy())
tail.write_parquet('./outputs/tail.parquet')
print('✅tail.parquet')

# zスコア用 （forward_returns のリストを予測値の累積として扱う）
preds_sim = pl.DataFrame(data[['forward_returns']].copy())
preds_sim.write_parquet('./outputs/preds_sim.parquet')
print('✅preds_sim.parquet')

# ===================================================================================================
# CatBoost Hyper Parameter（初期設定）
# ---------------------------------------------------------------------------------------------------
cat_params = {
    'loss_function': 'RMSE',
    'depth': 5,
    'learning_rate': 0.02,
    'l2_leaf_reg': 2,
    'random_strength': 0.8,
    'bagging_temperature': 0.8,
    'bootstrap_type': 'Bayesian',
    'iterations': 2000,
    'random_seed': 42,
    'verbose': False
}

# ===================================================================================================

# ===================================================================================================
# CatBoost Baseline CV
# ---------------------------------------------------------------------------------------------------
def catboost_cv(df_train, features, target='forward_returns', params=cat_params, n_splits=5, purge=5):
    
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

        # spearman score
        score = spearmanr(y_va, pred)[0]
        scores.append(score)
        
        print(f'Fold {fold} | Spearman: {score:.4f} | best_iter: {model.get_best_iteration()}')

    # 全体 OOF
    oof_score = spearmanr(df_train[target], oof)[0]
    print('\n======================')
    print(' CatBoost Baseline CV ')
    print('======================')
    print(f'Mean Spearman: {np.mean(scores):.5f}')
    print(f'OOF Spearman: {oof_score:.5f}')

    return oof

# ===================================================================================================

# ===================================================================================================
# CatBoost Baseline CV（スイープ用）
# ---------------------------------------------------------------------------------------------------
def catboost_cv_for_sweep(df_train, features, target='forward_returns', params=cat_params, n_splits=5, purge=5):
    
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

        # spearman score
        score = spearmanr(y_va, pred)[0]
        scores.append(score)
        
    # 全体 OOF
    oof_score = spearmanr(df_train[target], oof)[0]
    print(f'Mean Spearman: {np.mean(scores):.5f}')
    print(f'OOF Spearman: {oof_score:.5f}')

    return

# ===================================================================================================

# ===================================================================================================
# LightGBM Hyper Parameter（初期設定）
# ---------------------------------------------------------------------------------------------------
lgb_params = {
    'objective': 'regression',
    'metric': 'RMSE',
    'num_leaves': 31,
    'learning_rate': 0.02,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'max_depth': -1,
    'n_estimators': 2000,
    'random_state': 42,
    'verbose': -1
}

# ===================================================================================================

# ===================================================================================================
# LightGBM Baseline CV
# ---------------------------------------------------------------------------------------------------
def lightgbm_cv(df_train, features, target='forward_returns', params=lgb_params, n_splits=5, purge=5):
    
    cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge=purge)
    oof = np.zeros(len(df_train))
    scores = []

    for fold, (idx_tr, idx_va) in enumerate(cv.split(df_train)):
        x_tr, x_va = df_train.iloc[idx_tr][features], df_train.iloc[idx_va][features]
        y_tr, y_va = df_train.iloc[idx_tr][target], df_train.iloc[idx_va][target]
        model = LGBMRegressor(**params)
        model.fit(x_tr, y_tr, eval_set=(x_va, y_va), callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

        pred = model.predict(x_va, num_iteration=model.best_iteration_)
        oof[idx_va] = pred

        # spearman score
        score = spearmanr(y_va, pred)[0]
        scores.append(score)
        
        print(f'Fold {fold} | Spearman: {score:.4f} | best_iter: {model.best_iteration_}')

    # 全体 OOF
    oof_score = spearmanr(df_train[target], oof)[0]
    print('\n======================')
    print(' LightGBM Baseline CV ')
    print('======================')
    print(f'Mean Spearman: {np.mean(scores):.5f}')
    print(f'OOF Spearman: {oof_score:.5f}')

    return oof

# ===================================================================================================

# ===================================================================================================
# LightGBM Baseline CV（スイープ用）
# ---------------------------------------------------------------------------------------------------
def lightgbm_cv_for_sweep(df_train, features, target='forward_returns', params=lgb_params, n_splits=5, purge=5):
    
    cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge=purge)
    oof = np.zeros(len(df_train))
    scores = []

    for fold, (idx_tr, idx_va) in enumerate(cv.split(df_train)):
        x_tr, x_va = df_train.iloc[idx_tr][features], df_train.iloc[idx_va][features]
        y_tr, y_va = df_train.iloc[idx_tr][target], df_train.iloc[idx_va][target]
        model = LGBMRegressor(**params)
        model.fit(x_tr, y_tr, eval_set=(x_va, y_va), callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

        pred = model.predict(x_va, num_iteration=model.best_iteration_)
        oof[idx_va] = pred

        # spearman score
        score = spearmanr(y_va, pred)[0]
        scores.append(score)
        
    # 全体 OOF
    oof_score = spearmanr(df_train[target], oof)[0]
    print(f'Mean Spearman: {np.mean(scores):.5f}')
    print(f'OOF Spearman: {oof_score:.5f}')

    return oof

# ===================================================================================================

# ===================================================================================================
# 匿名特徴量 前処理
# ---------------------------------------------------------------------------------------------------
# 匿名特徴量リスト（'D~'以外）
anon_cols = [c for c in data.columns if c.startswith(('E','I','M','P','S','V'))]

# 匿名特徴量の格付け（=大小関係のエンコーディング）
neo_train = train.copy()
for c in anon_cols:
    neo_train[f'{c}_rank'] = neo_train[c].rank(method='average') / len(neo_train)

# 格付け匿名特徴量リスト（'D~'以外）
rank_cols = [f'{c}_rank' for c in anon_cols]

# ★ joblib.dump -> 'col_list.pkl'
joblib.dump({'anon_cols': anon_cols, 'rank_cols': rank_cols}, './outputs/col_list.pkl')

print('✅anon_cols, rank_cols(col_list.pkl)')

# ===================================================================================================

# ===================================================================================================
# macro PCA（主要因子 -景気/金利/リスク- 抽出）
# ---------------------------------------------------------------------------------------------------
macro_pca_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('macro_pca', PCA(n_components=3, random_state=42))
])
macro_pca_pipeline.fit(neo_train[rank_cols])

# ★ joblib.dump -> 'macro_pca_pipeline.pkl'
joblib.dump(macro_pca_pipeline, './outputs/macro_pca_pipeline.pkl')

print('✅macro_pca_pipeline.pkl')

# ===================================================================================================

# ===================================================================================================
# KMeans regime
# ---------------------------------------------------------------------------------------------------
kmeans_pipeline = Pipeline([
    ('kmeans', KMeans(n_clusters=6, random_state=42, n_init=20))
])
kmeans_pipeline.fit(macro_pca_pipeline.transform(neo_train[rank_cols]))

# ★ joblib.dump -> 'kmeans_pipeline.pkl'
joblib.dump(kmeans_pipeline, './outputs/kmeans_pipeline.pkl')

print('✅kmeans_pipeline.pkl')

# ===================================================================================================


# ===================================================================================================
# ★★★★★★★★★★★★★★★★★★ PCA（匿名特徴量の次元圧縮）for CatBoost ★★★★★★★★★★★★★★★★★★
# ---------------------------------------------------------------------------------------------------
cat_components_list = [36,30,31,32]
# ★ joblib.dump -> 'cat_components_list.pkl'
joblib.dump(cat_components_list, './outputs/cat_components_list.pkl')

for n_components in cat_components_list:
    pca_pipeline_for_cat = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, svd_solver='full'))
    ])

    # ★ Official Fit
    pca_pipeline_for_cat.fit(neo_train[rank_cols])

    # ★ joblib.dump -> 'pca_pipeline_n{n_components}.pkl'
    joblib.dump(pca_pipeline_for_cat, f'./outputs/pca_pipeline_for_cat_n{n_components}.pkl')

    print(f'✅pca_pipeline_for_cat_n{n_components}.pkl')

# ===================================================================================================


# ===================================================================================================
# ★★★★★★★★★★★★★★★★★★ PCA（匿名特徴量の次元圧縮）for LightGBM ★★★★★★★★★★★★★★★★★★
# ---------------------------------------------------------------------------------------------------
#lgb_components_list = [83,51,74,67,72]
## ★ joblib.dump -> 'cat_components_list.pkl'
#joblib.dump(lgb_components_list, './outputs/lgb_components_list.pkl')

pca_pipeline_for_lgb_n83 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=83, svd_solver='full'))
])

# ★ Official Fit
pca_pipeline_for_lgb_n83.fit(neo_train[rank_cols])

# ★ joblib.dump -> 'pca_pipeline_for_lgb_n83.pkl'
joblib.dump(pca_pipeline_for_lgb_n83, './outputs/pca_pipeline_for_lgb_n83.pkl')

print('✅pca_pipeline_for_lgb_n83.pkl')

# ===================================================================================================

# ===================================================================================================
# Features
# ---------------------------------------------------------------------------------------------------
def create_features(df):

    df = df.sort_values('date_id').copy()

    # 基本統計（分布）
    for w in [5,7,14,21,63]:
        df[f'kurt_{w}'] = df['lagged_forward_returns'].shift(1).rolling(w).kurt()

    # lag 系
    for k in [1,2,3,5,7]:
        df[f'lag_{k}'] = df['lagged_forward_returns'].shift(k)
        df[f'abs_lag_{k}'] = df[f'lag_{k}'].abs()
        
    # MA 系
    for w in [5,7,14]:
        df[f'ma_{w}'] = df['lagged_forward_returns'].shift(1).rolling(w).mean()

    # vol 系
    for w in [5,10,20]:
        df[f'vol_{w}'] = df['lagged_forward_returns'].shift(1).rolling(w).std()

    # Momentum
    for k in [1,3]:
        df[f'mom_{k}'] = df['lagged_forward_returns'].shift(1) - df['lagged_forward_returns'].shift(k+1)
    
    df['accel_3'] = df['lagged_forward_returns'].shift(1).diff(3)
    
    # Seasonal
    df['sin_5'] = np.sin(2*np.pi*df['date_id'] / 5)
    df['cos_5'] = np.cos(2*np.pi*df['date_id'] / 5)
    df['sin_252'] = np.sin(2*np.pi*df['date_id'] / 252)
    df['cos_252'] = np.cos(2*np.pi*df['date_id'] / 252)

    # Scaling features
    for k in [1,3,5,7]:
        df[f'scaled_lag_{k}'] = df[f'lag_{k}'] / (df['vol_20'] + 1e-6)
        df[f'lag_vol_{k}'] = df[f'lag_{k}'] * df['vol_20']

    return df

# ===================================================================================================

# ===================================================================================================
# Regime Features
# ---------------------------------------------------------------------------------------------------
def add_regime_features(df):

    # lagged_market_forward_excess_returns
    mfer = df['lagged_market_forward_excess_returns']
    df['mfer_ema_126'] = mfer.ewm(span=126, adjust=False).mean()
    df['mfer_ema_252'] = mfer.ewm(span=252, adjust=False).mean()
    df['mfer_ema_336'] = mfer.ewm(span=336, adjust=False).mean()

    # Market Vol Regime
    rv_21 = mfer.shift(1).rolling(21).std()
    rv_63 = mfer.shift(1).rolling(63).std()
    df['mkt_vol_regime_raw'] = rv_21 / (rv_63 + 1e-9)
    
    # smoothing
    df['mkt_vol_regime'] = df['mkt_vol_regime_raw'].ewm(span=21, adjust=False).mean()

    # VIX Proxy Regime
    df['vix_proxy'] = rv_21 / rv_63.rolling(5).mean()

    return df

# ===================================================================================================

# ===================================================================================================
# 匿名特徴量加工 for CatBoost（※ n_components 複数）
# ---------------------------------------------------------------------------------------------------
def processing_anon_features_for_cat(df, n_components):
    # ↑で保存済の joblib をロード
    loaded_pca = joblib.load(f'./outputs/pca_pipeline_for_cat_n{n_components}.pkl')

    # transform のみ
    pca_features = loaded_pca.transform(df[rank_cols])
    
    pca_df = pd.DataFrame(pca_features, columns=[f'pca_rank_{i+1}' for i in range(pca_features.shape[1])])
    df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)
    macro_factors = macro_pca_pipeline.transform(df[rank_cols])
    df['macro_f1'] = macro_factors[:, 0]
    df['macro_f2'] = macro_factors[:, 1]
    df['macro_f3'] = macro_factors[:, 2]
    df['cluster_regime'] = kmeans_pipeline.predict(macro_factors)
    df = df.drop(columns=anon_cols+rank_cols)
    return df

# ===================================================================================================

# ===================================================================================================
# Make Train Data for CatBoost（※ n_components 複数）
# ---------------------------------------------------------------------------------------------------
def quick_train_for_cat(df, n_components):
    # 複数 ver.
    df_train = processing_anon_features_for_cat(df, n_components)
    
    df_train = create_features(df_train)
    df_train = add_regime_features(df_train)
    df_train = df_train.dropna().reset_index(drop=True)
    x_train = df_train.copy()
    feature_cols = x_train.columns.tolist()
    y_train = data.loc[data['date_id'].isin(df_train['date_id'].unique()), 'forward_returns']
    return x_train, y_train, feature_cols

# ===================================================================================================

# ===================================================================================================
# 匿名特徴量加工 for LightGBM（※ n_components 単体）
# ---------------------------------------------------------------------------------------------------
def processing_anon_features_for_lgb(df):
    # ↑で保存済の joblib をロード
    loaded_pca = joblib.load('./outputs/pca_pipeline_for_lgb_n83.pkl')

    # transform のみ
    pca_features = loaded_pca.transform(df[rank_cols])
    
    pca_df = pd.DataFrame(pca_features, columns=[f'pca_rank_{i+1}' for i in range(pca_features.shape[1])])
    df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)
    macro_factors = macro_pca_pipeline.transform(df[rank_cols])
    df['macro_f1'] = macro_factors[:, 0]
    df['macro_f2'] = macro_factors[:, 1]
    df['macro_f3'] = macro_factors[:, 2]
    df['cluster_regime'] = kmeans_pipeline.predict(macro_factors)
    df = df.drop(columns=anon_cols+rank_cols)
    return df

# ===================================================================================================

# ===================================================================================================
# Make Train Data for LightGBM（※ n_components 単体）
# ---------------------------------------------------------------------------------------------------
def quick_train_for_lgb(df):
    # 単体 ver.
    df_train = processing_anon_features_for_lgb(df)
    
    df_train = create_features(df_train)
    df_train = add_regime_features(df_train)
    df_train = df_train.dropna().reset_index(drop=True)
    x_train = df_train.copy()
    feature_cols = x_train.columns.tolist()
    y_train = data.loc[data['date_id'].isin(df_train['date_id'].unique()), 'forward_returns']
    return x_train, y_train, feature_cols

# ===================================================================================================

# ===================================================================================================
# 匿名特徴量加工 for Sweep
# ---------------------------------------------------------------------------------------------------
def processing_anon_features_for_sweep(df, num_sweep):

    # LightGBM 用（※パワープレイ（笑））
    loaded_pca = joblib.load('./outputs/pca_pipeline_for_lgb_sweep.pkl')
    sweep_features = loaded_pca.transform(df[rank_cols])

    # CatBoost 用
    # Fit Each Time（使い捨て）★ KEYPOINT ★
    #pca_pipeline_for_sweep = Pipeline([
    #    ('imputer', SimpleImputer(strategy='median')),
    #    ('scaler', StandardScaler()),
    #    ('pca', PCA(n_components=num_sweep, svd_solver='full'))
    #])
    #sweep_features = pca_pipeline_for_sweep.fit_transform(df[rank_cols])
    
    sweep_df = pd.DataFrame(sweep_features, columns=[f'pca_rank_{i+1}' for i in range(sweep_features.shape[1])])
    df = pd.concat([df.reset_index(drop=True), sweep_df], axis=1)
    macro_factors = macro_pca_pipeline.transform(df[rank_cols])
    df['macro_f1'] = macro_factors[:, 0]
    df['macro_f2'] = macro_factors[:, 1]
    df['macro_f3'] = macro_factors[:, 2]
    df['cluster_regime'] = kmeans_pipeline.predict(macro_factors)
    df = df.drop(columns=anon_cols+rank_cols)
    return df

# ===================================================================================================

# ===================================================================================================
# Make Train Data for Sweep
# ---------------------------------------------------------------------------------------------------
def quick_train_for_sweep(df, num_sweep):
    df_train = processing_anon_features_for_sweep(df, num_sweep)
    df_train = create_features(df_train)
    df_train = add_regime_features(df_train)
    df_train = df_train.dropna().reset_index(drop=True)
    x_train = df_train.copy()
    feature_cols = x_train.columns.tolist()
    y_train = data.loc[data['date_id'].isin(df_train['date_id'].unique()), 'forward_returns']
    return x_train, y_train, feature_cols

# ===================================================================================================

# ===================================================================================================
# コンポーネント別 学習＆推論（★ cat_components_list = [36,30,31,32]）
# ---------------------------------------------------------------------------------------------------
def validation_catboost():
    oof_predictions = {} 
    for n_components in cat_components_list:
        x_train, y_train, feature_cols = quick_train_for_cat(neo_train, n_components)
        print('')
        print('-'*19)
        print(f'< n_components={n_components} >')
        print('-'*19)
        #confirm0(x_train)
        df_tmp = x_train.copy()
        df_tmp['forward_returns'] = y_train.reset_index(drop=True)
        current_oof = catboost_cv(df_tmp, feature_cols)
        oof_predictions[f'oof_n{n_components}'] = current_oof
    
    # 相関係数
    print('')
    print('-'*5, '< 相関（対n36）>', '-'*5)
    for k, v in oof_predictions.items():
        corr = spearmanr(oof_predictions['oof_n36'], v)[0]
        print(k, corr)
    
    # アンサンブル（単純平均）
    mean_ensemble_oof = np.mean(list(oof_predictions.values()), axis=0)
    print('')
    print('-'*5, '< アンサンブル（単純平均）結果 >', '-'*5)
    print(f'Ensemble_OOF_Spearman: {spearmanr(df_tmp["forward_returns"], mean_ensemble_oof)[0]}')

    # アンサンブル（重み付き平均）
    cat_weights = {'oof_n36': 0.65, 'oof_n30': 0.1, 'oof_n31': 0.15, 'oof_n32': 0.1}
    weight_ensemble_oof = sum(cat_weights[k] * v for k, v in oof_predictions.items())
    print('')
    print('-'*5, '< アンサンブル（重み付き平均）結果 >', '-'*5)
    print(f'Ensemble_OOF_Spearman: {spearmanr(df_tmp["forward_returns"], weight_ensemble_oof)[0]}')

    return oof_predictions

# 実行
oof_predictions = validation_catboost()


# ===================================================================================================
# PCA Sweep for CatBoost
# ---------------------------------------------------------------------------------------------------
def pca_sweep_cat():
    print('entry -> CatBoost')
    print('')
    for i in range (1, 85):
        print(f'---------- n_components = {i} ----------')
        x_train, y_train, feature_cols = quick_train_for_sweep(neo_train, i)
        df_tmp = x_train.copy()
        df_tmp['forward_returns'] = y_train.reset_index(drop=True)
        catboost_cv_for_sweep(df_tmp, feature_cols)

# ===================================================================================================
        
# 実行
#pca_sweep_cat()

# ===================================================================================================
# LightGBM Regressor
# ---------------------------------------------------------------------------------------------------
def validation_lightgbm():
    x_train, y_train, feature_cols = quick_train_for_lgb(neo_train)
    #confirm0(x_train)
    df_tmp = x_train.copy()
    df_tmp['forward_returns'] = y_train.reset_index(drop=True)
    print('')
    print(f'n_components -> {pca_pipeline_for_lgb_n83.named_steps["pca"].n_components_}')
    print('')
    lgb_oof = lightgbm_cv(df_tmp, feature_cols)
    lgb_comp = oof_predictions.copy()
    lgb_comp['oof_lgb83'] = lgb_oof
    print('')
    print('< 相関係数（対 CatBoost） >')
    for k, v in lgb_comp.items():
        corr = spearmanr(lgb_comp['oof_lgb83'], v)[0]
        print(k, corr)

    return lgb_comp, y_train

# ===================================================================================================

# 実行
lgb_comp, y_true = validation_lightgbm()

# ===================================================================================================
# PCA Sweep for lgb
# ---------------------------------------------------------------------------------------------------
def pca_sweep_lgb():
    print('entry -> LightGBM')
    print('')
    for i in range (1, 85):
    #for i in [72]:
        print(f'---------- n_components = {i} ----------')

        # パワープレイ（笑）
        pca_pipeline_for_lgb_sweep = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=i, svd_solver='full'))
        ])
        pca_pipeline_for_lgb_sweep.fit(neo_train[rank_cols])
        joblib.dump(pca_pipeline_for_lgb_sweep, './outputs/pca_pipeline_for_lgb_sweep.pkl')
        
        x_train, y_train, feature_cols = quick_train_for_sweep(neo_train, i)
        df_tmp = x_train.copy()
        df_tmp['forward_returns'] = y_train.reset_index(drop=True)
        sweep_oof = lightgbm_cv_for_sweep(df_tmp, feature_cols)
        lgb_comp = oof_predictions.copy()
        lgb_comp['oof_lgb'] = sweep_oof
        print('')
        print('< 相関係数（対 CatBoost） >')
        for k, v in lgb_comp.items():
            corr = spearmanr(lgb_comp['oof_lgb'], v)[0]
            print(k, corr)
        print('')

# ===================================================================================================

# 実行
#pca_sweep_lgb()

# ===================================================================================================
# Ridge Stacking
# ---------------------------------------------------------------------------------------------------
keys = ['oof_n36', 'oof_n30', 'oof_n31', 'oof_n32', 'oof_lgb83']
X = np.column_stack([lgb_comp[k] for k in keys])
scaler = StandardScaler(with_mean=True, with_std=True)

# Official Fit
X_std = scaler.fit_transform(X)

y = np.asarray(y_true).ravel()

# 指数 =0.55 確定
y_rank = np.sign(y) * (rankdata(np.abs(y)) ** 0.55)

# alpha =12.5 確定
ridge = Ridge(alpha=12.5, fit_intercept=True)

# Official Fit
ridge.fit(X_std, y_rank)

# ★ joblib.dump -> 'ridge_stack.pkl'
joblib.dump({'stack_scaler': scaler, 'ridge_for_weights': ridge}, './outputs/ridge_stack.pkl')


print('✅ridge_stack.pkl')

# ===================================================================================================

# ======================================================
# CatBoost Regressor
# ======================================================
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

cat_pipeline = Pipeline([
    ('model', CatBoostRegressor(**cat_params))
])

print(f'cat_components_list:{cat_components_list}')

# PCA 次元数別にモデル学習＆保存
for n_components in cat_components_list:
    
    # 学習用データ作成 & 特徴量選定
    x_train, y_train, feature_cols = quick_train_for_cat(neo_train, n_components=n_components)

    # Official Fit
    cat_pipeline.fit(x_train, y_train)

    # モデル保存
    joblib.dump({'cat_pipeline': cat_pipeline, 'cat_feature_cols': feature_cols}, f'./outputs/cat_pipeline_n{n_components}.pkl')

    print(f'✅cat_pipeline_n{n_components}.pkl')


# ======================================================
# LightGBM Regressor
# ======================================================
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

lgb_pipeline = Pipeline([
    ('model', LGBMRegressor(**lgb_params))
])

# 学習用データ作成 & 特徴量選定
x_train, y_train, feature_cols = quick_train_for_lgb(neo_train)

# Official Fit
lgb_pipeline.fit(x_train, y_train)

# モデル保存
joblib.dump({'lgb_pipeline': lgb_pipeline, 'lgb_feature_cols': feature_cols}, './outputs/lgb_pipeline_n83.pkl')

print('✅lgb_pipeline_n83.pkl')

# ---------------------------------
# 特徴量作成
# ---------------------------------
def create_features_for_test(df):

    df = df.sort_values('date_id').copy()

    # 基本統計（分布）
    for w in [5,7,14,21,63]:
        df[f'kurt_{w}'] = df['lagged_forward_returns'].shift(1).rolling(w).kurt()

    # lag 系
    for k in [1,2,3,5,7]:
        df[f'lag_{k}'] = df['lagged_forward_returns'].shift(k)
        df[f'abs_lag_{k}'] = df[f'lag_{k}'].abs()
        
    # MA 系
    for w in [5,7,14]:
        df[f'ma_{w}'] = df['lagged_forward_returns'].shift(1).rolling(w).mean()

    # vol 系
    for w in [5,10,20]:
        df[f'vol_{w}'] = df['lagged_forward_returns'].shift(1).rolling(w).std()

    # Momentum
    for k in [1,3]:
        df[f'mom_{k}'] = df['lagged_forward_returns'].shift(1) - df['lagged_forward_returns'].shift(k+1)
    
    df['accel_3'] = df['lagged_forward_returns'].shift(1).diff(3)
    
    # Seasonal
    df['sin_5'] = np.sin(2*np.pi*df['date_id'] / 5)
    df['cos_5'] = np.cos(2*np.pi*df['date_id'] / 5)
    df['sin_252'] = np.sin(2*np.pi*df['date_id'] / 252)
    df['cos_252'] = np.cos(2*np.pi*df['date_id'] / 252)
    
    # Scaling features
    for k in [1,3,5,7]:
        df[f'scaled_lag_{k}'] = df[f'lag_{k}'] / (df['vol_20'] + 1e-6)
        df[f'lag_vol_{k}'] = df[f'lag_{k}'] * df['vol_20']

    return df

# ---------------------------------
# Regime作成
# ---------------------------------
def add_regime_features_for_test(df):

    # lagged_market_forward_excess_returns
    mfer = df['lagged_market_forward_excess_returns']
    df['mfer_ema_126'] = mfer.ewm(span=126, adjust=False).mean()
    df['mfer_ema_252'] = mfer.ewm(span=252, adjust=False).mean()
    df['mfer_ema_336'] = mfer.ewm(span=336, adjust=False).mean()

    # Market Vol Regime
    rv_21 = mfer.shift(1).rolling(21).std()
    rv_63 = mfer.shift(1).rolling(63).std()
    df['mkt_vol_regime_raw'] = rv_21 / (rv_63 + 1e-9)
    
    # smoothing
    df['mkt_vol_regime'] = df['mkt_vol_regime_raw'].ewm(span=21, adjust=False).mean()

    # VIX Proxy Regime
    df['vix_proxy'] = rv_21 / rv_63.rolling(5).mean()
    
    return df

# ---------------------------------
# 匿名特徴量 rank 加工
# ---------------------------------
def make_rank_features_for_test(df):
    
    # anon_cols -> rank 化
    for c in anon_cols:
        df[f'{c}_rank'] = df[c].rank(method='average') / len(df)

    # macro PCA
    macro_factors = macro_pca_pipeline.transform(df[rank_cols])
    df['macro_f1'] = macro_factors[:, 0]
    df['macro_f2'] = macro_factors[:, 1]
    df['macro_f3'] = macro_factors[:, 2]

    # kmeans macro PCA
    df['cluster_regime'] = kmeans_pipeline.predict(macro_factors)

    return df

# ---------------------------------
# 匿名特徴量加工
# ---------------------------------
def processing_anon_features_for_test(df, pca_pipeline):
    
    # PCA
    pca_features = pca_pipeline.transform(df[rank_cols])
    pca_df = pd.DataFrame(pca_features, columns=[f'pca_rank_{i+1}' for i in range(pca_features.shape[1])])
    df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)
    
    # 不要カラム除去
    df = df.drop(columns=anon_cols+rank_cols)

    return df

def regime_window(vol_regime):
    if vol_regime > 1.2:      # high vol
        return 60
    elif vol_regime > 0.9:    # mid vol
        return 90
    else:                     # low vol
        return 120

def robust_zscore(x, eps=1e-9):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad)

def soft_threshold_scalar(z, th=0.25):    
    return 0.0 if abs(z) < th else z

def regime_scale(vol_regime):
    if vol_regime > 1.2:
        return 0.3
    elif vol_regime > 0.9:
        return 0.5
    else:
        return 0.7
    
def convert_to_position(preds_trace: np.ndarray,
                        vol_regime: float,
                        confidence: float,
                        cluster_scale: float,
                        vol_cap: float
                       ) -> float:

    # 想定外データ処理
    preds_trace = np.nan_to_num(preds_trace, nan=0.0, posinf=2.0, neginf=0.0)

    # Volatility Regime 別 window
    win = regime_window(vol_regime)
    window = preds_trace[-win:]
    if len(window) < 30:
        window = preds_trace
    
    # 標準化（ロバスト Z スコア：外れ値、regime変化に強い）
    z = float(robust_zscore(window)[-1])

    # 予測値が「誤差レベル(閾値=0.25)」の場合は position=1（ニュートラル） 
    z = soft_threshold_scalar(z, th=0.25)

    # regime scale
    scale = regime_scale(vol_regime)
    
    # base position
    base_pos = 1 + scale * np.tanh(z)

    # confidence 調整
    pos = base_pos * (1 + 0.5 * confidence)

    # cluster 調整
    pos *= cluster_scale

    # cap
    return float(np.clip(pos, 0.0, vol_cap))

def predict(test: pl.DataFrame, buffer=336) -> pl.DataFrame:
    
    global last_days, preds_trace

    # pl.DataFrame -> pd.DataFrame
    test_pd = test.to_pandas()
    tail_num = test_pd['date_id'][0]
    
    # 過去データの読み込み
    if last_days is None:
        last_days = tail.to_pandas().head(tail_num)
        preds_trace = preds_sim.head(tail_num)['forward_returns'].to_list()

    # 過去データの積み上げ
    last_days = pd.concat([last_days, test_pd], ignore_index=True)
    last_days = last_days.tail(buffer)
    last_days = last_days.sort_values(by='date_id')

    # 特徴量作成
    tmp = create_features_for_test(last_days)

    # regime作成
    tmp = add_regime_features_for_test(tmp)

    # 匿名特徴量 rank 加工
    tmp = make_rank_features_for_test(tmp)

    # 安全装置
    assert set(rank_cols).issubset(tmp.columns), 'rank_cols mismatch!'

    # PCA -> 当日分予測
    preds = []
    for pca_pipeline_for_cat, cat_pipeline, cat_feature_cols in zip(pca_pipeline_for_cat_list, cat_pipeline_list, cat_feature_cols_list):
        
        # 匿名特徴量加工
        pca_for_cat_df = processing_anon_features_for_test(tmp.copy(), pca_pipeline_for_cat)
        
        # 予測
        test_feature = pca_for_cat_df.tail(1)
        preds.append(cat_pipeline.predict(test_feature[cat_feature_cols])[0])

    for pca_pipeline_for_lgb, lgb_pipeline, lgb_feature_cols in zip(pca_pipeline_for_lgb_list, lgb_pipeline_list, lgb_feature_cols_list):
        
        # 匿名特徴量加工
        pca_for_lgb_df = processing_anon_features_for_test(tmp.copy(), pca_pipeline_for_lgb)
        
        # 予測
        test_feature = pca_for_lgb_df.tail(1)
        preds.append(lgb_pipeline.predict(test_feature[lgb_feature_cols])[0])
    
    # 重み付けアンサンブル（Ridge Stacking）
    preds_stack = np.array(preds).reshape(1, -1)
    preds_stack_std = stack_scaler.transform(preds_stack)
    w = ridge_for_weights.coef_.copy()
    w = w / (np.sum(np.abs(w)) + 1e-9)
    pred_today = (preds_stack_std @ w + ridge_for_weights.intercept_)[0]
    
    # NaN ガード
    pred_today = float(np.nan_to_num(pred_today, nan=0.0, posinf=0.0, neginf=0.0))

    # 予測値の累積
    preds_trace.append(pred_today)
    preds_trace = preds_trace[-(buffer):]

    # 当日の Volatility Regime 取得
    current_vol_regime = tmp['mkt_vol_regime'].iloc[-1]
    if not np.isfinite(current_vol_regime):
        current_vol_regime = 1.0

    # abs(pred) × vol_regime = 確信度
    confidence = np.tanh(0.5 * abs(pred_today) * current_vol_regime)
        
    # 当日の Cluster Regime 取得
    cluster = int(tmp['cluster_regime'].iloc[-1])
    cluster_scale = CLUSTER_SCALE.get(cluster, 1.0)

    # quantile bin（学習期間基準）
    try:
        vol_q = pd.qcut(tmp['mkt_vol_regime'].dropna(), q=[0, 0.3, 0.6, 1.0], labels=[0, 1, 2]).iloc[-1]
    except Exception:
        vol_q = 1  # mid vol fallback (=planB)
        
    # vol_regime × confidence
    cap = VOL_CAP[int(vol_q)]

    # position に変換
    position = convert_to_position(np.array(preds_trace),
                                   vol_regime=current_vol_regime,
                                   confidence=confidence,
                                   cluster_scale=cluster_scale,
                                   vol_cap=cap
                                  )

    return pl.DataFrame({'prediction': position}), pred_today

# model(CatBoost)
cat_pipeline_bundle_list = [
    joblib.load('./outputs/cat_pipeline_n36.pkl'),
    joblib.load('./outputs/cat_pipeline_n30.pkl'),
    joblib.load('./outputs/cat_pipeline_n31.pkl'),
    joblib.load('./outputs/cat_pipeline_n32.pkl')
]
cat_pipeline_list = [b['cat_pipeline'] for b in cat_pipeline_bundle_list]
cat_feature_cols_list = [b['cat_feature_cols'] for b in cat_pipeline_bundle_list]

# model(LightGBM)
lgb_pipeline_bundle_list = [
    joblib.load('./outputs/lgb_pipeline_n83.pkl')
]
lgb_pipeline_list = [b['lgb_pipeline'] for b in lgb_pipeline_bundle_list]
lgb_feature_cols_list = [b['lgb_feature_cols'] for b in lgb_pipeline_bundle_list]

# PCA for CatBoost
pca_pipeline_for_cat_bundle_list = [
    joblib.load('./outputs/pca_pipeline_for_cat_n36.pkl'),
    joblib.load('./outputs/pca_pipeline_for_cat_n30.pkl'),
    joblib.load('./outputs/pca_pipeline_for_cat_n31.pkl'),
    joblib.load('./outputs/pca_pipeline_for_cat_n32.pkl')
]
pca_pipeline_for_cat_list = [i for i in pca_pipeline_for_cat_bundle_list]

# PCA for LightGBM
pca_pipeline_for_lgb_bundle_list = [
    joblib.load('./outputs/pca_pipeline_for_lgb_n83.pkl')
]
pca_pipeline_for_lgb_list = [i for i in pca_pipeline_for_lgb_bundle_list]

# column_list
col_list_bundle = joblib.load('./outputs/col_list.pkl')
anon_cols = col_list_bundle['anon_cols']
rank_cols = col_list_bundle['rank_cols']

# macro PCA
macro_pca_pipeline = joblib.load('./outputs/macro_pca_pipeline.pkl')

# KMeans
kmeans_pipeline = joblib.load('./outputs/kmeans_pipeline.pkl')

# cat_components_list
cat_components_list = joblib.load('./outputs/cat_components_list.pkl')

# Ridge Stack
ridge_stack_bundle = joblib.load('./outputs/ridge_stack.pkl')
stack_scaler = ridge_stack_bundle['stack_scaler']
ridge_for_weights = ridge_stack_bundle['ridge_for_weights']

# data
tail = pl.read_parquet('./outputs/tail.parquet')
preds_sim = pl.read_parquet('./outputs/preds_sim.parquet')

# エラー非表示
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in greater')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in less')

# cluster scale 確定
CLUSTER_SCALE = {
    0: 1.0,   # normal
    1: 0.8,   # risk-off
    2: 1.25,   # trend
    3: 0.6,   # crash / chaos
    4: 1.15,
    5: 0.9
}

# vol cap
VOL_CAP = {
    0: 1.8,   # low vol → 強く張れる
    1: 1.4,
    2: 1.0    # high vol → 抑える
}

# last_days 初期化
last_days = None

# Gateway 再現
def reproduction_gateway():
    pos_list = []
    pred_list = []
    for i in list(test_sim['date_id']):
        test = pl.DataFrame(test_sim[test_sim['date_id']==i])
        pos, pred = predict(test, buffer=336)
        pos_list.append(pos)
        pred_list.append(pred)

    metrics = evaluate_model(true_forward_returns, pred_list)
    print(f'metrics：{metrics}')

    # 実際の提出データと同じもの
    submission = pl.concat(pos_list)

    # シャープレシオ
    calc_sharpe_ratio(data.tail(180), list(submission.to_pandas()['prediction']))

    print(submission)


reproduction_gateway()
