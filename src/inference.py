# ─── Libraries ───
import os
import pandas as pd
import polars as pl
import numpy as np
import joblib

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

import sys
sys.path.append(os.getcwd())
ARTIFACTS_DIR = 'outputs'
DATA_DIR = 'data'

# ─── Loading Models and Data ───
# model(CatBoost)
cat_pipeline_bundle_list = [
    joblib.load(f'{ARTIFACTS_DIR}/cat_pipeline_n36.pkl'),
    joblib.load(f'{ARTIFACTS_DIR}/cat_pipeline_n30.pkl'),
    joblib.load(f'{ARTIFACTS_DIR}/cat_pipeline_n31.pkl'),
    joblib.load(f'{ARTIFACTS_DIR}/cat_pipeline_n32.pkl')
]
cat_pipeline_list = [b['cat_pipeline'] for b in cat_pipeline_bundle_list]
cat_feature_cols_list = [b['cat_feature_cols'] for b in cat_pipeline_bundle_list]

# model(LightGBM)
lgb_pipeline_bundle_list = [
    joblib.load(f'{ARTIFACTS_DIR}/lgb_pipeline_n83.pkl')
]
lgb_pipeline_list = [b['lgb_pipeline'] for b in lgb_pipeline_bundle_list]
lgb_feature_cols_list = [b['lgb_feature_cols'] for b in lgb_pipeline_bundle_list]

# column list
col_list_bundle = joblib.load(f'{ARTIFACTS_DIR}/col_list.pkl')
anon_cols = col_list_bundle['anon_cols']
rank_cols = col_list_bundle['rank_cols']

# PCA for CatBoost
pca_pipeline_for_cat_bundle_list = [
    joblib.load(f'{ARTIFACTS_DIR}/pca_pipeline_for_cat_n36.pkl'),
    joblib.load(f'{ARTIFACTS_DIR}/pca_pipeline_for_cat_n30.pkl'),
    joblib.load(f'{ARTIFACTS_DIR}/pca_pipeline_for_cat_n31.pkl'),
    joblib.load(f'{ARTIFACTS_DIR}/pca_pipeline_for_cat_n32.pkl')
]
pca_pipeline_for_cat_list = [i for i in pca_pipeline_for_cat_bundle_list]

# PCA for LightGBM
pca_pipeline_for_lgb_bundle_list = [
    joblib.load(f'{ARTIFACTS_DIR}/pca_pipeline_for_lgb_n83.pkl')
]
pca_pipeline_for_lgb_list = [i for i in pca_pipeline_for_lgb_bundle_list]

# macro PCA
macro_pca_pipeline = joblib.load(f'{ARTIFACTS_DIR}/macro_pca_pipeline.pkl')

# KMeans regime
kmeans_pipeline = joblib.load(f'{ARTIFACTS_DIR}/kmeans_pipeline.pkl')

# Ridge Stacking
ridge_stack_bundle = joblib.load(f'{ARTIFACTS_DIR}/ridge_stack.pkl')
stack_scaler = ridge_stack_bundle['stack_scaler']
ridge_for_weights = ridge_stack_bundle['ridge_for_weights']

# data
tail = pl.read_parquet(f'{ARTIFACTS_DIR}/tail.parquet')
preds_sim = pl.read_parquet(f'{ARTIFACTS_DIR}/preds_sim.parquet')


# ─── Prameters ───
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


# ─── Feature Extraction Functions ───
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

# Regime
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

# Anon Features -> Rank + PCA + KMeans
def make_rank_features(df):
    
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

def processing_anon_features(df, pca_pipeline):
    
    # PCA
    pca_features = pca_pipeline.transform(df[rank_cols])
    pca_df = pd.DataFrame(pca_features, columns=[f'pca_rank_{i+1}' for i in range(pca_features.shape[1])])
    df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)
    
    # 不要カラム除去
    df = df.drop(columns=anon_cols+rank_cols)

    return df


# Prediction -> Position
def regime_window(vol_regime):
    if vol_regime > 1.2:
        return 60
    elif vol_regime > 0.9:
        return 90
    else:
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


# ─── Inference Function ───
def predict(test: pl.DataFrame, buffer=315) -> pl.DataFrame:
    
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
    tmp = create_features(last_days)

    # regime作成
    tmp = add_regime_features(tmp)

    # 匿名特徴量 rank 加工
    tmp = make_rank_features(tmp)

    # 安全装置
    assert set(rank_cols).issubset(tmp.columns), 'rank_cols mismatch!'

    # PCA -> 当日分予測
    preds = []
    for pca_pipeline_for_cat, cat_pipeline, cat_feature_cols in zip(pca_pipeline_for_cat_list, cat_pipeline_list, cat_feature_cols_list):
        
        # 匿名特徴量加工
        pca_for_cat_df = processing_anon_features(tmp.copy(), pca_pipeline_for_cat)
        
        # 予測
        test_feature = pca_for_cat_df.tail(1)
        preds.append(cat_pipeline.predict(test_feature[cat_feature_cols])[0])

    for pca_pipeline_for_lgb, lgb_pipeline, lgb_feature_cols in zip(pca_pipeline_for_lgb_list, lgb_pipeline_list, lgb_feature_cols_list):
        
        # 匿名特徴量加工
        pca_for_lgb_df = processing_anon_features(tmp.copy(), pca_pipeline_for_lgb)
        
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
        # plan B
        vol_q = 1
        
    # vol_regime × confidence
    cap = VOL_CAP[int(vol_q)]

    # position に変換
    position = convert_to_position(np.array(preds_trace),
                                   vol_regime=current_vol_regime,
                                   confidence=confidence,
                                   cluster_scale=cluster_scale,
                                   vol_cap=cap
                                  )

    return pl.DataFrame({'prediction': position})


# ─── Run Inference Server ───
last_days = None

try:
    import kaggle_evaluation.default_inference_server
    inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)
    
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))

except ModuleNotFoundError:
    print('Local environment detected. Launching inference emulator...')
    
    test_path = os.path.join(DATA_DIR, 'test.csv')
    full_test_df = pl.read_csv(test_path)
    print(f'Simulating stream inference for {min(len(full_test_df), 10)} steps...')
    
    for i in range(min(len(full_test_df), 10)):
        row_df = full_test_df.slice(i, 1)
        res_df = predict(row_df)
        print(f'Date ID: {row_df["date_id"][0]} -> Position Score: {res_df["prediction"][0]:.4f}')
        
    print('✅ Local inference completed successfully!')
