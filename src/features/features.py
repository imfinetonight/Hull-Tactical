import numpy as np
import pandas as pd

from src.data.loader import load_data
from src.data.dataset import make_train_test
from src.pipelines.rank import rank_encoding
from src.inference.load_artifacts import load_col_list, load_macro_pipeline, load_kmeans_pipeline


def create_features(df):
    df = df.sort_values('date_id').copy()

    # 基本統計
    for w in [5,7,14,21,63]:
        #df[f'skew_{w}'] = df['lagged_forward_returns'].shift(1).rolling(w).skew()
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


def add_regime_features(df):
    # EMA
    mfer = df['lagged_market_forward_excess_returns']
    df['mfer_ema_126'] = mfer.ewm(span=126, adjust=False).mean()
    df['mfer_ema_252'] = mfer.ewm(span=252, adjust=False).mean()
    df['mfer_ema_336'] = mfer.ewm(span=336, adjust=False).mean()

    # Market Vol Regime
    rv_21 = mfer.shift(1).rolling(21).std()
    rv_63 = mfer.shift(1).rolling(63).std()
    df['mkt_vol_regime_raw'] = rv_21 / (rv_63 + 1e-9)
    df['mkt_vol_regime'] = df['mkt_vol_regime_raw'].ewm(span=21, adjust=False).mean() # smoothing

    # VIX Proxy
    df['vix_proxy'] = rv_21 / rv_63.rolling(5).mean()
    
    return df


def add_macro_features(df, anon_cols, rank_cols, macro_pca_pipeline, kmeans_pipeline):
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


def add_anon_features(df, pca_pipeline, anon_cols, rank_cols):
    # PCA
    pca_features = pca_pipeline.transform(df[rank_cols])
    pca_df = pd.DataFrame(pca_features, columns=[f'pca_rank_{i+1}' for i in range(pca_features.shape[1])])
    df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)
    
    # 不要カラム除去
    df = df.drop(columns=anon_cols+rank_cols)
    return df


def feature_extraction():
    train, _ = load_data()  
    anon_cols, rank_cols = load_col_list()
    macro_pca_pipeline = load_macro_pipeline()
    kmeans_pipeline = load_kmeans_pipeline()
    df_train, _, _, _ = make_train_test(train)
    df_train_ranked, _ = rank_encoding(train, df_train)
    df_train = create_features(df_train_ranked)
    df_train = add_regime_features(df_train)
    df_train = add_macro_features(df_train, anon_cols, rank_cols, macro_pca_pipeline, kmeans_pipeline)
    y_train = train.loc[train['date_id'].isin(df_train['date_id'].unique()), 'forward_returns']
    return df_train, y_train, anon_cols, rank_cols
