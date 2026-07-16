import numpy as np
import pandas as pd
import polars as pl

from src.features.features import create_features, add_regime_features, add_macro_features, add_anon_features
from src.inference.position_sizing import convert_to_position
from src.config import config as cfg
import src.inference.state as state
from src.inference.initialize import initialize


def predict(test: pl.DataFrame, buffer=cfg.BUFFER, flg=False) -> pl.DataFrame:
    test_pd = test.to_pandas()
    tail_num = test_pd['date_id'].iloc[0]

    initialize(tail_num)

    # test data 累積
    state.last_days = pd.concat([state.last_days, test_pd], ignore_index=True)
    state.last_days = state.last_days.tail(buffer)
    state.last_days = state.last_days.sort_values(by='date_id')

    # 特徴量
    tmp = create_features(state.last_days)
    tmp = add_regime_features(tmp)
    tmp = add_macro_features(tmp, state.anon_cols, state.rank_cols, state.macro_pca_pipeline, state.kmeans_pipeline)

    assert set(state.rank_cols).issubset(tmp.columns), 'rank_cols mismatch!'

    # PCA -> 予測
    preds = []
    for cat_pipeline, cat_feature_cols, pca_pipeline_for_cat in state.cat_pipelines:
        pca_for_cat_df = add_anon_features(tmp.copy(), pca_pipeline_for_cat, state.anon_cols, state.rank_cols)
        test_feature = pca_for_cat_df.tail(1)
        preds.append(cat_pipeline.predict(test_feature[cat_feature_cols])[0])

    for lgb_pipeline, lgb_feature_cols, pca_pipeline_for_lgb in state.lgb_pipelines:
        pca_for_lgb_df = add_anon_features(tmp.copy(), pca_pipeline_for_lgb, state.anon_cols, state.rank_cols)
        test_feature = pca_for_lgb_df.tail(1)
        preds.append(lgb_pipeline.predict(test_feature[lgb_feature_cols])[0])
    
    # Ridge Stacking
    preds_stack = np.array(preds).reshape(1, -1)
    preds_stack_std = state.stack_scaler.transform(preds_stack)
    w = state.ridge_for_weights.coef_.copy()
    w = w / (np.sum(np.abs(w)) + 1e-9)
    pred_today = (preds_stack_std @ w + state.ridge_for_weights.intercept_)[0]

    # NaN ガード
    pred_today = float(np.nan_to_num(pred_today, nan=0.0, posinf=0.0, neginf=0.0))

    # pred 累積
    state.preds_trace.append(pred_today)
    state.preds_trace = state.preds_trace[-(buffer):]

    # 当日の Volatility Regime
    current_vol_regime = tmp['mkt_vol_regime'].iloc[-1]
    if not np.isfinite(current_vol_regime):
        current_vol_regime = 1.0

    # abs(pred) × vol_regime = confidence
    confidence = np.tanh(0.5 * abs(pred_today) * current_vol_regime)
        
    # 当日の Cluster Regime
    cluster = int(tmp['cluster_regime'].iloc[-1])
    cluster_scale = cfg.CLUSTER_SCALE.get(cluster, 1.0)

    # quantile bin
    try:
        vol_q = pd.qcut(tmp['mkt_vol_regime'].dropna(), q=[0, 0.3, 0.6, 1.0], labels=[0, 1, 2]).iloc[-1]
    except Exception:
        vol_q = 1
        
    # vol_regime × confidence
    cap = cfg.VOL_CAP[int(vol_q)]

    # position sizing
    position = convert_to_position(np.array(state.preds_trace),
                                   vol_regime=current_vol_regime,
                                   confidence=confidence,
                                   cluster_scale=cluster_scale,
                                   vol_cap=cap
                                  )
    if flg:
        return pl.DataFrame({'prediction': position}), pred_today
    
    return pl.DataFrame({'prediction': position})