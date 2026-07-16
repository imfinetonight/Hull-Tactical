import numpy as np


def _regime_window(vol_regime):
    if vol_regime > 1.2:
        return 60
    elif vol_regime > 0.9:
        return 90
    else:
        return 120

def _robust_zscore(x, eps=1e-9):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad)

def _soft_threshold_scalar(z, th=0.25):
    return 0.0 if abs(z) < th else z

def _regime_scale(vol_regime):
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
    win = _regime_window(vol_regime)
    window = preds_trace[-win:]
    if len(window) < 30:
        window = preds_trace
    
    # 標準化（対　外れ値・regime変化）
    z = float(_robust_zscore(window)[-1])

    # 予測値が「誤差レベル(閾値=0.25)」の場合は position=1
    z = _soft_threshold_scalar(z, th=0.25)

    # regime scale
    scale = _regime_scale(vol_regime)
    
    # base position
    base_pos = 1 + scale * np.tanh(z)

    # confidence 調整
    pos = base_pos * (1 + 0.5 * confidence)

    # cluster 調整
    pos *= cluster_scale

    # cap
    return float(np.clip(pos, 0.0, vol_cap))