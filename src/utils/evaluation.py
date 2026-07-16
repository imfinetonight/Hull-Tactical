import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error


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