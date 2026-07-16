import polars as pl


def save_data(train2test, train):
    # 特徴量作成用
    tail = pl.DataFrame(train2test.copy())
    tail.write_parquet('./outputs/tail.parquet')
    print("✅saved 'tail.parquet'")

    # zスコア用 （forward_returns のリストを予測値の累積として扱う）
    preds_sim = pl.DataFrame(train[['forward_returns']].copy())
    preds_sim.write_parquet('./outputs/preds_sim.parquet')
    print("✅saved 'preds_sim.parquet'")