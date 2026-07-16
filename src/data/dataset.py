


def make_train_test(train):
    train2test = train.copy()
    train2test['lagged_forward_returns'] = train2test['forward_returns'].shift(1)
    train2test['lagged_risk_free_rate'] = train2test['risk_free_rate'].shift(1)
    train2test['lagged_market_forward_excess_returns'] = train2test['market_forward_excess_returns'].shift(1)
    train2test = train2test.drop(columns=['forward_returns', 'risk_free_rate', 'market_forward_excess_returns'])
    train2test.insert(loc=95, column='is_scored', value=False)

    # train (テストデータ部分除外）
    df_train = train2test.head(-180).copy()
    df_train = df_train.drop(columns=['is_scored'])

    # test (擬似テストデータ)
    df_test = train2test.tail(180).copy().reset_index(drop=True)
    true_forward_returns = train.tail(180)['forward_returns']

    return df_train, df_test, true_forward_returns, train2test
