import polars as pl

from src.inference.prediction import predict
from src.utils.evaluation import evaluate_model, calc_sharpe_ratio
from src.config import config as cfg


def gateway_emulator(train, df_test, true_forward_returns):
    pos_list = []
    pred_list = []
    for i in list(df_test['date_id']):
        test = pl.DataFrame(df_test[df_test['date_id']==i])
        pos, pred = predict(test, buffer=cfg.BUFFER, flg=True)
        pos_list.append(pos)
        pred_list.append(pred)
    metrics = evaluate_model(true_forward_returns, pred_list)
    print(f'metrics：{metrics}')
    submission = pl.concat(pos_list)
    calc_sharpe_ratio(train.tail(180), list(submission.to_pandas()['prediction']))
    print(submission)