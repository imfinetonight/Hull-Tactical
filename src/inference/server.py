import os
import polars as pl

from src.config import config as cfg
from src.inference.prediction import predict


def run_server():
    try:
        import kaggle_evaluation.default_inference_server
        inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)
        
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            inference_server.serve()
        else:
            inference_server.run_local_gateway((cfg.COMPETITION_DIR,))

    except ModuleNotFoundError:
        test_path = os.path.join(cfg.INPUT_DIR, 'test.csv')
        full_test_df = pl.read_csv(test_path)
        for i in range(min(len(full_test_df), 10)):
            row_df = full_test_df.slice(i, 1)
            res_df = predict(row_df)
            print(f'Date ID: {row_df["date_id"][0]} -> Position Score: {res_df["prediction"][0]:.4f}')
        print('✅ Local inference completed successfully!')