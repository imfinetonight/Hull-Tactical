from src.config import config as cfg
from src.data.loader import load_data
from src.data.dataset import make_train_test
from src.data.saver import save_data
from src.pipelines.rank import rank_encoding
from src.pipelines.macro import macro_kmeans_pipeline
from src.pipelines.pca import pca_pipeline_for_cat, pca_pipeline_for_lgb
from src.train.models import train_catboost, train_lightgbm, ridge_stacking
from src.train.emulator import gateway_emulator


def main():
    train, _ = load_data()
    df_train, df_test, true_forward_returns, train2test = make_train_test(train)
    save_data(train2test, train)
    df_train_ranked, rank_cols = rank_encoding(train, df_train)
    macro_kmeans_pipeline(df_train_ranked, rank_cols)
    pca_pipeline_for_cat(df_train_ranked, rank_cols, cfg.cat_components_list)
    pca_pipeline_for_lgb(df_train_ranked, rank_cols, cfg.lgb_components_list)
    train_catboost()
    train_lightgbm()
    ridge_stacking()

    gateway_emulator(train, df_test, true_forward_returns)


if __name__ == "__main__":
    main()