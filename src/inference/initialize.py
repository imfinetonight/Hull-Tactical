import src.inference.state as state

from src.inference.load_artifacts import (
    load_models,
    load_macro_pipeline,
    load_kmeans_pipeline,
    load_col_list,
    load_ridge_stack,
    load_parquet,
)

def initialize(tail_num):

    if state.last_days is not None:
        return

    tail, preds_sim = load_parquet()

    state.last_days = tail.to_pandas().head(tail_num)
    state.preds_trace = preds_sim.head(tail_num)['forward_returns'].to_list()

    state.cat_pipelines = load_models('cat')
    state.lgb_pipelines = load_models('lgb')

    state.macro_pca_pipeline = load_macro_pipeline()
    state.kmeans_pipeline = load_kmeans_pipeline()

    state.anon_cols, state.rank_cols = load_col_list()

    state.stack_scaler, state.ridge_for_weights = load_ridge_stack()