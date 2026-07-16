import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.config import config as cfg


def pca_pipeline_for_cat(df_train_ranked, rank_cols, cat_components_list):
    for n_components in cat_components_list:
        pca_pipeline_for_cat = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, svd_solver='full'))
        ])
        pca_pipeline_for_cat.fit(df_train_ranked[rank_cols])
        joblib.dump(pca_pipeline_for_cat, f'{cfg.OUT_DIR}/pca_pipeline_for_cat_n{n_components}.pkl')
        print(f"✅saved 'pca_pipeline_for_cat_n{n_components}.pkl'")


def pca_pipeline_for_lgb(df_train_ranked, rank_cols, lgb_components_list):
    for n_components in lgb_components_list:
        pca_pipeline_for_lgb = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, svd_solver='full'))
        ])
        pca_pipeline_for_lgb.fit(df_train_ranked[rank_cols])
        joblib.dump(pca_pipeline_for_lgb, f'{cfg.OUT_DIR}/pca_pipeline_for_lgb_n{n_components}.pkl')
        print(f"✅saved 'pca_pipeline_for_lgb_n{n_components}.pkl'")