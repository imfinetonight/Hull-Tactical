import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.config import config as cfg


def macro_kmeans_pipeline(df_train_ranked, rank_cols):
    # macro PCA（主要因子抽出 -景気/金利/リスク-）
    macro_pca_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('macro_pca', PCA(n_components=3, random_state=cfg.SEED))
    ])
    macro_pca_pipeline.fit(df_train_ranked[rank_cols])
    joblib.dump(macro_pca_pipeline, f'{cfg.OUT_DIR}/macro_pca_pipeline.pkl')
    print("✅saved 'macro_pca_pipeline.pkl'")

    # KMeans regime
    kmeans_pipeline = Pipeline([
        ('kmeans', KMeans(n_clusters=cfg.n_clusters, random_state=cfg.SEED, n_init=cfg.n_init))
    ])
    kmeans_pipeline.fit(macro_pca_pipeline.transform(df_train_ranked[rank_cols]))
    joblib.dump(kmeans_pipeline, f'{cfg.OUT_DIR}/kmeans_pipeline.pkl')
    print("✅saved 'kmeans_pipeline.pkl'")
