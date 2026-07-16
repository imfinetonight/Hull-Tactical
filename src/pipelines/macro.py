import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def macro_kmeans_pipeline(df_train_ranked, rank_cols):
    # macro PCA（主要因子抽出 -景気/金利/リスク-）
    macro_pca_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('macro_pca', PCA(n_components=3, random_state=42))
    ])
    macro_pca_pipeline.fit(df_train_ranked[rank_cols])
    joblib.dump(macro_pca_pipeline, './outputs/macro_pca_pipeline.pkl')
    print("✅saved 'macro_pca_pipeline.pkl'")

    # KMeans regime
    kmeans_pipeline = Pipeline([
        ('kmeans', KMeans(n_clusters=6, random_state=42, n_init=20))
    ])
    kmeans_pipeline.fit(macro_pca_pipeline.transform(df_train_ranked[rank_cols]))
    joblib.dump(kmeans_pipeline, './outputs/kmeans_pipeline.pkl')
    print("✅saved 'kmeans_pipeline.pkl'")
