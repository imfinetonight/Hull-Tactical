# ===================================================================================================
# PCA Sweep for CatBoost
# ---------------------------------------------------------------------------------------------------
def pca_sweep_cat():
    print('entry -> CatBoost')
    print('')
    for i in range (1, 85):
        print(f'---------- n_components = {i} ----------')
        x_train, y_train, feature_cols = quick_train_for_sweep(neo_train, i)
        df_tmp = x_train.copy()
        df_tmp['forward_returns'] = y_train.reset_index(drop=True)
        catboost_cv_for_sweep(df_tmp, feature_cols)

# 実行
#pca_sweep_cat()
# ===================================================================================================


# ===================================================================================================
# PCA Sweep for lgb
# ---------------------------------------------------------------------------------------------------
def pca_sweep_lgb():
    print('entry -> LightGBM')
    print('')
    for i in range (1, 85):
    #for i in [72]:
        print(f'---------- n_components = {i} ----------')
        pca_pipeline_for_lgb_sweep = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=i, svd_solver='full'))
        ])
        pca_pipeline_for_lgb_sweep.fit(neo_train[rank_cols])
        joblib.dump(pca_pipeline_for_lgb_sweep, './outputs/pca_pipeline_for_lgb_sweep.pkl')
        
        x_train, y_train, feature_cols = quick_train_for_sweep(neo_train, i)
        df_tmp = x_train.copy()
        df_tmp['forward_returns'] = y_train.reset_index(drop=True)
        sweep_oof = lightgbm_cv_for_sweep(df_tmp, feature_cols)
        lgb_comp = oof_predictions.copy()
        lgb_comp['oof_lgb'] = sweep_oof
        print('')
        print('< 相関係数（対 CatBoost） >')
        for k, v in lgb_comp.items():
            corr = spearmanr(lgb_comp['oof_lgb'], v)[0]
            print(k, corr)
        print('')


# 実行
#pca_sweep_lgb()
# ===================================================================================================