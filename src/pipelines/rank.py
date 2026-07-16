import joblib


def rank_encoding(train, df_train):
    # 匿名特徴量リスト（'D~'以外）
    anon_cols = [c for c in train.columns if c.startswith(('E','I','M','P','S','V'))]

    # 匿名特徴量の格付け（=大小関係のエンコーディング）
    df_train_ranked = df_train.copy()
    for c in anon_cols:
        df_train_ranked[f'{c}_rank'] = df_train_ranked[c].rank(method='average') / len(df_train_ranked)

    # 格付け匿名特徴量リスト（'D~'以外）
    rank_cols = [f'{c}_rank' for c in anon_cols]

    # ★ joblib.dump -> 'col_list.pkl'
    joblib.dump({'anon_cols': anon_cols, 'rank_cols': rank_cols}, './outputs/col_list.pkl')
    print("✅saved 'col_list.pkl'")

    return df_train_ranked, rank_cols