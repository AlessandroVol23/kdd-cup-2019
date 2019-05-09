import pandas as pd
import numpy as np


def create_query_file(df_r):
    df = df_r.copy()

    df = df.sort_values("sid")

    df = df.reset_index(drop=True)

    df = df.assign(row=df.index)

    df.row = df.row+1

    sid_rows = pd.DataFrame(df.groupby("sid").last()["row"])

    sid_diff = sid_rows.assign(difference=sid_rows.diff())

    sid_diff.iloc[0, 1] = sid_diff.iloc[0, 0]

    query = sid_diff.difference.values

    return query


def lgbm_train(x_train, x_test, num_leaves=61, max_depth=-1, n_estimators=100, subsample=0.8, min_child_samples=50,
               learning_rate=0.05, random_state=2019):
    lgb_model = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=61, reg_alpha=0, reg_lambda=0.01,
                                   max_depth=-1, n_estimators=100, objective='multiclass', subsample=0.8, colsample_bytree=0.8,
                                   subsample_freq=1, min_child_samples=50,  learning_rate=0.05, random_state=2019, metric="multiclass", n_jobs=-1)

    lgb_model.fit(x_train, y_train, verbose=10)

    return lgb_model
