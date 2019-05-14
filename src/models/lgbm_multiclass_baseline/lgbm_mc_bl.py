import pandas as pd
import numpy as np


def lgbm_train(x_train, x_test, num_leaves=61, max_depth=-1, n_estimators=100, subsample=0.8, min_child_samples=50,
               learning_rate=0.05, random_state=2019):
    lgb_model = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=61, reg_alpha=0, reg_lambda=0.01,
                                   max_depth=-1, n_estimators=100, objective='multiclass', subsample=0.8, colsample_bytree=0.8,
                                   subsample_freq=1, min_child_samples=50,  learning_rate=0.05, random_state=2019, metric="multiclass", n_jobs=-1)

    lgb_model.fit(x_train, y_train, verbose=10)

    return lgb_model
