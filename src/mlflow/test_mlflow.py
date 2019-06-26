import logmlflow as x
import pandas as pd


summary = pd.read_csv("../data/Summaries/XGBoost/Summary16.csv")

metrics_dict, parameter_dict, rest_param_dict, feature_dict = x.mlflow_dict(summary)
x.mlflow_log(2, metrics_dict, parameter_dict, rest_param_dict, feature_dict)
