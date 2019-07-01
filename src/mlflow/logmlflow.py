import pysftp
from paramiko import sftp
import mlflow as mlf
import pandas as pd
import json


'''
model = 0 (Default), 1 (LightGBM), 2 (Multiclass), 3 (TF-Ranking)
features passed saved as dictionary 
F1 result of model
CVS of trained model
'''


def mlflow_log(model, metrics, parameters, rest_params, features):
    cnopts = pysftp.CnOpts()

    with pysftp.Connection('138.246.233.10', username='ubuntu', private_key='~/.ssh/id_rsa', port=22, cnopts=cnopts):
        mlf.set_tracking_uri('http://138.246.233.10:80')
        print('SFTP connection successful')

    if model == 1:
        experiment_name = 'LightGBM'
    elif model == 2:
        experiment_name = 'Multiclass'
    elif model == 3:
        experiment_name = 'TF-Ranking'
    else:
        print('Model is missing!')

    experiment_id = model
    mlf.set_experiment(experiment_name)

    with mlf.start_run(experiment_id=experiment_id, run_name=experiment_name, nested=False):

        for name, value in metrics.items():
            mlf.log_metric(name, value)

        for name, value in parameters.items():
            mlf.log_param(name, value)

        for name, value in rest_params.items():
            mlf.log_param(name, value)

        for name, value in features.items():
            mlf.log_param(name, value)

        print(f'Logging successful')


def mlflow_dict(df_summary):
    columns = df_summary.columns.values
    metrics = ['Acc_mean', 'F1_mean', 'Prec_mean']
    rest_param = ['model_type']
    metrics_dict = {}
    parameter_dict = {}
    rest_param_dict = {}
    feature_dict = {}

    for metric in metrics:
        value = float(df_summary[metric])
        if metric not in metrics_dict:
            metrics_dict[metric] = value
    print('Metrics successful')

    if 'features' in columns:
        features = df_summary['features'].values[0].split(' - ')
        feature_dict = {features[i]: features[i] for i in range(0, len(features))}
    print('Features successful')

    if 'parameters' in columns:
        parameter_dict = json.loads(df_summary.parameters.iloc[0])
    print('Parameters successful')

    for param in rest_param:
        value = df_summary[param].values[0]
        if param not in rest_param_dict:
            rest_param_dict[param] = value
    print('Rest_Params successful')

    return (metrics_dict, parameter_dict, rest_param_dict, feature_dict)