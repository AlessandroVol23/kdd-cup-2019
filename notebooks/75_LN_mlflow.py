import pysftp
from paramiko import sftp
import mlflow as mlf

'''
model = 0 (Default), 1 (LightGBM), 2 (Multiclass), 3 (TF-Ranking)
features passed saved as dictionary 
F1 result of model
CVS of trained model
'''


def mlflow(model, features, f1, cvs):
    cnopts = pysftp.CnOpts()

    with pysftp.Connection('138.246.233.10', username='ubuntu', private_key='~/.ssh/id_rsa', port=22, cnopts=cnopts):
        mlf.set_tracking_uri('http://138.246.233.10:80')
        print('SFTP connection successful')

    experiment_id = model

    if model == 1:
        experiment_name = 'LightGBM'
    elif model == 2:
        experiment_name = 'Multiclass'
    elif model == 3:
        experiment_name = 'TF-Ranking'
    else:
        print('Model is missing!')

    mlf.set_experiment(experiment_name)

    with mlf.start_run(experiment_id=experiment_id, run_name=experiment_name, nested=False):

        for name, value in features.items():
            mlf.log_param(name, value)

        mlf.log_metric("f1", f1)
        mlf.log_metric("cvs", cvs)

        print(f'Logging successful')


params = {
         'distance': 10,
         'weather': 300,
         'id': 0.6,
         'weekend': True,
         'weekday': False,}
f1 = 0.695
cvs = 0.613

mlflow(3, params, f1, cvs)