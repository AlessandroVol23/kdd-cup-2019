# Issue #75: Setup MLflow

## Step 1 - Virtual Machine

By default, the MLflow Python API logs runs locally to files in an mlruns directory wherever the program is run. 
To log runs remotely, set up a tracking server on the VM consisting of two components: a backend store and an artifact store.

1. Install backend store: MySQL, Microsoft SQL Server, SQLite or PostgreSQL
2. Install artifact store: Amazon S3, Azure Blob Storage, Google Cloud Storage, FTP server, SFTP Server, NFS or HDFS

`Our combination of tracking server: SFTP & MySQL`

1. Install Python
2. Install MLflow
3. As default port of MLflow is 5000 use Nginx to set up a reverse proxy
4. `$ mlflow server --host 0.0.0.0 --artifact-default-root sftp://<user>@<server_ip>/<path_to_ftp_directory> 
--backend-store-uri <sql_uri>`
5. Check if Server is available via browser 

## Step 2 - Locale Machine

In order for all to use the module mlflow.py install

1. MLflow `pip install mlflow`
2. PySftp `pip install pysftp`
3. Paramiko `pip install paramiko`


## Step 3 - Prepare Model

`logmlflow.py` contains two functions:  `mlflow_dict()` and `mlflow_log()`

- `mlflow_dict(df_summary)`: In case results where tracked locally already this function prepares all data to fit requirements of `mlflow_log()`. 
- `mlflow_log(model, metrics, parameters, rest_params, features)`: This function logs all results to the tracking server. Following inputs are required
    - model : `0 (Default), 1 (LightGBM), 2 (Multiclass), 3 (TF-Ranking)`
    - metrics : All metrics (e.g. F1, CVS etc.) to be logged as a dictionary
    - parameters : All parameters to be logged as a dictionary
    - rest_params : Additional parameters to be logged as a dictionary
    - features : All features to be logged as a dictionary

#### Sample Code
```
summary = pd.read_csv("../data/summary/Summary4.csv")

metrics_dict, parameter_dict, rest_param_dict, feature_dict = x.mlflow_dict(summary)
x.mlflow_log(2, metrics_dict, parameter_dict, rest_param_dict, feature_dict)
```

# FAQ

### Tracking
MLflow Tracking is organized around the concept of runs, which are executions of some piece of data science code.

Each run records the following information:

- **Code Version:** Git commit hash used for the run, if it was run from an MLflow Project.
- **Start & End Time:** Start and end time of the run
- **Source:** Name of the file to launch the run, or the project name and entry point for the run if run from an MLflow Project.
- **Parameters:** Key-value input parameters of your choice. Both keys and values are strings.
- **Metrics:** Key-value metrics where the value is numeric. Each metric can be updated throughout the course of the run (for example, to track how your model’s loss function is converging), and MLflow records and lets you visualize the metric’s full history.
- **Artifacts:** Output files in any format. For example, you can record images (for example, PNGs), models (for example, a pickled scikit-learn model), or even data files (for example, a Parquet file) as artifacts.

## MLflow UI 
View at http://localhost:5000 with `mlflow ui` or on tracking URI. Refresh to get updates. 
In case of running mlflow ui locally with an error following might help.
```
ps ax|grep unicorn
kill <id>
```