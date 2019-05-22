# Issue #16: MLflow

MLflow is an open source platform for managing the end-to-end machine learning lifecycle.

### Installing

To install this package run:

```
pip install mlflow
```

## Tracking

MLflow Tracking is organized around the concept of runs, which are executions of some piece of data science code.

### Concepts

Each run records the following information:

- **Code Version:** Git commit hash used for the run, if it was run from an MLflow Project.
- **Start & End Time:** Start and end time of the run
- **Source:** Name of the file to launch the run, or the project name and entry point for the run if run from an MLflow Project.
- **Parameters:** Key-value input parameters of your choice. Both keys and values are strings.
- **Metrics:** Key-value metrics where the value is numeric. Each metric can be updated throughout the course of the run (for example, to track how your model’s loss function is converging), and MLflow records and lets you visualize the metric’s full history.
- **Artifacts:** Output files in any format. For example, you can record images (for example, PNGs), models (for example, a pickled scikit-learn model), or even data files (for example, a Parquet file) as artifacts.

### Experiments

MLflow allows to group runs under experiments, which can be useful for comparing runs intended to tackle a particular task. 

Create experiments and pass it for an individual run

```
mlflow experiments create <name>
export MLFLOW_EXPERIMENT_ID = <id>
```
View Tracking UI at http://localhost:5000 with

```
mlflow ui
```
Refresh Browser to get update. In case of error run and kill processes with unicorn. Try to reach MLflow UI again.
```
ps ax|grep unicorn
kill <id>
```
#Code Skeleton
Copy code below and adjust to your need. Usually following steps shall help

1.  Change experiment name to ```"LightGBM", "Multiclass" or "Tensorflow"```\
2.  Update code to read the data.csv\
3.  Replace with your code to split train and test set\
4.  Change experiment_id to ```"1" for LightGBM, "2" for Multiclass and "3" for TF```\
5.  Replace with your baseline\

```
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow.sklearn


mlrun_path = mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = mlflow.set_experiment("test")


# Read the csv file
music_data = pd.read_csv("music.csv")

# Split the data into training and test sets
X = music_data.drop(columns=["genre"])
y = music_data["genre"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# start run
with mlflow.start_run(experiment_id=4, run_name="test"):
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)

    # Log mlflow attributes for mlflow UI
    # Log parameter
    mlflow.log_param('predictions', predictions)

    # Log metric; metrics can be updated throughout the run
    mlflow.log_metric('score', score)
```
### Run & Store Results

1. Change directory to run mlflow `cd data`
2. Run the file. Experiments will be stored in ./kdd-cup-2019/data/mlruns/<experiment_id>
3. View results by calling `mlflow ui`
