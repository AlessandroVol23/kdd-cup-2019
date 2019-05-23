# Issue #16: MLflow

# Step 1
## Setup MLflow

1. Install `pip install mlflow`
2. Import  `import mlflow`
3. In your code, after you've defined your parameters create a run `with mlflow.start_run():`
4. Log your parameters with 
```
for name, value in PARAMS.items():
        mlflow.log_param(name, value)
```
5. Following your code to train the model, log your metrics `mlflow.log_metric("f1", f1)`
6. Log your model to ensure all important info is saved `mlflow.log_artifact('model.h5')`
7. Change directory to data `cd ./kdd-cup-2019/data`
8. Run your training
9. View results with `mlflow ui`


### Sample Code
```
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import mlflow

mlrun_path = mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = mlflow.set_experiment("LightGBM/Multiclass/Tensorflow")

# Read the csv file
data = pd.read_csv("data.csv")

# Your parameters
PARAMS = {
    'distance': 5,
    'weather': 256,
    'id': 0.1,
    'weekend': False,
    'weekday': True,
}

# experiment_id "1" for LightGBM, "2" for Multiclass, "3" for TF
with mlflow.start_run(experiment_id=1, run_name="optional info about run"):
    
    for name, value in PARAMS.items():
        mlflow.log_param(name, value)
    
    
    ### Your code to train the model

    f1 = model.evaluate(x_test, y_test, verbose=2)
    mlflow.log_metric("f1", f1)
    
    model.save('model.h5')
    mlflow.log_artifact('model.h5')
```

#Step 2
## Setup Neptune
...


# FAQ
##MLflow
MLflow is an open source platform for managing the end-to-end machine learning lifecycle. But
- "MLflow focuses on tracking, reproducibility, and deployment, not on organization and collaboration."
- „For sharing / accessing information about runs across machines, running a tracking server is recommended over data version.“

Therefor we will use Neptune MLflow to integrate that feature.

## Tracking
MLflow Tracking is organized around the concept of runs, which are executions of some piece of data science code.

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

## MLflow UI 
View at http://localhost:5000 with `mlflow ui`. 
Refresh Browser to get update. In case of error run and kill processes with unicorn. Try to reach MLflow UI again.
```
ps ax|grep unicorn
kill <id>
```