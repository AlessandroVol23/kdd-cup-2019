kdd-cup-2019
==============================
This is the repository for the Big Data Science practical course @ LMU.

## Data Preprocessing

In ./src/data you'll find the ``make_dataset.py`` script. This script takes the files from the input parameter and creates one concatenated and joined file with which you can train your ML model. 

__1. Create conda environment__

In ``./environments/`` you'll find the yml file ``preprocessing_kdd.yml``. 

Create a new conda env with:

```shell
conda env create -f ./environments/preprocessing_kdd.yml
```

__2. Activate conda environment__

```shell
conda activate preprocessing_kdd
```

__3. Execute preprocessing script__

```shell
python ./src/data/make_dataset.py /path/to/kdd-cup-2019/data/raw/ /path/to/kdd-cup-2019/data/processed/YOUR_FILE.pickle TRAIN_OR_TEST
```

After executing the script you see some output, wait until it says "Preprocessing Done!"

## Features

You can add all features by running `../src/features/add_features.py`, or add them individually:

If you have library problems, install them:

```bash
pip install geopandas
```

### Raw preprocessing

For the initial raw preprocessing with features (no external), run this file in `../kdd-cup-2019`. 

* Choose `row` or `col` depending on what model you want to train
* Choose `first` or `last` depending on which transport mode you prefer to pick, the one displayed first in the plan list or last.

This adds the 'raw' features:

* Coordinates
* Profiles
* Targets
* Unstack of plans
* 

```shell
python ./src/data/make_raw_preprocessing.py /path/to/kdd-cup-2019/data 'col' 'first'
```

### External features preprocessing

This adds the external features:

* Distance to closest subway
* Time features
* Public holiday
* Weather features

```shell
python ./src/features/add_external.py
```

### Join

## Models

### How to save a model?

```python
# Add sys to sys path to import custom modules from source
import sys
sys.path.append("../src/")
<<<<<<< README.md
```

# Import custome function save model
```python
from src.models.utils import save_model
save_model(lgb_model, "../models/test_model.pickle")
# Be careful! The path varies of course.
# It is just allowed to save models in the models folder
```

### Light GBM Multiclass Baseline Model

The model is saved in `models/lgbmodel_2k.pickle`. You can load it with:

```python
# Add sys to sys path to import custom modules from source
import sys
sys.path.append("../src/")

# Import custom function load_model
from src.models.utils import load_model
lgb_model = load_model("../models/lgbmodel_2k.pickle")
```

## Project description

A short description of the project:

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    |   |   |___  15_to_one_df          <- preprocessed data for light gbm approach [via feature/#-15 preprocess data]
    |   |   |                              [no features added]
    |   |   |___  Multiclass_Approach   <- preprocessed data for multiclass approaches [37_BaselineModell_SVM]
    |   |   |                              [no features added]
    |   |   |___  Test_Train_Splits     <- Folders with the SID of the k-fold splits [5-fold, ...]
    |   |
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── environments       <- Folder for all different conda environments e.g. PyTorch, TFRanking, Light-GBM
    │    
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    |
    |___ submissions       <- submission files [SID, y_hat] as .csv
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
