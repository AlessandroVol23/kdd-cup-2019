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

For the initial raw preprocessing with features (no external), run this file in `../kdd-cup-2019`. Choose `first` or `last` depending on which transport mode you prefer to pick, the one displayed first in the plan list or last.

```shell./src/data/make_raw_preprocessing.py /path/to/kdd-cup-2019/data 'first'
python 
```

The two dataframes are stored in `../data/processed_raw/`

* train pickle has 500000 x 114 dimensions
* test pickle has 94358 x 112 dimensions

Train pickle has `click_time` and `click_mode` as additional columns.

### time_features

```python
from src.features.build_features import time_features

df_train = time_features(df_train, type='req')
df_test = time_features(df_test, type='req')
```

### add_public_holidays

```python
from src.features.build_features import add_public_holidays

df_train = add_public_holidays(df_train)
df_test = add_public_holidays(df_test)
```

### add_dist_nearest_subway

```python
from src.features.build_features import add_dist_nearest_subway

df_train = add_dist_nearest_subway(df_train)
df_test = add_dist_nearest_subway(df_test)
```

## Models

### How to save a model?

```python
# Add sys to sys path to import custom modules from source
import sys
sys.path.append("../src/")
<<<<<<< README.md
```

# Import custome function save model
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
    ├── .dvc               <- Folder for the Data Version Control
    ├── .git               <- Folder for all the GIT Settings [including more subfolders]
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- folder that holds all data used in this project
    │   │   
    │   ├── raw            <- The original, immutable data dump.
    │   │   ├─── data_set_phase1        <- the original, untouched data from phase 1
    │   │   └─── data_set_phase1_joined <- the original train- & testdata merged into a single DF, 
    │   │                                  without any external features, but with internal created features
    │   │
    │   ├── external       <- Data from third party sources, that were not from the datacompetition itself
    │   │   ├─── KDD_extern_processed   <- DF from other repositiories used for feature engineering inspiration
    |   |   └─── external_features      <- dataframe to add external features [subway stations, national holidays...]
    |   |
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   │   ├─── Test_Train_Splits      <- SIDs used for splitting the data for CV [5fold, 10fold, ...]
    │   │   ├─── Ranking                <- all dataframes that can be used for training Ranking Models [TFRanking, LamdaRank,...]
    |   |   └─── multiclass             <- dataframes to train multiclass classifier
    |   |                                  [also contains subfolders w/ names of the special preprocessing applied]
    │   │  
    │   └── interim        <- Intermediate data that has been transformed.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── environments       <- Folder for all different conda environments e.g. for TFRanking, Light-GBM, Multiclass Approach...
    │                         [all saved as '.yml' file]
    │    
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   │                     --> each folder w/ subfolder of the name with the DF used to do the CV
    │   │                     [summary.csv-files; predictions.csv-files; finalmodel.pickle-files]
    │   │  
    │   ├── lgb_multi      <- all trained models, CV Predicitons and summaries for the Light-GBM MulticlassApproach
    │   ├── multiclass     <- all trained models, CV Predicitons and summaries for the Multiclass Approaches
    │   ├── stacking       <- all trained models, CV Predicitons and summaries for the stacked Approach
    │   └── ranking        <- all trained models, CV Predicitons and summaries for the Ranking Approaches
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering), the creator's initials
    │                         and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`.
    |                         [just explorative & experimental]
    |
    ├── submissions        <- submission files as .csv [SID, y_hat]
    |                         naming convention: Name of model & date e.g. 'lamdarank_04_06'
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   │ 
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to add (external) features  to merged_raw_data 
    │   │   ├── add_features.py
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and to use trained models to make
    │   │                     predictions [each mpodel type an own subfolder]
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
