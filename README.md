kdd-cup-2019
==============================
This is the repository for the Big Data Science practical course @ LMU.

## Raw preprocessing

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

## External features preprocessing

This adds the external features:

* Distance to closest subway
* Time features
* Public holiday
* Weather features

```shell
python ./src/features/add_external.py
```

# Fitting Models
All Scripts to fit Models are in "src/models". 
Corresponding subfolders contain the script to fit the corresponding model. 
All Scripts are documented, please look into the files itself to see how to run it!
Basically they have the following structure:
    [1] Load Packages
    [2] Define Functions
    [3] main() --> call all Functions!

# Project description

A short description of the project:
KDD Cup is with Baiduu (Chinese Google) and their Maps-App.
Goal (1) is to predict the transportation mode a user is going to click on
Goal (2) is an open use case on how to use the provided data in a meaningful way 

The Data provided is split into train and testset.
Basically to each Session [Query of a User on how to get from A -> B] we have a spartial feature [Coordinates of the Origin & Destiny],
the transportaion mode specific features [distance, time, price] and a User-Specific Personal-ID [63k Categories incl. NAs]

Project Organization
------------

    ├── LICENSE
    ├── .dvc               <- Folder for the Data Version Control
    ├── .git               <- Folder for all the GIT Settings [including more subfolders]
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- folder that holds all data used in this project
    │   │   
    │   ├── raw            <- The original, immutable data dump.
    │   │   ├─── data_set_phase1        <- the original, untouched data from phase 1
    │   │   │                              [multiple DFs, that are merged to one big DF (s. 'Processed')]
    │   │   │    
    │   │   └─── data_set_phase1_joined <- raw data that was joined on the Session IDs, so we have all
    │   │                                  data of single session in a single row 
    │   │
    │   ├── external       <- Data from third party sources, that were not from the KDD-DataCompetition itself
    │   │   ├─── districts              <- dataframes for districts assignments
    │   │   │
    │   │   └─── external_features      <- dataframes with external Informations
    │   │                                  [e.g. coordinates of subway stations, national holidays, ...]
    │   │                       
    │   ├── processed      <- Final data sets for modeling!
    │   │   ├─── split_test_train       <- SIDs used for splitting the data for CV
    │   │   │                              [has subfolders like: '5fold', '10fold', ... (corresponding to the CV-Tactic)]
    │   │   │
    │   │   ├─── features               <- contains "multiclass_1.pickle",..., "multiclass_3.pickle" 
    │   │   │                              w/ the different feature names, we use for modeling [which features]
    │   │   │
    │   │   ├─── ranking                <- all Testing and Training DFs that can be used for Ranking Models 
    │   │   │                              [TFRanking, LamdaRank,...]
    │   │   │      
    │   │   └─── multiclass             <- all Testing and Training DFs that can be used for Multiclass Models
    │   │                                  [Difference: first_all / last_all --> 
    │   │                                  sometimes the same transmode is recommenden twice: either keep the first or the last]
    │   │  
    │   └── interim        <- Intermediate data that has been transformed.
    │
    ├── docs               <- Documentationes
    │
    ├── environments       <- Folder for all different conda environments [all saved as '.yml' file]
    │    
    ├── models             <- Trained models | model predictions | model summaries
    │   │                     Structure as follows:
    │   │                       - 1. FOLDER: Category of the model (mutliclass, ranking, stacking,...)
    │   │                       - 2. Folder: Name of the feature space we have used (1, 2, 3, ...)
    │   │                       - 3. folder: Name of the concrete Model (KNN, MLP, XGboost)
    │   │                                    --> this folder can contain:   - summary files (F1, Confusion Matrix,...)
    │   │                                                                   - CV_predicitons (Predicted Class Probabilites on CV)
    │   │                                                                                     [used for stacking]
    │   │                                                                   - final_mod (model that were trained on all TrainData )
    │   │                                                                     we use for prediciting on the testset!
    │   │                                                                     [name corresponds to summary- / CV_predicitons-file!]
    │   │  
    │   ├── multiclass     <- all trained models, CV Predicitons and summaries for the Multiclass Approaches
    │   │   ├─── 1         <- all models that were create with features names in data/processed/features/multiclass_1.pickle
    │   │   │                 [contains subfolders with names of mutliclass learners]
    │   │
    │   ├── stacking       <- all trained models, CV Predicitons and summaries for the stacked Approach
    │   │
    │   └── ranking        <- all trained models, CV Predicitons and summaries for the Ranking Approaches
    │       ├─── 1         <- all models that were create with features names in data/processed/features/multiclass_1.pickle
    │       │                 [contains subfolders with names of mutliclass learners]
    │
    ├── notebooks          <- Jupyter notebooks [just explorative & experimental]
    |                         Naming convention: number (for ordering), the creator's initials and a short description!
    |                         e.g. '0.4-DS-data-exploration'.                         
    |
    ├── submissions        <- submission files as .csv [SID, y_hat]
    |                         naming convention: Name of model & date e.g. 'lamdarank_04_06'
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures           <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download data or generate data
    │   │
    │   ├── evaluator      <- Script to get the F1 Score of a Submission File
    │   │
    │   ├── features       <- Scripts to add (external) features & raw data 
    │   │   ├── add_features.py
    │   │   └── build_features.py
    │   │
    │   ├── MLFlow         <- Script to log ModelResults w/ MLFlow
    │   │
    │   ├── models         <- Scripts to train models and to use trained models to make
    │   │                     predictions [each mpodel type an own subfolder]
    │   │                     [documentaion see the files itself!]
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
