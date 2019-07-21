kdd-cup-2019
==============================
This is the repository for the Big Data Science practical course @ LMU. The aim of the course was to attend at the KDD Cup this year which was hosted from Baidu.

# Project description

A short description of the project:
This year's KDD Cup is in cooperation with Baidu. Baidu is a chinese search engine (comparable to Google) and also offers a Maps-Service.
The Challenge has 2 tasks in total:
[1] for queries on Baidu maps, we shall model the relevance of the different offered transportation modes and predict the one, that the user will click on!
[2] is an open use case on how to use the provided data in a meaningful way! 

The Data provided is already split into train and test set.
Basically to each Session [Query of a User on how to get from A -> B] we have a spatial feature [Coordinates of the Origin & Destiny],
the transportaion mode specific features [distance, time, price] and user specific personal IDs [63k categories incl. NAs]

__Small note on data__:

Since, we've hosted all datasets on an internal DVC (data version control) system you have to download first all datasets and then execute the preprocessing scripts to get all datasets. Put all the raw user data into `data/raw`.
You can find all data sources here:

__Data Sources__:
* User data: https://dianshi.baidu.com/competition/29
* Weather: https://github.com/yaoxuefeng6/Paddle_baseline_KDD2019
* Holidays: https://publicholidays.cn/2018-dates/
* Subway stations: https://en.wikipedia.org/wiki/List_of_Beijing_Subway_stations

# Preprocessing 

## [1] Raw preprocessing
For the initial raw preprocessing, run the file `src/data/raw_features`. 
This will merge all 4 different raw DFs to a single one and will extract the first basic features!

* Choose `row` or `col` depending on what model you want to train [multiclass approach | ranking approach]
* Choose `first` or `last` depending on which transport mode you prefer to pick, when the same transport mode is offered multiple times. The one displayed first in the plan list or the one displayed last.

This adds the 'raw' features:
* Coordinates
* Profiles
* Targets
* Unstack of plans
 
``` bash
python ./src/data/raw_features.py <PATH_TO_KDD_CUP_2019/DATA> 'col' 'first'
```

## [2] External features preprocessing
This adds the external features on the merged DFs (based on [1]):

* Distance to closest subway
* Time features
* Public holiday
* Weather features

```shell
python ./src/features/external_features.py /path/to/kdd-cup-2019/data
```

## [3] Split SIDs for k-fold CV
--> sample SIDs from Trainset, split them into k evenly sized chunks and return a list with k sublists containing the SIDs for each fold!
If you do not want to have5 folds, open the script and change the "amount_folds" Parameter to any integer > 1.
Results will be saved in "data/processed/split_test_train/k-fold/SIDs.pickle"

```shell
python ./src/features/split_SIDs_for_CV.py
```

## [4] Fitting Models
All Scripts to fit Models are in the corresponding subfolders in "src/models".
All Scripts are documented, please look into the files itself to adjust parameters, select the models types etc.
Basically they have the following structure:
    [1] Load Packages
    [2] Define Functions
    [3] main() --> call all Functions!

### [4-1] LGBM Multiclass
Fit an LGBM Multiclass Model to the "processed/multiclass"-data.

### [4-2] LGBM Ranking
Fit an LGBM Lamda Rank Model to the "processed/ranking"-data.

### [4-3] Multivariate Models
Fit any multiclass model from the sklearn library on the "processed/multiclass"-data.

### [4-4] Stacking Models
Stack the Predicitons of the submodels to create a metalearner!

### [4-5] TFR
Fit an Tensorflow Ranking Model on the "processed/ranking"-data.
Parameters are set in the "#%% Set the global params" Section!

Project Organization
------------

    ├── LICENSE
    ├── .dvc               <- Folder for the Data Version Control
    ├── .git               <- Folder for all the GIT Settings [including more subfolders]
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- folder that holds all data used in this project
    │   │   
    │   ├── raw            <- The original, immutable data dump.
    │   │   │  
    │   │   ├─── data_set_phase1        <- contains all the original raw data that was published by the KDD Challenge
    │   │   │                              [multiple DFs, that are merged to one big DF ('./src/data/make_raw_preprocessing.py')]
    │   │   │    
    │   │   └─── data_set_phase1_joined <- all the raw data joined to one big DF - for ranking or multiclass approaches in the modelling
    │   │                                  for details see: ('./src/data/make_raw_preprocessing.py')
    │   │
    │   ├── external       <- Data from third party sources, that were not from the KDD-DataCompetition itself
    │   │   │    
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
    │       ├─── LGBM      <- all LGBM Models [contains subfolders with names of mutliclass learners]
    │       └─── TFR       <- all TFR summaries, predicitons etc... [no subfolders!]
    │
    ├── notebooks          <- Jupyter notebooks [just explorative & experimental]
    |                         Naming convention: number (for ordering), the creator's initials and a short description!
    |                         e.g. '0.4-DS-data-exploration'.                         
    |
    ├── submissions        <- submission files as .csv [SID, y_hat]
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
    │   │                     For describtions in more detail, see the 'Order of Scripts' Section - Subsection [4]                   
    │   │
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to merge data, add (external) features and create the splits for test and train
    │   │
    │   ├── evaluator      <- Script to get the F1 Score of a Submission File
    │   │
    │   ├── features       <- Scripts with the functions neeeded for the feature extraction 
    │   │   │                 [needed for the data extraction in the "src/data"]
    │   │   ├── add_features.py
    │   │   └── build_features.py
    │   │
    │   ├── MLFlow         <- Script to log ModelResults w/ MLFlow
    │   │
    │   ├── models         <- Scripts to train models and to use trained models to make
    │   │                     predictions [each mpodel type an own subfolder]
    │   │                     [documentaion see the files itself OR  the 'Order of Scripts' Section - Subsection [4]]
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
--------