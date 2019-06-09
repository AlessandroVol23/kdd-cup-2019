import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import sys
import click
from time import gmtime, strftime

print(os.getcwd())

import os
os.chdir(os.path.dirname(__file__))
print(os.getcwd())
sys.path.append("../")

from utils import save_model

os.chdir('/home/sandro/repo/kdd-cup-2019/')

import pickle
from sklearn.metrics import f1_score

    
split_1 = pd.read_pickle('data/interim/splits/SIDs_1.txt')
split_2 = pd.read_pickle('data/interim/splits/SIDs_2.txt')
split_3 = pd.read_pickle('data/interim/splits/SIDs_3.txt')
split_4 = pd.read_pickle('data/interim/splits/SIDs_4.txt')
split_5 = pd.read_pickle('data/interim/splits/SIDs_5.txt')
SIDs = [split_1, split_2, split_3, split_4, split_5]

submit = pd.read_pickle("data/interim/submit.pickle")

def submit_result(submit, result, model_name):
    print('Saving submit...')
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    submit['recommend_mode'] = result
    submit.to_csv(
        'submissions/{}_result_{}.csv'.format(model_name, now_time), index=False)

    
    print('Saved submit at {}'.format(model_name))
    
def save_preds(sids, preds, path):
    df = pd.DataFrame(preds, index=sids, columns=['p0','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11'])
    df.to_csv(path)
    

def downsample(df, mode, amount):
    df_just_mode =df.loc[df.click_mode == mode]  
    
    df_mode_target = df_just_mode.sample(amount, replace=True)
    df = df.loc[df.click_mode != mode]
    df = pd.concat([df_mode_target, df], axis=0)
    
    return df

def upsample(df, mode, amount):
    df_just_mode =df.loc[df.click_mode == mode]  
    
    df_mode_target = df_just_mode.sample(amount, replace=True)
    df = df.loc[df.click_mode != mode]
    df = pd.concat([df_mode_target, df], axis=0)
    return df
    
def eval_f(y_pred, train_data):
    y_true = train_data.label
    y_pred = y_pred.reshape((12, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True


def lgbm_train(df_train, test_x, model_name, features, lr, num_leaves, downsample_mode, downsample_amount,
         upsample_mode_1, upsample_1_amount,  upsample_mode_2, upsample_2_amount, upsample_mode_3, upsample_3_amount,
              upsample_mode_4, upsample_4_amount, upsample_mode_5, upsample_5_amount):
    print("Start to train light gbm model")
    print("df train shape", df_train.shape)

    data = df_train.copy()

    lgb_paras = {
        'objective': 'multiclass',
        'metrics': 'multiclass',
        'learning_rate':  lr,
        'num_leaves': num_leaves,
        'lambda_l1': 0.01,
        'lambda_l2': 10,
        'num_class': 12,
        'seed': 2019,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4
    }
    count = 1
    scores = []
    result_proba = []

    # Loop over the different Test/Train Splits
    for i in range(len(SIDs)):
        
        
        # Print process
        print(str(i) + " / " + str(len(SIDs) - 1))

        # Extract the Test_Set based on the current SID:
        val_x          = data.loc[data["sid"].isin(SIDs[i]), features].values
        val_y = data.loc[data["sid"].isin(SIDs[i]), "click_mode"].values

        # Extract the SIDs we use for training, and select correponding train points!
        train_sids = []
        for j in range(len(SIDs)):
            if j != i:
                train_sids = train_sids + SIDs[j]
                
                
        df_train_split = data.loc[data["sid"].isin(train_sids), :]
        
        if downsample_mode != 99:
            print("Downsample mode {} on {}".format(downsample_mode, downsample_amount))
            df_train_split = downsample(df_train_split, downsample_mode, downsample_amount)
        else:
            print("Downsample mode is 99")
        
        if upsample_mode_1 != 99:
            print("Upsample mode 1 is {} on {}".format(upsample_mode_1, upsample_1_amount))
            df_train_split = upsample(df_train_split, upsample_mode_1, upsample_1_amount)
        else:
            print("Upsample mode 1 is 99")
            
        if upsample_mode_2 != 99:
            print("upsample_mode_2  is {} on {}".format(upsample_mode_2, upsample_2_amount))
            df_train_split = upsample(df_train_split, upsample_mode_2, upsample_2_amount)
        else:
            print("Upsample mode 2 is 99")
            
        if upsample_mode_3 != 99:
            print("upsample_mode_3  is {} on {}".format(upsample_mode_3, upsample_3_amount))
            df_train_split = upsample(df_train_split, upsample_mode_3, upsample_3_amount)
        else:
            print("Upsample mode 3 is 99")
            
        if upsample_mode_4 != 99:
            print("upsample_mode_4  is {} on {}".format(upsample_mode_4, upsample_4_amount))
            df_train_split = upsample(df_train_split, upsample_mode_4, upsample_4_amount)
        else:
            print("Upsample mode 4 is 99")
        
        if upsample_mode_5 != 99:
            print("upsample_mode_5  is {} on {}".format(upsample_mode_5, upsample_5_amount))
            df_train_split = upsample(df_train_split, upsample_mode_5, upsample_5_amount)
        else:
            print("Upsample mode  is 99")
            
        print('Current Splits')
        print(df_train_split.groupby('click_mode').count()['sid'])
     
             
        tr_x = df_train_split[features].values
        tr_y = df_train_split['click_mode'].values

        train_set = lgb.Dataset(tr_x, tr_y)
        val_set = lgb.Dataset(val_x, val_y)
        # Train on this split
        lgb_model = lgb.train(lgb_paras, train_set,
                              valid_sets=[val_set], early_stopping_rounds=50,
                              num_boost_round=40000, verbose_eval=50, feval=eval_f)

        # Predict on best iteration of this split with validation set
        val_probs = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
        val_pred = np.argmax(val_probs, axis=1)

        # F1 val score
        val_score = f1_score(val_y, val_pred, average='weighted')

        # Predict with test set on best iteration of this split
        result_proba.append(lgb_model.predict(
            test_x, num_iteration=lgb_model.best_iteration))
        scores.append(val_score)
        save_model(lgb_model, os.path.join(os.getcwd(), 'models', 'split_' + str(count) + '_' + model_name))
        
        save_preds(SIDs[i], val_probs, 'models/test/split_' + str(count))
        count += 1
        
    print('cv f1-score: ', np.mean(scores))
    pred_test = np.argmax(np.mean(result_proba, axis=0), axis=1)
    save_model(lgb_model, os.path.join(os.getcwd(), 'models', 'final_' + model_name))
    return pred_test

@click.command()
@click.argument("path_train", required=True)
@click.argument("path_test", required=True)
@click.argument("path_feature_file", required=True)
@click.argument("name", required=True)
@click.argument("lr", required=False, default=0.05)
@click.argument("num_leaves", required=False, default=60)
@click.argument("downsample_mode", required=False, default=99)
@click.argument("downsample_amount", required=False, default=99)
@click.argument("upsample_mode_1", required=False, default=99)
@click.argument("upsample_1_amount", required=False, default=99)
@click.argument("upsample_mode_2", required=False, default=99)
@click.argument("upsample_2_amount", required=False, default=99)
@click.argument("upsample_mode_3", required=False, default=99)
@click.argument("upsample_3_amount", required=False, default=99)
@click.argument("upsample_mode_4", required=False, default=99)
@click.argument("upsample_4_amount", required=False, default=99)
@click.argument("upsample_mode_5", required=False, default=99)
@click.argument("upsample_5_amount", required=False, default=99)
def main(path_train, path_test, path_feature_file, name, lr=0.05, num_leaves=60, downsample_mode=99, downsample_amount=99,
         upsample_mode_1=99, upsample_1_amount=99, upsample_mode_2=99, upsample_2_amount=99,
        upsample_mode_3=99, upsample_3_amount=99, upsample_mode_4=99, upsample_4_amount=99, upsample_mode_5=99, upsample_5_amount=99):
    """
        path_train: Path to train file 
        path_test: Path to test file
        name: Name how the model and submit should be saved (model folder has to exist)
        downsample_mode: Downsample which mode
        downsample_amount: Downsample on target amount
        upsample_mode_1: Upsample which mode
        upsample_1_amount: Upsample amount
        upsample_mode_2
        upsample_2_amount
        If you don't want to use one mode put in 99 as a number and it'll be ignored
    """
    print("Start Main")
    print("Downsample mode: {} on amount: {}".format(downsample_mode, downsample_amount))
    print("Upsample mode 1: {} on amount: {}".format(upsample_mode_1, upsample_1_amount))
    print("Upsample mode 2: {} on amount: {}".format(upsample_mode_2, upsample_2_amount))
    print("Upsample mode 3: {} on amount: {}".format(upsample_mode_3, upsample_3_amount))
    print("Upsample mode 4: {} on amount: {}".format(upsample_mode_4, upsample_4_amount))
    print("Upsample mode 5: {} on amount: {}".format(upsample_mode_5, upsample_5_amount))
    
    df_train = pd.read_pickle(path_train)
    print("Loaded df_train with shape: ", df_train.shape)
    with open (path_feature_file, 'rb') as fp:
        features = pickle.load(fp)
    df_test = pd.read_pickle(path_test)
    test_x = df_test[features].values
    
    lr = float(lr)
    num_leaves = int(num_leaves)
    downsample_mode = int(downsample_mode)
    downsample_amount = int(downsample_amount)
    upsample_mode_1 = int(upsample_mode_1)
    upsample_1_amount = int(upsample_1_amount)
    upsample_mode_2 = int(upsample_mode_2)
    upsample_2_amount = int(upsample_2_amount)
    upsample_mode_3 = int(upsample_mode_3)
    upsample_3_amount = int(upsample_3_amount)
    upsample_mode_4 = int(upsample_mode_4)
    upsample_4_amount = int(upsample_4_amount)
    upsample_mode_5 = int(upsample_mode_5)
    upsample_5_amount = int(upsample_5_amount)
    
    preds = lgbm_train(df_train, test_x, name, features, lr, num_leaves, downsample_mode, downsample_amount,
         upsample_mode_1, upsample_1_amount,  upsample_mode_2, upsample_2_amount, upsample_mode_3, upsample_3_amount,
                      upsample_mode_4, upsample_4_amount, upsample_mode_5, upsample_5_amount) 
    submit_result(submit, preds, name)
               
        
    
if __name__ == "__main__":
    main()