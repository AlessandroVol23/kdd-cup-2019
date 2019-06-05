import pandas as pd
import numpy as np
import sklearn
import lightgbm as lgb
import os
import click
from pathlib import Path

'''
load the kdd cup data

Parameters
----------
    path : str
        path to the pickle file

Returns
-------
    data frame
        with the data
'''
def load_data(path):

    # path to raw pickle_file
    print("load_data...")
    data = pd.read_pickle(path)
    # print(list(data.columns.values))
    # print(data.head(10))

    # sort data to sid for mapping to the query file
    print(" sort values...")
    data.sort_values(by="sid", inplace=True)
    # print(data.head(10))

    # drop same transport_mode in every session (sid)
    print(" drop duplicates...")
    data.drop_duplicates(["sid", "transport_mode"], inplace=True)
    # print(data.head(10))

    # reindexing
    print(" reindexing...")
    data.reset_index(drop=True, inplace=True)
    # print(data.head(10))

    # assign new column target to the data
    print(" set target for lambdarank...")
    if "click_mode" in data:
        data = data.assign(target=data.apply(lambda x: 3 if x.click_mode == x.transport_mode else 0, axis=1))
        # print(data.head(10))

    return data

'''
create a query file for lambda rank with LightGBM

Parameters
----------
    df_r : dataset
        data set for lambda ranking

Returns
-------
    numpy array 
        with counts of rows are assign to one sid
'''
def create_query_file(df_r):
    print("create query file ...")
    df = df_r.copy()
    df = df.sort_values("sid")
    df = df.reset_index(drop=True)
    df = df.assign(row=df.index)
    df.row = df.row + 1
    sid_rows = pd.DataFrame(df.groupby("sid").last()["row"])
    sid_diff = sid_rows.assign(difference=sid_rows.diff())
    sid_diff.iloc[0, 1] = sid_diff.iloc[0, 0]
    query_file = sid_diff.difference.values
    return query_file
'''
get the max value from an array interval

Parameters
----------
    array : array
        the whole array

    array_from : int
        begin - index

    array_to : int
        end - index-1

Returns
-------
    value 
        max value from the given interval in the array

'''
def get_max_from_diff(array, array_from, array_to):
    array_diff = np.array(array[array_from:array_to])

    # tmp = np.array(y_predict[0:4])
    # print(tmp)
    # print(tmp[np.argmax(tmp)])

    return array_diff[np.argmax(array_diff)]

'''
get the index from the max value from an array interval

Parameters
----------
    array : array
        the whole array

    array_from : int
        begin - index

    array_to : int
        end - index-1

Returns
-------
    value 
        index from the max value from the given interval in the array

'''
def get_max_from_diff_index(array, array_from, array_to):
    array_diff = np.array(array[array_from:array_to])

    # tmp = np.array(y_predict[0:4])
    # print(tmp)
    # print(tmp[np.argmax(tmp)])

    return np.argmax(array_diff)

'''
create a array wit the best transport mode within one sid

Parameters
----------
    x : all data
        the input data with all transport modes

    y_all_transport_mode : numpy array
        all possible rated transport_mode

    query_array : numpy array
        mapping for sid and transport_mode

Returns
-------
    y_best_transport_mode : numpy array 
        best possible transport mode for one sid

'''
def create_y_best_transport_mode(x, y_all_transport_mode, query_array):

    # init params
    start = 0
    end = 0

    y_best_transport_mode_list = []
    sid_list = []

    i = 0

    # walk through query
    for q in query_array:
        # set the end
        end = start + q

        # compute best click
        y_best_transport_mode_list.append(
            x.at[int(start + get_max_from_diff_index(y_all_transport_mode, int(start), int(end))), "transport_mode"])
        sid_list.append(x.at[int(start + get_max_from_diff_index(y_all_transport_mode, int(start), int(end))), "sid"])

        # set new start
        start = end

        i += 1

        # sccuss printing
        # if( end % 10000 == 0):
        #    print(" build sampled labels : " + str(int(end/y_predict.size*100))+" %")

    # finished
    # print(" build sampled labels : " + str(int(end/y_predict.size*100))+" %")
    # print()

    return pd.DataFrame(data={'sid': pd.Series(sid_list).values, 'y': pd.Series(y_best_transport_mode_list).values})

'''
do  train, test and evaluate with lightgbm standard settings

Parameters
----------
    x_train : pandas.df
        train data

    x_test : pandas.df
        test data

    y_train : pandas.df
        train labels

    y_test : pandas.df
        test labels

Returns
-------
    print the F1-score with average weighted if y_test hast column target

    return the y_predict_best_transport_mode

'''
def train_test_evaluate_lightgbm(x_train, x_test, y_train, y_test=None):
    # create query file

    print("create query_train ...")
    query_train = create_query_file(x_train)

    print("create query_test ...")
    query_test = create_query_file(x_test)

    print("select features for x_train ...")
    x_train_feature = x_train[
        ['distance_plan', 'eta', 'price', 'transport_mode', 'pid', 'distance_query', 'req_hour', 'req_weekend',
         'req_night', 'req_day', 'req_evening', 'is_holiday', 'transport_mode']]

    print("select features for x_test ...")
    x_test_feature = x_test[
        ['distance_plan', 'eta', 'price', 'transport_mode', 'pid', 'distance_query', 'req_hour', 'req_weekend',
         'req_night', 'req_day', 'req_evening', 'is_holiday', 'transport_mode']]

    # create training set
    print("create LightGBM dataset ...")
    lgb_train = lgb.Dataset(x_train_feature, y_train, group=query_train)

    # set params
    print("config param ...")
    params = {}
    # params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'lambdarank'

    # training
    print("LightGBM training ...")
    gbm = lgb.train(params, lgb_train)

    # predict
    print("LightGBM prediction ...")
    y_predict = gbm.predict(x_test_feature)

    print("compute y_predict_best_transport_mode ...")
    y_predict_best_transport_mode = create_y_best_transport_mode(x_test, y_predict, query_test)

    # if y_test == None:

    print("compute y_best_transport_mode ...")
    y_best_transport_mode = create_y_best_transport_mode(x_test, y_test, query_test)

    # evaluate
    print("compute f1 score :")
    print(sklearn.metrics.f1_score(y_best_transport_mode['y'], y_predict_best_transport_mode['y'], average='weighted'))

    print("finished")
    return y_predict_best_transport_mode

@click.command()
@click.argument("absolute_path_folder")
@click.argument("x_train")
def main(
        absolute_path_folder
):
    print("start lgbm_lr_bl.py ...")
    print("absolute_path_folder: " + str(absolute_path_folder))

    data = load_data(absolute_path_folder)

    print("loaded columns: ")
    print(data.columns.values)

main()
