import os
import sys
import json
import math
import click
import logging

import pandas as pd
import numpy as np

from pandas.io.json import json_normalize


def read_in_data(absolute_raw_data_path):
    """
        This function reads in all the data sets.
        returns: df_train_queries, df_train_plans, df_train_clicks, df_test_queries, df_test_queries, df_test_plans, df_profiles
    """

    data_set_path = os.path.join(absolute_raw_data_path, 'raw/data_set_phase1')

    df_profiles = pd.read_csv(os.path.join(data_set_path, "profiles.csv"))
    df_train_clicks = pd.read_csv(os.path.join(data_set_path, "train_clicks.csv"))
    df_train_plans = pd.read_csv(os.path.join(data_set_path, "train_plans.csv"))
    df_train_queries = pd.read_csv(os.path.join(data_set_path, "train_queries.csv"))

    df_test_queries = pd.read_csv(os.path.join(data_set_path, "test_queries.csv"))
    df_test_plans = pd.read_csv(os.path.join(data_set_path, "test_plans.csv"))
    return (df_profiles, df_train_queries, df_train_plans, df_train_clicks, df_test_queries, df_test_plans)


def write_data(absolute_raw_data_path, df, train_mode, df_mode, plan_mode='col'):
    if df_mode == 'row':
        filename = 'processed_raw/' + train_mode + '_raw_' + df_mode + '.pickle'
    else:
        filename = 'processed_raw/' + train_mode + '_raw_' + plan_mode + '.pickle'
    print("Writing df to pickle in ../data/processed_raw/")
    df.to_pickle(os.path.join(absolute_raw_data_path, filename))
    return


def raw_preprocessing(df, plandf, profiledf, clickdf=None, df_mode='col', plan_mode='first'):

    """ 
    Function to construct dataset with raw features:
    * sid, pid, req_time, o_lat, o_long, d_lat, d_long
    * 3 columns per plan mode, dist_X, price_X, eta_x
    Function looks in absolut_path_raw_folder for the data folder in the project.
    absolute_path_data_folder: Path to raw data folder. Typically /home/xxx/repo/data/raw
    plan_mode: 'first' or 'last'. Sometimes there are several suggestions for the same transport mode. 'first' mode takes the first suggestion, 'last' takes only the last suggestion.
    """

    def preprocess_coordinates(df):
        '''
        Generates following 5 columns
        * o_long
        * o_lat
        * d_long
        * d_lat

        And deletes these:
        * o
        * d

        +2 columns
        '''
        df[["o_long", "o_lat"]] = df.o.str.split(",", 1, expand=True).astype(float)
        df.drop("o", axis=1, inplace=True)

        df[["d_long", "d_lat"]] = df.d.str.split(",", 1, expand=True).astype(float)
        df.drop("d", axis=1, inplace=True)

        df['distance_query'] = df.apply(lambda x: (math.sqrt((x.o_long - x.d_long)**2 + (x.o_lat - x.d_lat)**2)), axis=1)

        return df

    def preprocess_datatypes(df_plans, df_clicks, df_queries):
        df_plans.sid = df_plans.sid.astype(int)

        # Check if clicks is empty because clicks just for train
        if df_clicks is not None:
            df_clicks.sid = df_clicks.sid.astype(int)

        df_queries.sid = df_queries.sid.astype(int)
        df_queries.req_time = pd.to_datetime(df_queries.req_time)

        return df_plans, df_clicks, df_queries

    def join_data_sets(df_plans, df_clicks, df_queries, df_profiles, df_mode):
        """
            This function joins all datasets together.
        """

        if df_mode == 'col':
            # adds 2 columns
            if df_clicks is not None:
                df = pd.merge(df_clicks, df_queries, on="sid", how='outer')
            else:
                df = df_queries.copy()
            
            # adds 66 columns
            df = pd.merge(df, df_profiles, how='outer')
            df = df[pd.notnull(df['o_long'])]
            
            # adds 2 columns
            df = pd.merge(df, df_plans, how='outer')

        elif df_mode == 'row':
            if df_clicks is not None:
                df = pd.merge(df_clicks, df_plans, on="sid")
            else:
                df = df_plans.copy()
            df = pd.merge(df, df_queries, on="sid")
            df = pd.merge(df, df_profiles, how='outer')
            df = df[pd.notnull(df['o_long'])]

        else:
            print("Wrong df_mode, try with 'col' or 'row'")
            sys.exit(-1)
        
        df.pid = df.pid.apply(lambda x: 0 if np.isnan(x) else x)
        for i in range(66):
            df['p'+str(i)] = df['p'+str(i)].apply(lambda x: -1 if np.isnan(x) else x)

        return df

    '''
    for 'row' mode, neede for lambdarank
    '''
    def unstack_plans(df_plans):
        df = df_plans.copy()

        df.plans = df.apply(
            lambda x: json.loads(
                '{"plans":'
                + x.plans
                + ',"sid":"'
                + str(x.sid)
                + '"'
                + ',"plan_time":'
                + '"'
                + str(x.plan_time)
                + '"}'
            ), axis=1)

        df_unstacked = json_normalize(df.plans.values, "plans", ["sid", "plan_time"])
        df_unstacked.rename({"distance": "distance_plan"}, axis=1, inplace=True)
        return df_unstacked

    '''
    for col mode, unstack plans in columns, necessary for random forest classifier
    '''
    def initialize_plan_cols(df, modes):
        for mode in modes:
            df[mode] = 0
        return df

    # 'first' mode: only the first proposed plan per transport mode is considered
    def preprocess_plans_first(df):
        '''
        Creates 33 new colums, 3 per transport mode
        * dist_0
        * price_0
        * eta_0

        +33 columns
        '''
        for i, r in df.iterrows():
            if (i+1) % 5000 == 0:
                print("Processing row {}".format(str(i + 1)), end="\r")
            if isinstance(r.plans, float):
                # nan, no plan suggestions
                continue
            for pl in json.loads(r.plans):
                df.at[i,'dist_' + str(pl['transport_mode'])] = pl['distance']
                if pl['price']:
                    df.at[i,'price_' + str(pl['transport_mode'])] = pl['price']
                else:
                    df.at[i,'price_' + str(pl['transport_mode'])] = 700
                df.at[i,'eta_' + str(pl['transport_mode'])] = pl['eta']
        print("\n")
        return df


    # 'last' mode: only the last proposed plan per transport mode is considered
    def preprocess_plans_last(df):
        for i, r in df.iterrows():
            if (i+1) % 5000 == 0:
                print("Processing row {}".format(str(i + 1)), end="\r")
            if isinstance(r.plans, float):
                # nan
                continue
            visited = []
            for pl in json.loads(r.plans):
                if pl['transport_mode'] in visited:
                    continue
                visited.append(pl['transport_mode'])
                df.at[i,'dist_' + str(pl['transport_mode'])] = pl['distance']
                if pl['price']:
                    df.at[i,'price_' + str(pl['transport_mode'])] = pl['price']
                else:
                    df.at[i,'price_' + str(pl['transport_mode'])] = 99999
                df.at[i,'eta_' + str(pl['transport_mode'])] = pl['eta']
        print("\n")
        return df

    '''
    Preprocessing pipeline
    '''

    print("Preprocessing coordinates")
    df = preprocess_coordinates(df)

    if df_mode == 'col':
        print("Preprocessing df in 'col' mode")
        plandf, clickdf, df = preprocess_datatypes(plandf, clickdf, df)
        df = join_data_sets(plandf, clickdf, df, profiledf, df_mode)

        num_modes = 12
        modes = []
        for i in range(num_modes):
            modes.append('dist_' + str(i))
            modes.append('price_' + str(i))
            modes.append('eta_' + str(i))

        df = initialize_plan_cols(df, modes)
        if plan_mode == 'first':
            print("Preprocessing plans in 'first' mode")
            df = preprocess_plans_first(df)
        elif plan_mode == 'last':
            print("Preprocessing plans in 'last' mode")
            df = preprocess_plans_last(df)
        else:
            print("ERROR: wrong plan mode. Try with 'first' or 'last'.")
            sys.exit(1)


    elif df_mode == 'row':
        print("Preprocessing df in 'row' mode")
        df_plans_pp = unstack_plans(plandf)
        df_plans_pp, clickdf, df = preprocess_datatypes(df_plans_pp, clickdf, df)
        df = join_data_sets(df_plans_pp, clickdf, df, profiledf, df_mode)
    else:
        print("Wrong df mode, try with 'row' or 'col'")
        sys.exit(-1)

    if 'click_mode' in df:
        print("Preprocessing click_mode")
        df.click_mode = df.click_mode.apply(lambda x: 0 if np.isnan(x) else x)

    if 'plans' in df:
        df = df.drop('plans', axis=1)
    return df

@click.command()
@click.argument("absolute_path_data_folder")
@click.argument("df_mode")
@click.argument("plan_mode")
def main(absolute_path_data_folder, df_mode, plan_mode):

    df_profiles, df_train_queries, df_train_plans, df_train_clicks, df_test_queries, df_test_plans = read_in_data(absolute_path_data_folder)
    
    print("traindf: creating raw features for df_train")
    df_train = raw_preprocessing(df_train_queries, df_train_plans, df_profiles, clickdf=df_train_clicks, df_mode=df_mode, plan_mode=plan_mode)
    print("Writing train to pickle in ../data/processed_raw/")
    write_data(absolute_path_data_folder, df_train, 'train', df_mode, plan_mode)

    print("\n")
    print("testdf_ creating raw features for df_test")
    df_test = raw_preprocessing(df_test_queries, df_test_plans, df_profiles, df_mode=df_mode, plan_mode=plan_mode)
    print("Writing test to pickle in ../data/processed_raw/")
    write_data(absolute_path_data_folder, df_test, 'test', df_mode, plan_mode)

    return

if __name__ == "__main__":
    main()