import os
import sys
import json
import math
import click
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm

from pandas.io.json import json_normalize

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def read_in_train_data(absolute_raw_data_path):
    """
        This function reads in all the train data sets.
        returns: df_train_queries, df_train_plans, df_train_clicks, id_profiles
    """

    data_set_path = os.path.join(absolute_raw_data_path, 'raw/data_set_phase1')

    df_profiles = pd.read_csv(os.path.join(data_set_path, "profiles.csv"))
    df_train_clicks = pd.read_csv(os.path.join(data_set_path, "train_clicks.csv"))
    df_train_plans = pd.read_csv(os.path.join(data_set_path, "train_plans.csv"))
    df_train_queries = pd.read_csv(os.path.join(data_set_path, "train_queries.csv"))

    return (df_profiles, df_train_queries, df_train_plans, df_train_clicks)

def read_in_test_data(absolute_raw_data_path):
    """
    This function reads in all the test data sets.
    returns: df_test_queries, df_test_plans
    """
    data_set_path = os.path.join(absolute_raw_data_path, 'raw/data_set_phase1')
    df_test_queries = pd.read_csv(os.path.join(data_set_path, "test_queries.csv"))
    df_test_plans = pd.read_csv(os.path.join(data_set_path, "test_plans.csv"))
    return (df_test_queries, df_test_plans)


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
            #df = pd.merge(df, df_profiles, how='outer')
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

    def fill_missing_price(df, median=True, mean=False):
        """
            This function fills all missing values in price with the median value. 
        """
        df.loc[df.price.isna(), "price"] = df.price.median()
        return df

    # DEPRECATED
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
        return df

    def gen_plan_feas(data, col_name='plans'):
        n = data.shape[0]
        mode_list_feas = np.zeros((n, 12))
        max_dist, min_dist, mean_dist, std_dist = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

        max_price, min_price, mean_price, std_price = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

        max_eta, min_eta, mean_eta, std_eta = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

        min_dist_mode, max_dist_mode, min_price_mode, max_price_mode, min_eta_mode, max_eta_mode, first_mode = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
        mode_texts = []
        
        for i, plan in tqdm(enumerate(data[col_name].values)):
            try:
                cur_plan_list = json.loads(plan)
            except:
                cur_plan_list = []
            if len(cur_plan_list) == 0:
                mode_list_feas[i, 0] = 1
                first_mode[i] = 0

                max_dist[i] = -1
                min_dist[i] = -1
                mean_dist[i] = -1
                std_dist[i] = -1

                max_price[i] = -1
                min_price[i] = -1
                mean_price[i] = -1
                std_price[i] = -1

                max_eta[i] = -1
                min_eta[i] = -1
                mean_eta[i] = -1
                std_eta[i] = -1

                min_dist_mode[i] = -1
                max_dist_mode[i] = -1
                min_price_mode[i] = -1
                max_price_mode[i] = -1
                min_eta_mode[i] = -1
                max_eta_mode[i] = -1

                mode_texts.append('word_null')
            else:
                distance_list = []
                price_list = []
                eta_list = []
                mode_list = []
                for tmp_dit in cur_plan_list:
                    distance_list.append(int(tmp_dit['distance']))
                    if tmp_dit['price'] == '':
                        price_list.append(0)
                    else:
                        price_list.append(int(tmp_dit['price']))
                    eta_list.append(int(tmp_dit['eta']))
                    mode_list.append(int(tmp_dit['transport_mode']))
                mode_texts.append(' '.join(['word_{}'.format(mode) for mode in mode_list]))
                distance_list = np.array(distance_list)
                price_list = np.array(price_list)
                eta_list = np.array(eta_list)
                mode_list = np.array(mode_list, dtype='int')
                mode_list_feas[i, mode_list] = 1
                distance_sort_idx = np.argsort(distance_list)
                price_sort_idx = np.argsort(price_list)
                eta_sort_idx = np.argsort(eta_list)

                max_dist[i] = distance_list[distance_sort_idx[-1]]
                min_dist[i] = distance_list[distance_sort_idx[0]]
                mean_dist[i] = np.mean(distance_list)
                std_dist[i] = np.std(distance_list)

                max_price[i] = price_list[price_sort_idx[-1]]
                min_price[i] = price_list[price_sort_idx[0]]
                mean_price[i] = np.mean(price_list)
                std_price[i] = np.std(price_list)

                max_eta[i] = eta_list[eta_sort_idx[-1]]
                min_eta[i] = eta_list[eta_sort_idx[0]]
                mean_eta[i] = np.mean(eta_list)
                std_eta[i] = np.std(eta_list)

                first_mode[i] = mode_list[0]
                max_dist_mode[i] = mode_list[distance_sort_idx[-1]]
                min_dist_mode[i] = mode_list[distance_sort_idx[0]]

                max_price_mode[i] = mode_list[price_sort_idx[-1]]
                min_price_mode[i] = mode_list[price_sort_idx[0]]

                max_eta_mode[i] = mode_list[eta_sort_idx[-1]]
                min_eta_mode[i] = mode_list[eta_sort_idx[0]]

        feature_data = pd.DataFrame(mode_list_feas)
        feature_data.columns = ['mode_{}_available'.format(i) for i in range(12)]
        feature_data['max_dist'] = max_dist
        feature_data['min_dist'] = min_dist
        feature_data['mean_dist'] = mean_dist
        feature_data['std_dist'] = std_dist

        feature_data['max_price'] = max_price
        feature_data['min_price'] = min_price
        feature_data['mean_price'] = mean_price
        feature_data['std_price'] = std_price

        feature_data['max_eta'] = max_eta
        feature_data['min_eta'] = min_eta
        feature_data['mean_eta'] = mean_eta
        feature_data['std_eta'] = std_eta

        feature_data['max_dist_mode'] = max_dist_mode
        feature_data['min_dist_mode'] = min_dist_mode
        feature_data['max_price_mode'] = max_price_mode
        feature_data['min_price_mode'] = min_price_mode
        feature_data['max_eta_mode'] = max_eta_mode
        feature_data['min_eta_mode'] = min_eta_mode
        feature_data['first_mode'] = first_mode
        
        print('mode tfidf...')
        tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_vec = tfidf_enc.fit_transform(mode_texts)
        svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
        mode_svd = svd_enc.fit_transform(tfidf_vec)
        mode_svd = pd.DataFrame(mode_svd)
        mode_svd.columns = ['TFIDF_clustered_{}'.format(i) for i in range(10)]
        data = pd.concat([data, feature_data, mode_svd], axis=1)
            
        #data = data.drop([col_name], axis=1)
        return data

    def gen_profile_feas(data, profile_data):
        
        # add "-1" as new PID, that has "-1" on all p0,...p65
        profile_data.loc[len(profile_data)] = list(np.repeat(-1, 67))
        x = profile_data.drop(['pid'], axis=1).values
        svd = TruncatedSVD(n_components=20, n_iter=20, random_state=2019)
        svd_x = svd.fit_transform(x)
        svd_feas = pd.DataFrame(svd_x)
        newcols = ['PID_MainComp_{}'.format(i) for i in range(20)]
        svd_feas.columns = newcols
        svd_feas['pid'] = profile_data['pid'].values
        data['pid'] = data['pid'].fillna(-1)
        data = data.merge(svd_feas, on='pid', how='left')
        for svd_mode in newcols:
            data[svd_mode] = data.apply(lambda x: -1 if x.pid == -1 else x[svd_mode], axis=1)
        return data

    def gen_time_feas(data):
        data['req_time'] = pd.to_datetime(data['req_time'])
        data['weekday'] = data['req_time'].dt.dayofweek
        data['hour'] = data['req_time'].dt.hour
        return data

    def split_train_test_by_col(df):
        mask = df['req_time'] >= '2018-11-14'
        train = df[~mask]
        test = df[mask]
        train = train.drop(['req_time'], axis=1)
        test = test.drop(['req_time'], axis=1)
        return (train, test)

    def split_train_test_by_row(df):
        limit = int(0.8*df.shape[0])
        traindf = df.head(limit)
        testdf = df.tail(len(df) - limit)
        return traindf, testdf

    '''
    **********************
    Preprocessing pipeline
    **********************
    '''

    print("1. Preprocessing coordinates and time")
    df = preprocess_coordinates(df)
    df = gen_time_feas(df)

    if df_mode == 'col':
        print("2. Preprocessing df in 'col' mode")
        plandf, clickdf, df = preprocess_datatypes(plandf, clickdf, df)
        print("3. Generating profile features")
        df = gen_profile_feas(df, profiledf)
        df = join_data_sets(plandf, clickdf, df, profiledf, df_mode)
        print("4. Generating plan features")
        df = gen_plan_feas(df, col_name='plans')
        
        num_modes = 12
        modes = []
        for i in range(num_modes):
            modes.append('dist_' + str(i))
            modes.append('price_' + str(i))
            modes.append('eta_' + str(i))

        df = initialize_plan_cols(df, modes)
        if plan_mode == 'first':
            print("5. Preprocessing plans in 'first' mode")
            df = preprocess_plans_first(df)
        elif plan_mode == 'last':
            print("5. Preprocessing plans in 'last' mode")
            df = preprocess_plans_last(df)
        else:
            print("ERROR: wrong plan mode. Try with 'first' or 'last'.")
            sys.exit(1)

    elif df_mode == 'row':
        print("2. Preprocessing df in 'row' mode")
        df0 = df.merge(plandf, on='sid', how='left')
        print("3. Generating profile features")
        df0 = gen_profile_feas(df0, profiledf)
        print("4. Generating plan features")
        df_with_plans = gen_plan_feas(df0)
        print("5. Unstacking plans into rows")
        df_plans_pp = unstack_plans(plandf)
        df_plans_pp, clickdf, df = preprocess_datatypes(df_plans_pp, clickdf, df)
        df = join_data_sets(df_plans_pp, clickdf, df_with_plans, profiledf, df_mode)
        df = fill_missing_price(df)
    else:
        print("Wrong df mode, try with 'row' or 'col'")
        sys.exit(-1)

    if 'click_mode' in df:
        print("6. Preprocessing click_mode")
        df.click_mode = df.click_mode.apply(lambda x: 0 if np.isnan(x) else x)
        df['Response'] = df.click_mode

    if 'plans' in df:
        df = df.drop('plans', axis=1)
    
    print("7. Split train and test")
    print(str(df.shape[0]), str(df.shape[1]))
    #traindf, testdf = split_train_test_by_col(df)
    traindf, testdf = split_train_test_by_row(df)

    return (traindf, testdf)

@click.command()
@click.argument("absolute_path_data_folder")
@click.argument("df_mode")
@click.argument("plan_mode")
def main(absolute_path_data_folder, df_mode, plan_mode):

    df_profiles, df_train_queries, df_train_plans, df_train_clicks = read_in_train_data(absolute_path_data_folder)
    # df_test_queries, df_test_plans = read_in_test_data(absolute_raw_data_path)
    
    print("traindf: creating raw features for df_train")
    df_tr_train, df_tr_test = raw_preprocessing(df_train_queries, df_train_plans, df_profiles, clickdf=df_train_clicks, df_mode=df_mode, plan_mode=plan_mode)
    write_data(absolute_path_data_folder, df_tr_train, 'train', df_mode, plan_mode)
    write_data(absolute_path_data_folder, df_tr_test, 'test', df_mode, plan_mode)

    return
    '''DEPRECATED
    print("\n")
    print("testdf_ creating raw features for df_test")
    df_test = raw_preprocessing(df_test_queries, df_test_plans, df_profiles, df_mode=df_mode, plan_mode=plan_mode)
    write_data(absolute_path_data_folder, df_test, 'test', df_mode, plan_mode)
    '''

if __name__ == "__main__":
    main()