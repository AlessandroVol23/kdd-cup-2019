import os
import sys
import json
import click
import logging
import pandas as pd
import numpy as np

def raw_preprocessing(df, plandf, plan_mode):

    def generate_coordinates(df):
        df[["o_long", "o_lat"]] = df.o.str.split(",", 1, expand=True).astype(float)
        df.drop("o", axis=1, inplace=True)

        df[["d_long", "d_lat"]] = df.d.str.split(",", 1, expand=True).astype(float)
        df.drop("d", axis=1, inplace=True)
        
        return df

    def initialize_plan_cols(df, modes):
        for mode in modes:
            df[mode] = 0
        return df

    def preprocess_plans_first(df):
        for i, r in df.iterrows():
            if (i+1) % 5000 == 0:
                print("Processing row {}".format(str(i + 1)), end="\r")
            if isinstance(r.plans, float):
                # nan
                continue
            for pl in json.loads(r.plans):
                df.at[i,'dist_' + str(pl['transport_mode'])] = pl['distance']
                if pl['price']:
                    df.at[i,'price_' + str(pl['transport_mode'])] = pl['price']
                else:
                    df.at[i,'price_' + str(pl['transport_mode'])] = 999
                df.at[i,'eta_' + str(pl['transport_mode'])] = pl['eta']
        return df

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
                    df.at[i,'price_' + str(pl['transport_mode'])] = 999
                df.at[i,'eta_' + str(pl['transport_mode'])] = pl['eta']
        return df

    df = generate_coordinates(df)

    df = pd.merge(df, plandf, how='outer')

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

    df.drop('plans', axis=1)

    return df

@click.command()
@click.argument("absolute_path_data_folder")
@click.argument("plan_mode")
def main(absolute_path_data_folder, plan_mode):

    """ 
        Script to construct dataset with raw features:
        * sid, pid, req_time, o_lat, o_long, d_lat, d_long
        * 3 columns per plan mode, dist_X, price_X, eta_x
        Script looks in absolut_path_raw_folder for the data folder in the project.
        absolute_path_data_folder: Path to raw data folder. Typically /home/xxx/repo/data/raw
        plan_mode: 'first' or 'last'. Sometimes there are several suggestions for the same transport mode. 'first' mode takes the first suggestion, 'last' takes only the last suggestion.
    """

    df_train_queries = pd.read_csv(os.path.join(absolute_path_data_folder, "raw/data_set_phase1/train_queries.csv"))
    df_train_plans = pd.read_csv(os.path.join(absolute_path_data_folder, "raw/data_set_phase1/train_plans.csv"))
    
    df_test_queries = pd.read_csv(os.path.join(absolute_path_data_folder, "raw/data_set_phase1/test_queries.csv"))
    df_test_plans = pd.read_csv(os.path.join(absolute_path_data_folder, "raw/data_set_phase1/test_plans.csv"))
    
    print("Creating raw features for df_train")
    df_train_queries = raw_preprocessing(df_train_queries, df_train_plans, plan_mode)
    print("Creating raw features for df_test")
    df_test_queries = raw_preprocessing(df_test_queries, df_test_plans, plan_mode)

    print("Writing train and test to pickle in ../data/processed/")
    df_train_queries.to_pickle(os.path.join(absolute_path_data_folder, 'processed/train_raw_' + plan_mode + '.pickle'))
    df_test_queries.to_pickle(os.path.join(absolute_path_data_folder, 'processed/test_raw_' + plan_mode + '.pickle'))
    return

if __name__ == "__main__":
    main()