import pandas as pd
import numpy as np
import json

def expand_plans(df, traindf):

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

    def preprocess_plans(df):
        for i, r in df.iterrows():
            if (i+1) % 1000 == 0:
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

    alltr = pd.merge(traindf,planstr,how='outer')

    num_modes = 12
    modes = []
    for i in range(num_modes):
        modes.append('dist_' + str(i))
        modes.append('price_' + str(i))
        modes.append('eta_' + str(i))

    
    alltr = initialize_plan_cols(alltr, modes)
    alldf = preprocess_plans(alltr)

    return alldf

traindf = pd.read_csv('data/raw/data_set_phase1/train_queries.csv')
testdf = pd.read_csv('data/raw/data_set_phase1/test_queries.csv')
planstr = pd.read_csv('data/raw/data_set_phase1/train_plans.csv')
planste = pd.read_csv('data/raw/data_set_phase1/test_plans.csv')

traindf = expand_plans(traindf, planstr)
testdf = expand_plans(testdf, planste)

traindf.to_pickle('data/processed/train_preprocess.pickle')
testdf.to_pickle('data/processed/test_preprocess.pickle')