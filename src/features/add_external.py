
import pandas as pd
import pickle
import os
import sys

from build_features import add_dist_nearest_subway, time_features, add_public_holidays, add_weather_features, preprocess_coordinates
from assign_districts_main import assign_districts

def add_features(df):
    print("Adding coordinate features")
    df = preprocess_coordinates(df)
    print("Adding subway features")
    df = add_dist_nearest_subway(df)
    df = pd.DataFrame(df)
    df = df.drop(['o_long', 'o_lat', 'd_long', 'd_lat'], axis=1)
    print("Adding time features")
    df = time_features(df)
    print("Adding public holidays features")
    df = add_public_holidays(df)
    print("Adding weather features")
    df = add_weather_features(df)
    print("Assign districts")
    df = assign_districts(df)
    return df


def main():
    
    print("Reading dataframes")
    df_train = pd.read_csv("data/raw/data_set_phase1/train_queries.csv")
    # df_test = pd.read_csv("data/raw/data_set_phase1/test_queries.csv")
    
    print("Adding features in df_train")
    df_train = add_features(df_train)

    print("Added all features in df_train")
    df_train.to_pickle("data/external_features/external.pickle")

    ''' DEPRECATED
    print("Adding features in df_test")
    df_test = add_features(df_test)

    print("Added all features in df_test")
    df_test.to_pickle("data/external_features/test_external.pickle")
    '''
    
    return

if __name__ == "__main__":
    main()