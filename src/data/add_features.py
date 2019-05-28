import pandas as pd
import pickle
import os
import sys
sys.path.append("../src/")

from features.build_features import add_dist_nearest_subway, time_features, add_public_holidays, add_weather_features

def add_features(df):
    print("Adding subway features")
    df = df = add_dist_nearest_subway(df)
    print("Adding time features")
    df = time_features(df)
    print("Adding public holidays features")
    df = add_public_holidays(df)
    print("Adding weather features")
    df = add_weather_features(df)
    return df

def main():
    df_train = pd.read_pickle("../data/processed/df_train_subway.pickle")
    df_test = pd.read_pickle("../data/processed/df_test_subway.pickle")
    
    print("Adding features in df_train")
    df_train = add_features(df_train)

    print("Adding features in df_test")
    df_test = add_features(df_test)

    print("Added all features in df_train and df_test")

    df_train.to_pickle("../data/processed/df_train_features.pickle")
    df_test.to_pickle("../data/processed/df_test_features.pickle")

if __name__ == "__main__":
    main()