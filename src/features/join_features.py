import pandas as pd
import pickle
import os
import sys

def main():
    
    print("Reading external dataframes")
    exttrain = pd.read_pickle('data/external_features/train_external.pickle')
    exttest = pd.read_pickle('data/external_features/test_external.pickle')

    exttrain = exttrain.drop(['pid', 'req_time'], axis=1)
    exttest = exttest.drop(['pid', 'req_time'], axis=1)
    
    inpath = 'data/processed_raw'
    outpath = 'data/processed_all'

    print("Processing pickles")
    for pick in os.listdir(inpath):
        print("Reading: " + pick)
        df = pd.read_pickle(os.path.join(inpath, pick))
        if 'train' in pick:
            newdf = pd.merge(df, exttrain)
        elif 'test' in pick:
            newdf = pd.merge(df, exttest)
        outpick = pick.replace('raw', 'all')
        newdf.to_pickle(os.path.join(outpath, outpick))


if __name__ == "__main__":
    main()