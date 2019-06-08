import pandas as pd
import pickle
import os
import sys

def main():
    
    print("Reading external dataframes")
    ext = pd.read_pickle('data/external_features/external.pickle')

    ext = ext.drop(['pid', 'req_time'], axis=1)
    
    inpath = 'data/processed_raw'
    outpath = 'data/processed_all'

    print("Processing pickles")
    for pick in os.listdir(inpath):
        if 'row' in pick:
            continue
        print("Reading: " + pick)
        df = pd.read_pickle(os.path.join(inpath, pick))
        print("Before: " + str(df.shape[0]) + ", " + str(df.shape[1]))
        newdf = pd.merge(df, ext, how='inner')
        print("After: " + str(newdf.shape[0]) + ", " + str(newdf.shape[1]))
        outpick = pick.replace('raw', 'all')
        print("Writing: " + outpick)
        newdf.to_pickle(os.path.join(outpath, outpick))


if __name__ == "__main__":
    main()