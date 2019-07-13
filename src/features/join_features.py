import pandas as pd
import pickle
import os
import sys

def main():
    
    print("Reading external dataframes")
    ext = pd.read_pickle('data/processed/features/external.pickle')

    ext = ext.drop(['pid', 'req_time'], axis=1)
    
    inpath = ['data/processed/ranking', 'data/processed/multiclass']

    print("Processing pickles")
    for inputdir in inpath:
        visited_files = []
        for pick in os.listdir(inputdir):
            if os.path.splitext(pick)[1] != '.pickle' or pick in visited_files or 'all' in pick:
                continue
            print("Reading: " + pick)
            df = pd.read_pickle(os.path.join(inputdir, pick))
            print("Before: " + str(df.shape[0]) + ", " + str(df.shape[1]))
            newdf = pd.merge(df, ext, how='inner')
            print("After: " + str(newdf.shape[0]) + ", " + str(newdf.shape[1]))
            outpick = pick.replace('raw', 'all')
            print("Writing: " + outpick)
            newdf.to_pickle(os.path.join(inputdir, outpick))
            visited_files.append(pick)
            visited_files.append(outpick)


if __name__ == "__main__":
    main()
