# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:55:07 2019

@author: kuche_000

Script to extract SIDs from the TrainSet, so we can do 5-fold-CV, on the same 
Session IDs, for all models [multiclass and LTR]
"""
import pandas as pd
import random
import pickle

# load the train_data:
train = pd.read_csv("data/raw/data_set_phase1/train_queries.csv")

# randomly sample 5 equal sized arrays of sids, use to slice the DFs
all_sids = train["sid"]

sampled_ids = random.sample(list(all_sids), k = len(all_sids))


# Sampling should not have changed any length settings:
len(sampled_ids) == len(all_sids)
len(set(sampled_ids)) == len(all_sids)


# Slice the List into 5 evely sized chunks:
sample_sizes = round(len(sampled_ids) / 5)
SIDs_1 = sampled_ids[0*sample_sizes: 1*sample_sizes]
SIDs_2 = sampled_ids[1*sample_sizes: 2*sample_sizes]
SIDs_3 = sampled_ids[2*sample_sizes: 3*sample_sizes]
SIDs_4 = sampled_ids[3*sample_sizes: 4*sample_sizes]
SIDs_5 = sampled_ids[4*sample_sizes:]

# Check for the length of the single samples:
len(SIDs_1)
len(SIDs_2)
len(SIDs_3)
len(SIDs_4)
len(SIDs_5)

# Save the Results as single pickle files:
with open("data/processed/Test_Train_Splits/SIDs_1.txt", "wb") as fp:
	pickle.dump(SIDs_1, fp)

with open("data/processed/Test_Train_Splits/SIDs_2.txt", "wb") as fp:
	pickle.dump(SIDs_2, fp)
	
with open("data/processed/Test_Train_Splits/SIDs_3.txt", "wb") as fp:
	pickle.dump(SIDs_3, fp)
	
with open("data/processed/Test_Train_Splits/SIDs_4.txt", "wb") as fp:
	pickle.dump(SIDs_4, fp)
	
with open("data/processed/Test_Train_Splits/SIDs_5.txt", "wb") as fp:
	pickle.dump(SIDs_5, fp)