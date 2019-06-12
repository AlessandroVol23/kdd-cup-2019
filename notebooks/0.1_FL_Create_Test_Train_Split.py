# -*- coding: utf-8 -*-
"""
Script to sample SIDs from the [multiclass-]TrainSet into 'k'-folds.
[These SIDs are used for the Test/Train Splits in the CrossValidation]
"""
# Load needed packages
import pandas as pd
import random
import pickle
import os

# Check Working Directory:
if os.getcwd()[-12:] != "kdd-cup-2019":
	raise ValueError("Error with WorkingDirectory \n \
			[1] either WD isn't set correctly to 'kdd-cup-2019' \n \
			[2] or WD was renamed")

# Load the multiclass Traindata & extract the "sid"-Column:
train_set = pd.read_pickle("data/processed/multiclass/train_all_first.pickle")
all_sids  = train_set["sid"]

# Set seed for Reproducibility and save sids in cahnged order!
random.seed(9001)
sampled_ids = random.sample(list(all_sids), k = len(all_sids))

# check whether the sampled_ids have the same length as TrainSet["sid"]:
if len(set(sampled_ids)) != len(set(all_sids)):
	raise ValueError("Error, sampled SIDs have less elements than TrainSet has SIDs")

def slice_to_evely_sized_chunks(k):
	"""
	slice the 'sampled_ids' into k evely sized chunks
	"""
	# get the length of each fold & check whether it has no floats:
	sample_sizes = len(sampled_ids) / k
	
	if sample_sizes % 1 != 0:
		raise ValueError("k can not divide the SIDs into evely sized chunks")
		
	# Check whether the folder to save the results is existent:
	if not os.path.isdir("data/processed/Test_Train_Splits/" + str(k) + "-fold"):
		raise ValueError("There is no " + str(k) + "-fold Folder in data/processed/Test_Train_Splits/")
	
	# Define List to save SIDs for each fold
	SIDs = []
	
	# Slice the 'sampled_ids' into k evely sized chunks:
	for fold in range(k):
		SIDs.append(sampled_ids[int(fold * sample_sizes) : int((fold + 1) * sample_sizes)])

	# Save the Results as Single pickle File:
	with open("data/processed/Test_Train_Splits/" + str(k) + "-fold/SIDs.pickle", "wb") as fp:
		pickle.dump(SIDs, fp)


slice_to_evely_sized_chunks(5)
slice_to_evely_sized_chunks(10)