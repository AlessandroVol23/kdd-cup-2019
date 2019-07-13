# -*- coding: utf-8 -*-
"""
Script to simulate a submitt on the TestSet.

For this we need to pass a ".csv"-File with the 2 columns: "SID" & "y_hat"
where y_hat is the predicted TransportationMode for the corresponding SID!  

If the passed ".csv" has the right form [2 columns: "SID" & "y_hat"] + 
contains all SIDs, that the original TestSet Contains, this script will return
a F1-Score!
"""
# Load needed Packages
import pandas as pd
import sklearn.metrics as metrics
import os

# Check Working Directory:
if os.getcwd()[-12:] != "kdd-cup-2019":
	raise ValueError("Error with WorkingDirectory \n \
			[1] either WD isn't set correctly to 'kdd-cup-2019' \n \
			[2] or WD was renamed")

#%% Define Function to calculate the F1-Score on the TestSet:

def get_f1_test_score(predictions):
	"""
	Function to calculate the F1-Score on the TestSet.
	
	For this we compare the predicted Value of each SID with the true Value of 
	the SID [stored at: 'data/processed/Ranking/test_raw_row.pickle']
	
	Args:
		- predictions (pandas) : DF with 2 Columns "SID" & "y_hat"
	
	Return:
		- F1-Score of the passed data compared with the true responses.
		  [stored at: 'data/processed/Ranking/test_raw_row.pickle']
	"""
	# [1] Load the TestData w/ the true responses
	print("--- Load the true Responses ---\n")
	# [1-1] Load the Data
	test_set = pd.read_pickle("data/processed/Ranking/test_raw_row.pickle")
	
	# [1-2] Only keep the SID and the click_mode [= Response]
	test_set = test_set[["sid", "Response"]]
	
	# [1-3] Only keep unique SIDs [ranking layout w/ multiple rows per SID]
	test_set = test_set.drop_duplicates()
	
	
	# [2] Check whether the Input is correct:
	print("--- Check the Input ---\n")
	# [2-1] Check wheter the predictions-DF has the right columns, and doesn't
	#       contain any more columns as the needed ones!
	if "sid" not in list(predictions):
		raise ValueError("the passed predictions-DF needs an 'sid' column")
	
	if "y_hat" not in list(predictions):
		raise ValueError("the passed predictions-DF needs an 'y_hat' column")
		
	if len(list(predictions)) != 2:
		raise ValueError("predictions-DF must only have 'y_hat' & 'sid' as columns")
	
	
	# [3] Check that the predictions-DF only contains SIDs that are in the 
	#     DF with the true responses + check that all SIDs from there are 
	#     also in the predictions-DF!
	# [3-1] Remove (possible) duplicated sids from the predictions-DF
	predictions = predictions.drop_duplicates()
	
	# [3-2] Check whether the SIDs are equal to the ones in the test_set!
	test_set["sid"].equals(predictions["sid"])
	
	
	# [4] Calculate the F-1 Score of the Predicitons
	print("--- Calcualte the F-1 Score ---\n")
	# [4-1] Sort both DFs according to their SID, so the responses do have 
	#         the same order and can be compared easily!
	test_set    = test_set.sort_values(by=['sid'])
	predictions = predictions.sort_values(by=['sid'])
	
	# [4-2] Calculate the F1 Score
	f1_score = metrics.f1_score(test_set["Response"], predictions["y_hat"], 
				                average = "weighted")
	
	return(f1_score)
	
#%% Example of Usage:
# Either load a File with the Predicitons 
# OR
# Load the TestSet and create an artifical prediction DF:
preds = pd.read_pickle("data/processed/Ranking/test_raw_row.pickle")

# Only keep relevant columns & rename them:
preds = preds[["sid", "Response"]]

# Drop duplicates and change a predicted Value, so we don't get an 1.0 F1-Score
preds = preds.drop_duplicates()
preds.columns = ['sid', 'y_hat']
preds["y_hat"].iloc[2] = 2.0

get_f1_test_score(predictions = preds)
