# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:07:36 2019

@author: kuche_000

Multiclass Baselinemodels - KDD Cup
"""
# Packages
import pandas as pd
import numpy as np
import sklearn.svm
import sklearn.metrics as metrics
import sklearn.tree
import sklearn.ensemble
import time
import os
import pickle
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load the preprocessed data
data = pd.read_pickle("data/processed/data_set_phase")

# --- (1) --- 
"""
Clean the DF:
	- each session should only offer one possibility for each transportation
	- if there are twice possibilities for the same transportation mode 
	  [e.g. twice possible connections with transportation mode "1"] 
	  only keep the first one!
	- checked couple of these secenarios, the ones offerd later, always 
	  worse in time or/and money... [reasonable sorted already]
"""
data_clean = data.drop_duplicates(["sid", "transport_mode"])

# --- (2) ---
"""
Create DF to train Multiclass with:
	- Each row one Session-ID
	- to each Session ID the transport specific cost/ time/ distance in a row
	  [cost_1, ..., cost_11 -> cost for each transportation mode in an own col]
    - Non offered transport modes "NaN" [can be replaced later on]
"""

# Function to extract Information fromk data_clean [created above^]
def create_multiclass_df(indeces, DF_name):
	"""
	Function to extraxt the transport specific distances/ costs/ times.
	Add them all to "df_" and save it as pickle
	
	
	Args:
		- indeces (list) : list of integers with all session IDs we want to 
		                   extract [need to be in data_clean]
		- DF_name (str)  : Addtion to the Name the DF shall be safed in
		                   [e.g. "2" --> "SVM_DF_2"]
    Return:
		- DF (pandas)    : Dataframe with a column for each transpotationmode
		                   specific Distance/Time/Cost
						   [Distance_1, ..., Distance_11,
						   Time_1, ...,Time_11,
						   Cost_1, ..., Cost_11] 
	"""
	df_ = pd.DataFrame(columns = ["Distance_1", "Distance_2", "Distance_3",
				  "Distance_4", "Distance_5", "Distance_6", "Distance_7", 
				  "Distance_8", "Distance_9", "Distance_10", "Distance_11",
				  "Time_1", "Time_2", "Time_3", "Time_4", "Time_5", "Time_6", 
				  "Time_7", "Time_8", "Time_9", "Time_10", "Time_11",
				  "Cost_1", "Cost_2", "Cost_3", "Cost_4", "Cost_5", "Cost_6",
				   "Cost_7", "Cost_8", "Cost_9", "Cost_10", "Cost_11"])
	
	# Loop over all SID and return an longitudinal DF [each transportsession one column]
	counter = 0
	for index in indeces:
		
		# print the process:
		counter += 1
		if counter % 100 == 0: print(str(counter) + " / " + str(len(indeces)))
		
		# get all rows with the same Index (same session):
		current_sess  = data_clean.loc[data_clean["sid"] == index]
	
		# Get all offered transportation modes:
		possible_trans_modes = current_sess["transport_mode"].values
			
		# Create Dict to add it to the DF
		values_to_add = {}
		
		# Add all possible transportmodes specific times/ costs/ distances to df_
		for i in possible_trans_modes:
			
			values_to_add["Distance_" + str(i)] = current_sess.loc[current_sess["transport_mode"] == i]["distance_plan"].values[0]
			values_to_add["Cost_"     + str(i)] = current_sess.loc[current_sess["transport_mode"] == i]["price"].values[0]
			values_to_add["Time_"     + str(i)] = current_sess.loc[current_sess["transport_mode"] == i]["eta"].values[0]
		
		df_ = df_.append(values_to_add, ignore_index=True)
		
	# Add the SessionIDs  & save the Preprocessed DF:
	df_["SID"] = indeces		
	df_.to_pickle("data/processed/SVM_Approach/SubDFs/SVM_DF_" + DF_name)
	
# ---- Start extracting all the information for all Data-Points! --------------	
# Extract all unqiue session ids [for 10k about 617seconds]
unique_session_ids = data_clean["sid"].unique()

# Loop over all SessionIDs, save the results from time to time!
j = 0
for i in range(25000, len(unique_session_ids), 25000):
	
	# Print the current Range we're working on:
	print("from: "+ str(j) + " to " + str(i))
	
	# Do the Trasnforming
	create_multiclass_df(unique_session_ids[j:i], str(i / 25000))
	
	# update Indices
	j = i

# Last Part:
create_multiclass_df(unique_session_ids[450000:len(unique_session_ids)], "_19.0")

# Create one big DF from all sub_DFs! -----------------------------------------
df_all     = pd.read_pickle("data/processed/SVM_Approach/SubDFs/SVM_DF_1.0")

for i in range(2, 20, 1):
	
	# Create Filename, based on the current number!
	file_number = str(i)
	file_name   = "SVM_DF_" + file_number + ".0"
	
	# Read in the current Results:
	df_current = pd.read_pickle("data/processed/SVM_Approach/SubDFs/" + file_name)
	
	# Bind it to the big DF!
	df_all = pd.concat([df_all, df_current])
	
	
# Add the Response:
df_all["Response"] = data_clean.drop_duplicates("sid")["click_mode"].values

# Save the Files:
df_all.to_pickle("data/processed/SVM_Approach/MulticlassTrainSet")
data_clean.to_pickle("data/processed/SVM_Approach/data_clean")

# Quality Check:
df_all.loc[df_all["SID"] == 2101050]["Response"]
data_clean.loc[data_clean["sid"] == 2101050]["click_mode"]
	
#------------------------------------------------------------------------------
# Read data in:
df_all     = pd.read_pickle("data/processed/SVM_Approach/MulticlassTrainSet")
data_clean = pd.read_pickle("data/processed/SVM_Approach/data_clean")

# --- Add more features to df_ ---

# (1) Add the Request time per SID and label it as night or day!
req_times_  = data_clean.drop_duplicates("sid")["req_time"]

# convert the time to day/night [can be done faster actually...]
day_request = []
for i, tt in enumerate(req_times_):
	time_ = tt.split(" ")[1]
	if int(time_.split(":")[0]) in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
		day_request.append(True)
	else:
		day_request.append(False)
		
# Add to DF
df_all["day_request"] = day_request

# subset NA's
df_all = df_all.fillna(0)

# Extract Features ---
feature_names = ['Distance_1','Distance_2','Distance_3','Distance_4','Distance_5',
				 'Distance_6', 'Distance_7', 'Distance_8', 'Distance_9', 'Distance_10',
				 'Distance_11','Time_1', 'Time_2', 'Time_3', 'Time_4', 'Time_5',
				 'Time_6','Time_7','Time_8','Time_9','Time_10','Time_11','Cost_1',
				 'Cost_2', 'Cost_3','Cost_4','Cost_5','Cost_6','Cost_7','Cost_8',
				 'Cost_9','Cost_10', 'Cost_11','day_request']
features   = df_all[feature_names]
response   = np.array(df_all["Response"])


# Split Data with 5-fold CV ---------------------------------------------------
kf = KFold(n_splits = 5, random_state = 10)
kf.get_n_splits(features)

# Run 5-fold CV on different predicters, save the results for each

# [1] First Tree
tree_1 = sklearn.tree.DecisionTreeClassifier(criterion='entropy', 
											 max_depth=3,
											 random_state=0)

test_1_res = []
for train_index, test_index in kf.split(features):
	
	tree_1.fit(features.iloc[train_index], response[train_index])
	test_1_res.append(sklearn.metrics.f1_score(response[test_index], 
						                       tree_1.predict(features.iloc[test_index]),
									           average="weighted"))
	
# [2] Second Tree
tree_2 = sklearn.tree.DecisionTreeClassifier(criterion='entropy', 
											 max_depth=5,
											 random_state=0)

tree_2_res = []
for train_index, test_index in kf.split(features):
	
	tree_2.fit(features.iloc[train_index], response[train_index])
	tree_2_res.append(sklearn.metrics.f1_score(response[test_index], 
						                       tree_2.predict(features.iloc[test_index]),
									           average="weighted"))

# [3] RF_1
forest = sklearn.ensemble.RandomForestClassifier(criterion='entropy',
												 n_estimators=10, 
												 random_state=1,
												 n_jobs=2) # Parallelisierung auf 2 Kerne 
forest_1_res = []
for train_index, test_index in kf.split(features):
	
	forest.fit(features.iloc[train_index], response[train_index])
	forest_1_res.append(sklearn.metrics.f1_score(response[test_index], 
						                         forest.predict(features.iloc[test_index]),
									             average="weighted"))
	
# [4] RF_2
forest2 = sklearn.ensemble.RandomForestClassifier(criterion='entropy',
												  n_estimators=15, 
												  random_state=1,
												  n_jobs=2) # Parallelisierung auf 2 Kerne 
forest_2_res = []
for train_index, test_index in kf.split(features):
	
	forest2.fit(features.iloc[train_index], response[train_index])
	forest_2_res.append(sklearn.metrics.f1_score(response[test_index], 
						                         forest2.predict(features.iloc[test_index]),
									             average="weighted"))


#[5] NN 
# for NN scale the features 
scaler = StandardScaler()
scaler.fit(features)
features_scaled = scaler.transform(features)


# Define NN 1
mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 30))

MLP_res = []
for train_index, test_index in kf.split(features):
	
	mlp.fit(features_scaled[train_index], response[train_index])
	MLP_res.append(sklearn.metrics.f1_score(response[test_index], 
										    mlp.predict(features_scaled[test_index]),
									        average="weighted"))


# Define NN 2
mlp2 = MLPClassifier(hidden_layer_sizes=(50, 40, 30, 20))

MLP_res2 = []
for train_index, test_index in kf.split(features):
	
	mlp2.fit(features_scaled[train_index], response[train_index])
	MLP_res2.append(sklearn.metrics.f1_score(response[test_index], 
										     mlp2.predict(features_scaled[test_index]),
									         average="weighted"))