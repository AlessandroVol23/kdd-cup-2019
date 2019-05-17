# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:32:20 2019

@author: kuche_000

First few Multiclass Models for KDD, based on the data in:
	"data\processed\Multiclass_Approach"
	
	- MulticlassTrainSet, is the DF, that can be used for training
	- MulticlassTestSet, is the DF, we hae to do predicitons for and submit them
	
	--> all features we use in MulticlassTrainSet, shall also be addded to:
		MulticlassTestSet!
		
	
	
"""
# Make sure, that your workingdirectory is on: "C:\Users\kuche_000\Desktop\kdd-cup-2019"

# Load all Packages needed
import pandas as pd
import numpy as np
import sklearn
import sklearn.tree
import sklearn.ensemble
import sklearn.metrics as metrics
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Read data in: ---------------------------------------------------------------
# Data we train the model with
df_train     = pd.read_pickle("data/processed/Multiclass_Approach/MulticlassTrainSet")
df_test      = pd.read_pickle("data/processed/Multiclass_Approach/MulticlassTestSet")

# data we use for feature extraction
df_train_clean = pd.read_pickle("data/processed/Multiclass_Approach/train_no_double_TransModes_per_sid")
df_test_clean  = pd.read_pickle("data/processed/Multiclass_Approach/test_no_double_TransModes_per_sid")

# Add features ----------------------------------------------------------------
# (1) Add the Request time per SID and label it as night or day!
req_times_train = df_train_clean.drop_duplicates("sid")["req_time"]
req_times_test  = df_test_clean.drop_duplicates("sid")["req_time"]

day_request_train = []
for i, tt in enumerate(req_times_train):
	time_ = tt.split(" ")[1]
	if int(time_.split(":")[0]) in [22, 23, 0, 1, 2, 3, 4, 5]:
		day_request_train.append(False)
	else:
		day_request_train.append(True)
		
day_request_test = []
for i, tt in enumerate(req_times_test):
	time_ = tt.split(" ")[1]
	if int(time_.split(":")[0]) in [22, 23, 0, 1, 2, 3, 4, 5]:
		day_request_test.append(False)
	else:
		day_request_test.append(True)

	
# Add to DFs
df_train["day_request"] = day_request_train
df_test["day_request"]  = day_request_test

# subset NA's
df_train = df_train.fillna(0)
df_test  = df_test.fillna(0)

# Select Features we want to use ----------------------------------------------
feature_names = ['Distance_1','Distance_2','Distance_3','Distance_4','Distance_5',
				 'Distance_6', 'Distance_7', 'Distance_8', 'Distance_9', 'Distance_10',
				 'Distance_11','Time_1', 'Time_2', 'Time_3', 'Time_4', 'Time_5',
				 'Time_6','Time_7','Time_8','Time_9','Time_10','Time_11','Cost_1',
				 'Cost_2', 'Cost_3','Cost_4','Cost_5','Cost_6','Cost_7','Cost_8',
				 'Cost_9','Cost_10', 'Cost_11','day_request']

# Start the process of finding a Modell----------------------------------------
# Load the SessionIDs for reproducible k-fold-CV
with open("data/processed/Test_Train_Splits/SIDs_1.txt", "rb") as fp:
	 SIDs_1 = pickle.load(fp)
	 
with open("data/processed/Test_Train_Splits/SIDs_2.txt", "rb") as fp:
	 SIDs_2 = pickle.load(fp)
	 
with open("data/processed/Test_Train_Splits/SIDs_3.txt", "rb") as fp:
	 SIDs_3 = pickle.load(fp)

with open("data/processed/Test_Train_Splits/SIDs_4.txt", "rb") as fp:
	 SIDs_4 = pickle.load(fp)
	 
with open("data/processed/Test_Train_Splits/SIDs_5.txt", "rb") as fp:
	 SIDs_5 = pickle.load(fp)

# bind it to one list!	 
SIDs = [SIDs_1, SIDs_2, SIDs_3, SIDs_4, SIDs_5]
	 

# To select rows and features of the TrainSet:
#         df_train.loc[df_train["SID"].isin(SIDs_1), 
#                      feature_names]
# --> DF with the rows in SIDs_1 and the feature_names column!


# Start Modelling -------------------------------------------------------------
# [1] First RF: ---------------------------------------------------------------
# Define its Settings:
forest = sklearn.ensemble.RandomForestClassifier(criterion='entropy',
												 n_estimators = 10, 
												 random_state = 1,
												 n_jobs = 2) # Parallelisierung
forest_1_res = []
for i in range(len(SIDs)):
	
	# print process
	print(str(i) + " / " + str(len(SIDs)))
	
	# Extract the Test_Set based on the SIDs:
	current_test          = df_train.loc[df_train["SID"].isin(SIDs[i]), feature_names]
	current_test_response = df_train.loc[df_train["SID"].isin(SIDs[i]), "Response"] 
	
	# Extract the SIDs we use for training
	train_sids = []
	for j in range(len(SIDs)):
		if j != i:
			train_sids = train_sids + SIDs[j]
			
	current_train          = df_train.loc[df_train["SID"].isin(train_sids), feature_names]
	current_train_response = df_train.loc[df_train["SID"].isin(train_sids), "Response"] 
	
	forest.fit(current_train, current_train_response)
	forest_1_res.append(sklearn.metrics.f1_score(current_test_response, 
						                         forest.predict(current_test),
									             average="weighted"))
	

# [2] Second RF_1 -------------------------------------------------------------
forest2 = sklearn.ensemble.RandomForestClassifier(criterion='entropy',
											 	  n_estimators=5, 
											      random_state=1,
												  n_jobs=2) # Parallelisierung 
forest_2_res = []
for i in range(len(SIDs)):
	
	# print process
	print(str(i) + " / " + str(len(SIDs)))
	
	# Extract the Test_Set based on the SIDs:
	current_test          = df_train.loc[df_train["SID"].isin(SIDs[i]), feature_names]
	current_test_response = df_train.loc[df_train["SID"].isin(SIDs[i]), "Response"] 
	
	# Extract the SIDs we use for training
	train_sids = []
	for j in range(len(SIDs)):
		if j != i:
			train_sids = train_sids + SIDs[j]
			
	current_train          = df_train.loc[df_train["SID"].isin(train_sids), feature_names]
	current_train_response = df_train.loc[df_train["SID"].isin(train_sids), "Response"] 
	
	forest2.fit(current_train, current_train_response)
	forest_2_res.append(sklearn.metrics.f1_score(current_test_response, 
						                         forest2.predict(current_test),
									             average="weighted"))
	
forest2.fit(df_train[feature_names], df_train["Response"])
y_preds = forest2.predict(df_test[feature_names])

a = pd.Series(df_test["SID"])
b = pd.Series(y_preds)
submission = pd.DataFrame(data={'sid':a.values, 'yhat':b.values})

import calendar
import time
ts = calendar.timegm(time.gmtime())

submission.to_csv("submissions/sub_RF" + str(ts) + ".csv", index=None, header=None)



# [3] NN ----------------------------------------------------------------------
print("nn")
# for NN scale the features 
scaler = StandardScaler()

# Define NN 1
mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 30))

MLP_res = []
for i in range(len(SIDs)):
	
	print(str(i) + " / " + str(len(SIDs)))
	
	# Extract the Test_Set based on the SIDs:
	current_test          = df_train.loc[df_train["SID"].isin(SIDs[i]), feature_names]
	current_test_response = df_train.loc[df_train["SID"].isin(SIDs[i]), "Response"] 
	
	# Scale it
	scaler.fit(current_test)
	scaled_current_test = scaler.transform(current_test)
	
	
	# Extract the SIDs we use for training
	train_sids = []
	for j in range(len(SIDs)):
		if j != i:
			train_sids = train_sids + SIDs[j]
			
	current_train          = df_train.loc[df_train["SID"].isin(train_sids), feature_names]
	current_train_response = df_train.loc[df_train["SID"].isin(train_sids), "Response"] 
	
	# Scale it
	scaler.fit(current_train)
	scaled_current_train = scaler.transform(current_train)
	
	mlp.fit(scaled_current_train, current_train_response)
	MLP_res.append(sklearn.metrics.f1_score(current_test_response, 
										    mlp.predict(scaled_current_test),
									        average="weighted"))
	


# Split the data:
train__         = df_train[feature_names]
train__response = df_train["Response"]

# Scale the features:
scaler.fit(train__)
scaled_train__ = scaler.transform(train__)

# Scale the TestFeatures!
test_features__ = df_test[feature_names]
scaler.fit(test_features__)
scaled_test_features__ = scaler.transform(test_features__)

mlp.fit(scaled_train__, train__response)
y_preds = mlp.predict(scaled_test_features__)

a = pd.Series(df_test["SID"])
b = pd.Series(y_preds)
submission = pd.DataFrame(data={'sid':a.values, 'yhat':b.values})

import calendar
import time
ts = calendar.timegm(time.gmtime())

submission.to_csv("submissions/NN_2_scaled__" + str(ts) + ".csv", index=None, header=None, )



# Define NN 2 -----------------------------------------------------------------
mlp2 = MLPClassifier(hidden_layer_sizes=(50, 40, 30, 20))

MLP_res2 = []
for i in range(len(SIDs)):
	
	
	print(str(i) + " / " + str(len(SIDs)))
	
	# Extract the Test_Set based on the SIDs:
	current_test          = df_train.loc[df_train["SID"].isin(SIDs[i]), feature_names]
	current_test_response = df_train.loc[df_train["SID"].isin(SIDs[i]), "Response"] 
	
	# Scale it
	scaler.fit(current_test)
	scaled_current_test = scaler.transform(current_test)
	
	
	# Extract the SIDs we use for training
	train_sids = []
	for j in range(len(SIDs)):
		if j != i:
			train_sids = train_sids + SIDs[j]
			
	current_train          = df_train.loc[df_train["SID"].isin(train_sids), feature_names]
	current_train_response = df_train.loc[df_train["SID"].isin(train_sids), "Response"] 
	
	# Scale it
	scaler.fit(current_train)
	scaled_current_train = scaler.transform(current_train)
	
	mlp2.fit(scaled_current_train, current_train_response)
	MLP_res2.append(sklearn.metrics.f1_score(current_test_response, 
									   	     mlp2.predict(scaled_current_test),
									         average="weighted"))
