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
		
	
	
Make sure, that your workingdirectory is on: '..\kdd-cup-2019'
"""

# Load all Packages needed
import pandas as pd
import numpy as np
import os
import sklearn
import sklearn.tree
import sklearn.ensemble
import sklearn.metrics as metrics
import pickle
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import json

# Check Working Directory:
if "kdd-cup-2019" not in os.getcwd():
	raise ValueError("Your Working Directory is not set correctly")

#%% Read in Data, the SIDs for CV & Select Features
# [1] Data we train the model with (features were added seperatly]
df_train     = pd.read_pickle("data/processed/Ready_to_Train_Test/Multiclass/Train_Set")
df_test      = pd.read_pickle("data/processed/Ready_to_Train_Test/Multiclass/Test_Set")

# [2] SessionIDs for reproducible k-fold-CV:
SIDs = []
for cv in range(1, 6):
	with open("data/processed/Test_Train_Splits/5-fold/SIDs_" + str(cv) + ".txt", "rb") as fp:
		 SIDs.append(pickle.load(fp))	 
	
# [3] Names of the features we want to use 
#     [need to be in colnames of df_test/ df_train]
Distances = ['Distance_{}'.format(i) for i in range(1, 12)]
Costs     = ['Cost_{}'.format(i) for i in range(1, 12)]
Time      = ['Time_{}'.format(i) for i in range(1, 12)]
svd_feas  = ['svd_fea_{}'.format(i) for i in range(20)]
time_feas = ['req_weekend','req_evening_bi', 'req_night_bi', 'req_day_bi'] # still more
pid_feas  = ['p{}'.format(i) for i in range(66)]

# Add the list of names we want to add:
feature_names = Distances + Costs + Time + svd_feas + time_feas

#%% Define Function needed for Modelling and Submitting
# [1] Define a Function, we pass our Train DF, our CV-SIDs and a Model  
#     which returns a list with the F1-Scores on each fold:

def get_cv_score(data, features, model, CV, scaled = False):
	"""
	A Function to calculate the [F-1]-CV-Score of a Model.
	
	Args:
		- data (pandas)   = dataframe with layout for multiclass learning 
		                    [s. _FL_Data_to_Multiclass_PreProcess]
						    DF shall contain a "Respond" Column
						                     a "SID"     Column 
						       		     &   all [passed] features
	   - features (list)  = list of all feature_names, that shall be used for
	                        modelling [NEED TO BE IN data, else ERROR]
						   
	   - model (sklearn)  = A Model for Multiclassification
	                        ParameterSetting done before passing as argument
	   - CV (list)        = A list filled with lists of length k (amount of test_train splits)
	                        The lists inside the CV-List should be of equal length
						    and only contain SesssionIDs 
	   - scaled (boolean) = Shall the Train and Test Data be scaled
	                        (important for NN)
    Return:
		- list of length k, with all F1-Scores!
	"""          
	
	# Define list to save results of the k-fold CV
	Res = []
	
	# Loop over the different Test/Train Splits
	for i in range(len(SIDs)):
	
		# Print process
		print(str(i) + " / " + str(len(SIDs) - 1))
		
		# Extract the Test_Set based on the current SID:
		current_test          = data.loc[data["SID"].isin(SIDs[i]), features]
		current_test_response = data.loc[data["SID"].isin(SIDs[i]), "Response"] 
		
		# Extract the SIDs we use for training, and select correponding train points!
		train_sids = []
		for j in range(len(SIDs)):
			if j != i:
				train_sids = train_sids + SIDs[j]
				
		current_train          = data.loc[data["SID"].isin(train_sids), features]
		current_train_response = data.loc[data["SID"].isin(train_sids), "Response"] 
		
		# Feature Scaling (only if "scaled" = True)
		if scaled:
			scaler = StandardScaler()
			
			scaler.fit(current_test)
			current_test = scaler.transform(current_test)
			
			scaler.fit(current_train)
			current_train = scaler.transform(current_train)
			
		# Fit the Model and add the F1-Score to 'Res'
		model.fit(current_train, current_train_response)
		Res.append(sklearn.metrics.f1_score(current_test_response, 
									        model.predict(current_test),
										    average="weighted"))
		
	return Res

"""
Example of Use:
forest = sklearn.ensemble.RandomForestClassifier(criterion='entropy',
												 n_estimators = 10, 
												 random_state = 1,
												 n_jobs = 2) # Parallelisierung

forest_res = get_cv_score(data = df_train, features = feature_names, 
			              model = forest, CV = SIDs, scaled = False)
"""
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# [2] Run over different Parameters and save the results as pickle-DF:
def track_cv_results(model, model_cv_res, folder, features, model_type):
	"""
	Function to track the Results of the 'get_cv_score' Function
	Every Single CV-Result will be saved as '.csv' in folder!
	
	Args:
		- model (skelarn)    : a model, we want to save
		- model_cv_res (list): CV Results of the corresponding Model
		- features (list)    : list of features that were used to fit the model
		- folder (str)       : A folder in './reports/Model_Results' 
	    - model_type (str)   : What Modeltype (NeuralNet, RandomForrest, ...)
							  
	Return:
		- A .csv-File, with all Informations to the current test
             each CV-Score, the mean of the CV-Scores, the feature_names,
             and the type of the model [extra Arg: 'model_type']					        
	"""
	# [1] Get the Parameters of the Models [as. character]
	_params = json.dumps(model.get_params())
	
	# [2] Check whether the Folder is existent already and throw error if not
	#     If existent, check how many files inside, so we can Index the files
	if not os.path.isdir("reports/Model_Results/Multiclass/" + str(folder)):
		raise ValueError("'folder'is not existent in 'reports/Model_Results/'")
	else:
		file_number = len(os.listdir("reports/Model_Results/Multiclass/" + str(folder)))
		
	# [3] Create Dict to save them to a Pandas:
	dict_to_pd = {'model_type' : model_type, 'parameters' : _params,
			      'features' : "-/".join(features)}
	
	# add the CV Results to the Dict:
	for index, cv_curr in enumerate(model_cv_res):
		dict_to_pd["CV_" + str(index)] = cv_curr
	
	dict_to_pd["CV_mean"] = np.mean(model_cv_res)
	
	pd_res = pd.DataFrame([dict_to_pd])
	
	pd_res.to_csv("reports/Model_Results/Multiclass/" + str(folder) + "/" + str(file_number) + ".csv")


"""
Example of Use:
	
knn_curr = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 2, n_jobs = 2)
		
knn_curr_res = get_cv_score(data = df_train, features = feature_names, 
	                        model = knn_curr, CV = SIDs, scaled = True)

track_cv_results(model  = knn_curr, 
		         model_cv_res = knn_curr_res, 
		         folder = "knn_PID_svd", 
				 features = feature_names, 
				 model_type = "KNN")
"""

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# [3] Get Predicitons for the Test_Set File ready to submitt on Baiduu
def predict_test_data(model, train_data, test_data, features, scaled = False):
	"""
	Function to train a model, do predictions on the test set and 
	giving back a 'pandas' of Form:    SID | Prediction
	                                   123 |     2
									   321 |     9
			           --> easy to save as .csv for submission
									 
	 Args:
		 - model (sklearn)      : A model we want to train on the train_data
		                          & do predicitons for test_data
				  
	     - train_data (pandas) : The DF we use for Training, must contain
		                         column "Response" & all the columnnames in 
								 'features'
								 
		 - test_data (pandas)  : The DF we want to do predicitons on.
		                         Must contain all all the columnnames in 
								 'features' and a Column called "SID"
								 
		 - features (list)    : list of strings, with the names of the features
		                        we want to use for modelling
		
		- scaled (boolean)   : Shall the feature values be scaled?
		
	 Return:
		 (pandas)-DF in the correct layout, that just needs to be saved as .csv
		                                                [pd.to_csv("...")]
	"""
	
	# Feature Scaling (only if "scaled" = True)
	if scaled:
		scaler = StandardScaler()
		
		scaler.fit(train_data[feature_names])
		train_data[feature_names] = scaler.transform(train_data[feature_names])
		
		scaler.fit(test_data[feature_names])
		test_data[feature_names] = scaler.transform(test_data[feature_names])
	
	model.fit(train_data[feature_names], train_data["Response"])
	y_preds = model.predict(test_data[feature_names])
	
	_SID_         = pd.Series(test_data["SID"])
	_predicitons_ = pd.Series(y_preds)
	
	submission = pd.DataFrame(data={'sid':_SID_.values, 'yhat':_predicitons_.values})
	
	return submission

"""
Example of Use:
	
subfile = predict_test_data(model      = mlp, 
			    	        train_data = df_train, 
				            test_data  = df_test, 
				            features   = feature_names, 
				            scaled     = True)

subfile.to_csv("submissions/NN_w_pid_features_MLPClassifier_40_25_20.csv", 
			   index = None, header = None)
"""

#%% Start creating Modells: 
"""
Basically select a model, its parameters, folder where it shall be saved & 

"""
if __name__ == '__main__':
	
	for k in [0.001, 0.01, 0.1, 1, 10, 100]:
		print("Current k: " + str(k))
		
		log_curr = sklearn.linear_model.LogisticRegression(penalty='l2', C = k)
		
		
		log_curr_res = get_cv_score(data = df_train, features = feature_names, 
			                        model = log_curr, CV = SIDs, scaled = True)
		
		track_cv_results(model        = log_curr, 
				         model_cv_res = log_curr_res, 
				         folder       = "_log_reg", 
						 features     = feature_names, 
						 model_type   = "log_reg")
	
# Create a Submission for the best Submitted File
subfile = predict_test_data(model      = mlp, 
			    	        train_data = df_train, 
				            test_data  = df_test, 
				            features   = feature_names, 
				            scaled     = False)

# subfile.to_csv("submissions/RF_w_PID_fea20_criterion=entropy_n_estimators=20.csv", 
# 			   index = None, header = None)