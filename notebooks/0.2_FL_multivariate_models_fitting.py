# -*- coding: utf-8 -*-
"""
First Multiclass Models for KDD, based on the data in:
	"data/processed/multiclass"
	
	- TrainSet is used for CV and training the model
	- TestSet is unkown an we need to submit predictions for it!
	
	--> all features we use must be in the train and testset! 
	
Make sure your workingdirectory is on: '..\kdd-cup-2019' else error
"""
# Load all Packages needed
import pandas as pd
import numpy as np
import os
import sklearn
import sklearn.tree
import xgboost as xgb
from sklearn import preprocessing
import sklearn.ensemble
import sklearn.metrics as metrics
import pickle
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import json
import click

# Check Working Directory:
if os.getcwd()[-12:] != "kdd-cup-2019":
	raise ValueError("Error with WorkingDirectory \n \
			[1] either WD isn't set correctly to 'kdd-cup-2019' \n \
			[2] or WD was renamed")

#%% Read in Data, the SIDs for CV & Select Features
print("Load SIDs, FeatureNames, Train- and TestData")
#  [1] Load Train and TestData ------------------------------------------------
DF_new_train = pd.read_pickle("data/processed/multiclass/train_all_first.pickle")
DF_new_test  = pd.read_pickle("data/processed/multiclass/Desktop/test_all_first.pickle")

# [2] Load SessionIDs for reproducible k-fold-CV ------------------------------
with open("data/processed/Test_Train_Splits/5-fold/SIDs.pickle", "rb") as fp:
	SIDs = pickle.load(fp)	 
	
# [3] Load FeatureNames -------------------------------------------------------
with open('data/processed/features/multiclass_1.pickle', 'rb') as ff:
	feature_names = pickle.load(ff)
	
print("Features being used: \n" + str(feature_names) + "\n")
#%% Define Functions for Calculating the CV-Score of a sklearn model & create a 
#   file for subsmitting a (sklearn) model.
print("Define functions\n")
# [1] Function to calculate the CV-Score for a Model, save the results and save
#     all submodell that were created.
def get_cv_and_modells(data, features, model, CV, scaled, folder, model_type):
	"""
	A Function to calculate the CV-Score of a Model for different performace 
	measures [F-1, Accuracy, ConfusionMatrix, Precision], save the submodells 
	and create a final model fitted on whole data! All Results, Submodels and 
    the Finalmodel are save in: "models/Multivariate Approach/"+ 'folder'
	
	Args:
		- data (pandas)   : dataframe for a multiclass model learner.
						    the dataframe shall contain a:
								- "Respond" Column
						        - "sid"     Column & 
								  all variables in 'features' Argument

	   - features (list)  : list of strings, indicating the features that shall
	                        be used for modelling.
							If any feature in the list is not in data this will
							produce an error
						   
	   - model (sklearn)  : model for multiclass-prediciton.
						    pass the model with the params set already.
	                        e.g.:
							model = MLPClassifier(hidden_layer_sizes=(1, 2, 3))
							
							need methods ".fit", ".predict" & ".get_params"
							
	   - CV (list)        : list of length k (amount of test_train splits), 
						    where each entrance is a list of the 'sids' in the 
							traindata, indicating wich observations to use for 
							training & testing.
							
	   - scaled (boolean) : Shall the Train- & Testdata be scaled
	                        [via sklearn.preprocessing --> StandardScaler]
							
	   - folder (string)  : the folder to save the results/ submodels in
	                        [on top of 'models/Multivariate Approach/']
							
	   - model_type (string) : type of the model [added in the summary file]
	                         
    Return:
		- save a.csv file with all predicitons of the fitted modells
		  on the corresponding CV-fold
		
		- save a final model, that was trained on all traindatapoints, in 
		  "models/Multivariate Approach/"+ 'folder'
		  
	    - save a .csv with all CV Scores in 
		  "models/Multivariate Approach/"+ 'folder'
	"""    

	# [1] Check Inputs:
	# [1-1] Data contains "Respond", "sid" & all names passed in 'features'
	for _feat in features:
		if _feat not in data.columns.values:
			raise ValueError(_feat + "is not a columname in data")
			
	if "Response" not in data.columns.values:
		raise ValueError("data needs a 'Repsonse' Column") 
		
	if "sid" not in data.columns.values:
		raise ValueError("data needs a 'sid' Column")  
	
	# [1-2] Check whether the model has the methods needed:
	if not hasattr(model, 'fit'): 
		raise ValueError("model has no 'fit'-method")
		
	if not hasattr(model, 'predict'): 
		raise ValueError("model has no 'predict'-method")
		
	if not hasattr(model, 'get_params'): 
		raise ValueError("model has no 'get_params'-method")
	
	# [1-3] CV should be of length two at least
	if len(CV) < 2: raise ValueError("CV has length < 2")
	
	# [1-4] Check whether the folder to save results is existent
	#       if it exists we count number of summaries [.csv-files] and assign 
	#       the number
	if not os.path.isdir("models/Multivariate Approach/" + str(folder)):
		raise ValueError(str(folder) + "is not existent in 'models/Multivariate Approach/'")
	else:
		numb = 0
		for _files_ in os.listdir("models/Multivariate Approach/" + str(folder)):
			if ".csv" in _files_:
				numb = numb + 1	
	
	# [2] Start doing the CV:
	# [2-1] lists to save the results of the k-fold CV for all performance measures:
	F1          = []
	Accuracy    = []
	Precision   = []
	Conf_Matrix = []	
	
	# [2-2] Loop over the different Test/Train Splits
	print("Start CV: \n")
	
	# Initalize a DF to save the predictions needed for stacking!
	res = pd.DataFrame()
	for i in range(len(CV)):
	
		# Print the current Process:
		print("CV number: " + str(i + 1) + " / " + str(len(CV)))
		
		# Extract Test_Set based on the current CV:
		current_test          = data.loc[data["sid"].isin(CV[i]), features]
		current_test_response = data.loc[data["sid"].isin(CV[i]), "Response"] 
		current_test_index    = data.loc[data["sid"].isin(CV[i]), "sid"]
		
		# Extract SIDs we use for training & select correponding training points!
		train_sids = []
		for j in range(len(CV)):
			if j != i:
				train_sids = train_sids + CV[j]
				
		current_train          = data.loc[data["sid"].isin(train_sids), features]
		current_train_response = data.loc[data["sid"].isin(train_sids), "Response"] 
		
		# Feature Scaling (only if "scaled" = True)
		if scaled:
			scaler = StandardScaler()
			
			scaler.fit(current_test)
			current_test = scaler.transform(current_test)
			
			scaler.fit(current_train)
			current_train = scaler.transform(current_train)
			
		# Fit the Model and get predcitions of the testset
		model.fit(current_train, current_train_response)
		predictions = model.predict(current_test)
		
		# Add Scores to the corresponding lists:
		F1.append(sklearn.metrics.f1_score(current_test_response, 
									       predictions,
										   average="weighted"))
		
		Accuracy.append(sklearn.metrics.accuracy_score(current_test_response, 
									                   predictions))
		
		Precision.append(sklearn.metrics.recall_score(current_test_response, 
							                          predictions,
									                  average = "weighted"))
		
		Conf_Matrix.append(sklearn.metrics.confusion_matrix(current_test_response, 
								                            predictions))
		
		
		# If predicting probabilities works, we add them as metafeatures to
		# our metafile, else we will use only the predicted class & save it!
		# Add also the fold of the current TestSIDs
		if hasattr(model, 'predict_proba'):
			predictions_prob = model.predict_proba(current_test)
			
			# join the predicitons w/ the current fold number
			colstoadd = np.append(predictions_prob,
						          np.repeat((i+1), len(predictions_prob)).reshape(len(predictions_prob), 1), 1)
			 
			colnames = ["M_prob_" + str(_i) for _i in range(12)]
			colnames.append("fold")
			
			res = pd.concat([res, pd.DataFrame(colstoadd,
									  columns = colnames, 
									  index = current_test_index)])
	
		else:
			colnames = ["M_1", "fold"]
			
			# join the predicitons w/ the current fold number
			colstoadd = np.column_stack((predictions, 
						            np.repeat((i+1), len(predictions)).reshape(len(predictions), 1)))
			
			res = pd.concat([res, pd.DataFrame(colstoadd,
									  columns = colnames,
									  index = current_test_index)])
		
	res.to_csv("models/Multivariate Approach/" + str(folder) + "/CV_Predicitons" + str(numb) + ".csv")
		
	
	print("CV done\n")
	# [3] Save the Results:
	# [3-1] Extract ParameterSettings
	_params = json.dumps(model.get_params())
	
	# [3-2] Create BasicShape for the Result .csv
	dict_to_pd = {'model_type' : model_type, 'parameters' : _params,
			      'features' : " - ".join(features), "Number": numb}
	
	# [3-3] Add CV-Scores to the Dict:
	for index in range(len(F1)):
		dict_to_pd["F1_" + str(index + 1)]        = F1[index]
		dict_to_pd["Acc_" + str(index + 1)]       = Accuracy[index]
		dict_to_pd["Precision_" + str(index + 1)] = Precision[index]
		dict_to_pd["Conf_" + str(index + 1)]      = Conf_Matrix[index]
		
	# [3-4] Add mean of the scores [except for Confmatrix]
	dict_to_pd["F1_mean"]   = np.mean(F1)
	dict_to_pd["Acc_mean"]  = np.mean(Accuracy)
	dict_to_pd["Prec_mean"] = np.mean(Precision)
	
	# [3-5] Transform it to pandas, order the columns and save it: 
	pd_res = pd.DataFrame([dict_to_pd])
			
	pd_res.to_csv("models/Multivariate Approach/" + str(folder) + "/Summary" + str(numb) + ".csv")
	
	
	# [4] Train the final model [on all train data] and save it:
	print("train the final model")
	model.fit(data.loc[:, features], data["Response"])
	
	final_model_name = "models/Multivariate Approach/"+ folder + "/Final_Mod" + str(numb) + ".pickle"
	pickle.dump(model, open(final_model_name, 'wb'))
	

"""
get_cv_and_modells(data = df_train_merged.iloc[:1000], 
				   features = ['req_evening', 'is_holiday', 'max_temp'], 
				   model =  MLPClassifier(hidden_layer_sizes=(2, 2, 3)), 
				   CV = SIDs, 
				   scaled = False, 
				   folder ='Preprocessed_Raw_PID_20_SVD/TEST', 
				   model_type = "NeuralNet")
"""

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# [2] Get Predicitons for the Test_Set File ready to submitt on Baiduu
def get_subfile(model, train_data, test_data, features, scaled, fitted):
	"""
	Function to to get the submission file for a model, doing predictons
	on the test_data and is learned on "train_data"
	Return a 'pandas'-DF of Form:      SID | Prediction
	                                   123 |     2
									   321 |     9
									 
	 Args:
		 - model (sklearn)      :  model for multiclass-prediciton, we want to
		                           train on the trainset and do predicitons on
								   the testset.
								   need methods ".fit", ".predict" & ".get_params"
								   pass the model with the params set already.
								   e.g.:
							       model = MLPClassifier(hidden_layer_sizes=(1, 2, 3))

	     - train_data (pandas)  : The DF we use for Training, must contain
		                          column "Response" & all the columnnames in 
								  'features'
								 
		 - test_data (pandas)  : The DF we want to do predicitons on.
		                         Must contain all all the columnnames in 
								 'features' and a Column called "sid"
								 
		 - features (list)     : list of strings, with the names of the features
		                         we want to use for modelling
		
		- scaled (boolean)     : Shall the feature values be scaled?
		
		- fitted (boolean)     : Was the model fitted already?
								 No need to retrain then!
		
	 Return:
		 (pandas)-DF in the correct layout for submission  [needs to be saved only]
	"""
	# [1] Check Inputs:
	# [1-1] Data contains "Respond", "sid" & all names passed in 'features'
	for _feat in features:
		if _feat not in train_data.columns.values:
			raise ValueError(_feat + "is not a columname in train_data")
			
	if "Response" not in train_data.columns.values:
		raise ValueError("train_data needs a 'Response' Column") 
		
	if "sid" not in train_data.columns.values:
		raise ValueError("train_data needs a 'sid' Column")  
		
	for _feat in features:
		if _feat not in test_data.columns.values:
			raise ValueError(_feat + "is not a columname in test_data")
			
	if "sid" not in test_data.columns.values:
		raise ValueError("test_data needs a 'sid' Column") 
	
	# [1-2] Check whether the model has the methods needed:
	if not hasattr(model, 'fit'): 
		raise ValueError("model has no 'fit'-method")
		
	if not hasattr(model, 'predict'): 
		raise ValueError("model has no 'predict'-method")
	
	print("Start creating submission file")
	# [2] Create Subfile
	# [2-1] Scale the data if wanted:
	if scaled:
		scaler = StandardScaler()
		
		scaler.fit(train_data[features])
		train_data[features] = scaler.transform(train_data[features])
		
		scaler.fit(test_data[features])
		test_data[features] = scaler.transform(test_data[features])

	# [2-2] Fit the model, if the passed model wasn't fit yet:
	if not fitted:
		model.fit(train_data[features], train_data["Response"])
	
	# [2-3] Get predicitions of the fitted model
	y_preds = model.predict(test_data[features])
	
	# [2-4] transform it into a pandas DF in the correct layout
	_SID_         = pd.Series(test_data["sid"])
	_predicitons_ = pd.Series(y_preds)
	
	submission = pd.DataFrame(data={'sid':_SID_.values, 'yhat':_predicitons_.values})
	
	print("submission file created")
	return submission

"""
Example:
subfile = get_subfile(model      = MLPClassifier(hidden_layer_sizes=(2, 2, 3)), 
					  train_data = df_train.iloc[:1000], 
				      test_data  = df_test.iloc[:1000], 
				      features   = ['max_temp', 'min_temp', 'wind'], 
				      scaled     = False,
					  fitted     = False)

# fitted can also be loaded and "fitted" can be set to "True" 

subfile.to_csv("submissions/TEST_SUBMISSION.csv", index = None, header = None)
"""

#%% Start creating Modells: 
"""
Select a model and set its parameters and pass it to "get_cv_and_modells" to
get its CV-Scores, its submodells and a final model trained on all train pts!
"""

if __name__ == '__main__':
	
	for k in [5, 10, 15, 50, 100, 200, 500, 1000]:
		print("Current k: " + str(k))
		
		knn_curr = sklearn.neighbors.KNeighborsClassifier(n_neighbors = k,
													n_jobs = 10)
		
		get_cv_and_modells(data       = df_train_merged, 
				           features   = feature_names,
				           model      = knn_curr, 
						   CV         = SIDs, 
						   scaled     = True,
						   folder     = '1/XGBoost', 
						   model_type = "KNN")