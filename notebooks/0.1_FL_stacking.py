# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:14:24 2019

@author: kuche_000

Script for Stacking

Basic Idea:
	let a couple of predictors do predictions on our transportation modes
	--> then we train a model to predict the correct transport mode only on
	    these predicitons --> MetaLearner
		
	--> good Intro on kaggle itself:
'http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/'

This File is the basis for the Stacking Script!
"""
# Load all Packages needed
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import sklearn
import sklearn.tree
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import f1_score
import sklearn.ensemble
import sklearn.metrics as metrics
import pickle
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import json

# Check Working Directory:
if os.getcwd()[-12:] != "kdd-cup-2019":
	raise ValueError("Error with WorkingDirectory \n \
			[1] either WD isn't set correctly to 'kdd-cup-2019' \n \
			[2] or WD was renamed")
	
#%% Functions
def create_stacked_data(dataframes, additional_features = []):
	"""
	Function to merge different dataframes with the SIDs and the models
	class_predictions/ predicited_class_probabilities.
	Additional we add the true Response from the processed data 
	"data/processed/multiclass/with_SVD_20/train_all_first.pickle"-DF!
	
	Optional we can add other features as "additional_features",
	if we want to train the meta learner not only on the predicted classes!
	
	Args:
		- dataframes (list) : list of dataframes!
		                      all dataframes, need a column with "sid" and
							  at least one column with the prediciton for the
							  corresponding class / class probability!
							  All DFs should be of equal length!
	   
	    - additonal_features (list) : list of strings we want to use as 
		                              additional features!
									  All of them need to be in the 
			  "data/processed/multiclass/with_SVD_20/train_all_first.pickle"-DF
				  
			  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
			  TO DO ASA THE FOLDER STRUCUTRE IS FIXED AND THE TRAIN DATA
			  ARE READY 
		      !!! AS THE FOLDER STRUCTURE IS STILL CHANGING WE MIGHT NEED TO 
		      ADJUST THE FOLDERs, WHERE WE LOAD THE Data from !!!
	         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	Return:
		- a pandas DF, with all selected metafeatures and a additional column
		  "Respond"
	    - list of features in the DF, that are not "sid" or "response", so 
		  basically the feature names used for training the stacked model!
	"""
	# [1] Check Inputs
	# [1 - 1] All passed dataframes of same length
	df_lengths = [len(item) for item in dataframes]
	if not any([df_lens == df_lengths[0] for df_lens in df_lengths]):
		raise ValueError("passed dataframes are not of the same lengths!")
	
	# [1 - 2] Do all dataframes have a "sid" column
	if any(["sid" not in list(item) for item in dataframes]):
		raise ValueError("not all dataframes have a 'sid' column")
		
	# [1 - 3] TO-DO Check whether all have used the same CV-Splits
	#         should be done when predictions have a "fold" column
	
	# [2] Merge the DFs and add [optional] additional features
	# load the raw train data, keep the response and the sid + all optional
	# selected features
	df_train_raw = pd.read_pickle("data/processed/multiclass/with_SVD_20/train_all_first.pickle")
	
	# Create the Basis for the merged DF, by choosing SID, response and any 
	# additional features of one of the trainsets + rename clickmode to "Response"
	df_meta = df_train_raw[["sid", "click_mode"] + additional_features]
	df_meta.columns.values[1] = "Response"
	
	# Merge all dataframes to df_meta [bases on 'sid'] 
	for i, _data in enumerate(dataframes):		
		df_meta = pd.merge(df_meta, _data, on = 'sid', 
					 suffixes=('_' + str(i) + '_1', '_' + str(i) + '_2'))
		
	# Get all feature names in the DF except for 'sid' & 'response':
	features_in_df_meta = list(df_meta)[3:]
	
	return(df_meta, features_in_df_meta)
	
	
def get_CV_stack(data, stacked_model, features_stack = [], CV = SIDs, 
				 scale_features_stack = False):
	"""
	Create an Stacking Learner!
	For this we will use the predicitons of submodels on the CV-folds and use
	these as "Meta"-features for the metamodel.
	--> Very Important:   The Predicitons of the submodels are based on the 
	                      same splits as we use now, to CV our stacked Model!

	As we do CV, the prediciton of the submodels are independent of the 
	features in the corresponding row, as these were not seen by the model 
	for its prediction.
	-> INDEPENDENCE between the features of row i and the target value of row i 
   
	Args:
		- data (pandas) : dataframe with layout for multiclass learning 
		                  DF shall contain   a "Respond" Column,
						                     a "sid"     Column,
						       		     &   all 'features_stack'
						  features_stack in this case are the predictions of
						  the submodells the stacked model builds on!
		                     
		- stacked_model (sklearn) : model, that we use as stacked model
		                            [multiclass model!]
									must have: - predict &
									           - fit method
									
		-  CV (list)  : A list filled with lists of length k [amount of folds]
		                Each list containing ~equal amount of SIDs we use for
						resmapling
    			 
	    - features_stack (list) : list of features that the StackedModel, 
		                          shall use for predicitons
								  
	    - scale_features_stack(boolean) : shall the data for the stacked model 
	                                      be normalized
										 
	Return:
		- create a .csv file for the CV-Performance Measures and save it in 
		  "models/stacking"
	"""
	
	# [1] Input Check
	# [1 - 1] all features_stack variables in data
	for _feat in features_stack:
		if _feat not in data.columns.values:
			raise ValueError(_feat + " is not a colname in data")
			
	# [1 - 2] data contains a "sid" & a "Response" column
	if "Response" not in data.columns.values:
		raise ValueError("data needs a 'Repsonse' Column") 
		
	if "sid" not in data.columns.values:
		raise ValueError("data needs a 'sid' Column")
		
	# [1 - 3] all SIDs passed in CV are in the data
	# flat the CV lists so its easier to compare:
	flat_cv = [item for sublist in CV for item in sublist]
	
	# get the intersections of "flat_cv" & "data["sid"]"
	intersection1 = set(flat_cv).intersection(data["sid"])
	intersection2 = set(data["sid"]).intersection(flat_cv)
	
	if len(intersection1) != len(intersection2):
		raise ValueError("SIDs are not the same in data and CV")
	
	# [1-4] Check whether the  "models/stacking" to save results is existent
	if not os.path.isdir("models/stacking"):
		raise ValueError("'models/stacking' is not existent in 'models/'")
	
    # [2] Fit the Stacked Model on the predicted outcomes of the different
	#     submodels & optional addtional features!
	print("Calculating the CV Score for the stacked Model \n")
	
	# Define list to save results of the k-fold CV!
	F1          = []
	Accuracy    = []
	Precision   = []
	Conf_Matrix = []	
	
	# Loop over the different Test/Train Splits
	for i in range(len(CV)):
	
		# Print process
		print(str(i + 1) + " / " + str(len(CV)))
		
		# Extract the Test_Set based on the current SID:
		current_test          = data.loc[data["sid"].isin(CV[i]), features_stack]
		current_test_response = data.loc[data["sid"].isin(CV[i]), "Response"] 
		
		# Extract the SIDs we use for training, and select correponding train points!
		train_sids = []
		for j_ in range(len(CV)):
			if j_ != i:
				train_sids = train_sids + CV[j_]
				
		current_train          = data.loc[data["sid"].isin(train_sids), features_stack]
		current_train_response = data.loc[data["sid"].isin(train_sids), "Response"] 
		
		# Feature Scaling (only if "scaled" = True)
		if scale_features_stack:
			scaler = StandardScaler()
			
			scaler.fit(current_test)
			current_test = scaler.transform(current_test)
			
			scaler.fit(current_train)
			current_train = scaler.transform(current_train)
			
		# Fit the Model and add the F1-Score to 'Res'
		stacked_model.fit(current_train, current_train_response)
		predictions = stacked_model.predict(current_test)
		
		# Get the Scores:
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
	
	print("CV done\n")
	# [3] Save the Results:
	# [3-1] Extract ParameterSettings
	_params = json.dumps(stacked_model.get_params())
	
	# [3-2] look into the folder and count how many stacked models we have  
	#       there already and assign a corresponding number
	numb = 0
	for _files_ in os.listdir("models/stacking/"):
		if ".csv" in _files_:
			numb = numb + 1	
			
	dict_to_pd = {'model_type' : "stacked", 'parameters' : _params,
			      'features' : " - ".join(features_stack), "Number": numb}
	
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
			
	pd_res.to_csv("models/stacking/Summary" + str(numb) + ".csv")
	
	# [4] Train the final model [on all train data] and save it:
	print("train the final model")
	stacked_model.fit(data.loc[:, features_stack], data["Response"])
	
	final_model_name = "models/stacking/Final_Mod" + str(numb) + ".pickle"
	pickle.dump(stacked_model, open(final_model_name, 'wb'))

# %% Train a stacked model- minimalistic example
# [1] Load the Folds for CV
SIDs = []
for cv in range(1, 6):
	with open("data/processed/Test_Train_Splits/5-fold/SIDs_" + str(cv) + ".txt", "rb") as fp:
		 SIDs.append(pickle.load(fp))
	 
# [2] Load predicitons of the submodels and merge them to one big DF:
xgb_df = pd.read_csv("models/Multivariate Approach/merged_dfs/xgboost/CV_Predicitons12.csv")
knn    = pd.read_csv("models/Multivariate Approach/merged_dfs/knn/CV_Predicitons0.csv")

# paste them into a single list
dataframes = [knn, xgb_df]

# Create DF to train stacked modell with:
data, features = create_stacked_data(dataframes, [])

# fast check, whether we lost any data
if data.shape[0] != len(dataframes[0]):
	raise ValueError("Unkown Error happend, merged data has more rows as original data")

# Train a stacked model on the metadata:
get_CV_stack(data = data,
			 stacked_model = sklearn.ensemble.RandomForestClassifier(criterion = 'entropy', 
														n_estimators = 25,
														random_state = 1, 
														n_jobs = 2),
			 features_stack = features, 
			 CV = SIDs,
			 scale_features_stack = True)