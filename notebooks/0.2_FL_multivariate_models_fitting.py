# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:35:37 2019

@author: kuche_000
"""

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
# Data we train the model with [features should be added in a seperate script]
df_train     = pd.read_pickle("data/processed/Ready_to_Train_Test/Multiclass/Train_Set")
df_test      = pd.read_pickle("data/processed/Ready_to_Train_Test/Multiclass/Test_Set")

# Should not contain any NAs
df_train.isnull().sum().sum()
df_test.isnull().sum().sum()


# Select Features we want to use ----------------------------------------------
# should be in list(df_test) & list(df_train)
feature_names = ['Distance_1','Distance_2','Distance_3','Distance_4','Distance_5',
				 'Distance_6', 'Distance_7', 'Distance_8', 'Distance_9', 'Distance_10',
				 'Distance_11','Time_1', 'Time_2', 'Time_3', 'Time_4', 'Time_5',
				 'Time_6','Time_7','Time_8','Time_9','Time_10','Time_11','Cost_1',
				 'Cost_2', 'Cost_3','Cost_4','Cost_5','Cost_6','Cost_7','Cost_8',
				 'Cost_9','Cost_10', 'Cost_11', 'req_weekend','req_evening_bi',
				  'req_night_bi', 'req_day_bi', 
				  'p0','p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9',
				  'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 
				  'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25',
				  'p26', 'p27', 'p28', 'p29', 'p30', 'p31', 'p32', 'p33',
				  'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p40', 'p41',
				  'p42', 'p43', 'p44', 'p45', 'p46', 'p47', 'p48', 'p49',
				  'p50', 'p51', 'p52', 'p53', 'p54', 'p55', 'p56', 'p57', 
				  'p58', 'p59', 'p60', 'p61', 'p62', 'p63', 'p64', 'p65']

# Start the process of finding a Modell----------------------------------------
# Load the SessionIDs for reproducible k-fold-CV and bind it in one list:
with open("data/processed/Test_Train_Splits/5-fold/SIDs_1.txt", "rb") as fp:
	 SIDs_1 = pickle.load(fp)	 
with open("data/processed/Test_Train_Splits/5-fold/SIDs_2.txt", "rb") as fp:
	 SIDs_2 = pickle.load(fp)	 
with open("data/processed/Test_Train_Splits/5-fold/SIDs_3.txt", "rb") as fp:
	 SIDs_3 = pickle.load(fp)
with open("data/processed/Test_Train_Splits/5-fold/SIDs_4.txt", "rb") as fp:
	 SIDs_4 = pickle.load(fp)	 
with open("data/processed/Test_Train_Splits/5-fold/SIDs_5.txt", "rb") as fp:
	 SIDs_5 = pickle.load(fp)
	 
SIDs = [SIDs_1, SIDs_2, SIDs_3, SIDs_4, SIDs_5]

# Start Modelling -------------------------------------------------------------
# Define a Function, we pass our Train DF, our CV-SIDs & a Model, and that 
# returns a list with the scores on each fold:
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
	
# [1] First RF: ---------------------------------------------------------------
forest = sklearn.ensemble.RandomForestClassifier(criterion='entropy',
												 n_estimators = 10, 
												 random_state = 1,
												 n_jobs = 2) # Parallelisierung

forest_res = get_cv_score(data = df_train, features = feature_names, 
			              model = forest, CV = SIDs, scaled = False)

# [2] Second RF ---------------------------------------------------------------
forest2 = sklearn.ensemble.RandomForestClassifier(criterion='entropy',
											 	  n_estimators=5, 
											      random_state=1,
												  n_jobs=2) # Parallelisierung 

forest_res2 = get_cv_score(data = df_train, features = feature_names, 
			               model = forest2, CV = SIDs, scaled = False)

# [3] NN1 ---------------------------------------------------------------------
mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 30))

mlp_res = get_cv_score(data = df_train, features = feature_names, 
			           model = mlp, CV = SIDs, scaled = True)

# [4] NN2----------------------------------------------------------------------
mlp2 = MLPClassifier(hidden_layer_sizes=(50, 40, 30, 20))

mlp_res2 = get_cv_score(data = df_train, features = feature_names, 
			            model = mlp2, CV = SIDs, scaled = True)

# [5] Sample Multiple Trees----------------------------------------------------
forrrest_res_all = []
for deppness in range(1, 21):
	
	print("State: " + str(deppness) + "/" + str(20))
	
	forest = sklearn.ensemble.RandomForestClassifier(criterion='entropy',
												 n_estimators = deppness, 
												 random_state = 1,
												 n_jobs = 2)
	
	forest_res_current = []
	forest_res_current = get_cv_score(data = df_train, features = feature_names, 
						              model = forest, CV = SIDs, scaled = False)
	
	forrrest_res_all.append(forest_res_current)
	
	_name = "criterion=entropy,n_estimators=" + str(deppness) + ",random_state = 1,n_jobs = 2"
	
	with open("data/processed/Multiclass_Approach/model_results/random_forrest/" +_name + ".txt", "wb") as fp:
		pickle.dump(forrrest_res_all, fp)

# Create A DF from it:	
forrest_results = pd.DataFrame({'5_fold_cv_results': forrrest_res_all,
								'n_estimator': range(1, 21)})
	
# [6] Sample Multiple NN-------------------------------------------------------
NN_net_all = []
for First_Layer in range(50, 60, 5):
	
	print("First Layer Size: " + str(First_Layer))
	
	mlp = MLPClassifier(hidden_layer_sizes=(First_Layer, 25, 20))

	mlp_res_current = get_cv_score(data = df_train, features = feature_names, 
			                       model = mlp, CV = SIDs, scaled = True)
	
	
	NN_net_all.append(mlp_res_current)
	
	_name = "MLPClassifier(hidden_layer_sizes=(" + str(First_Layer) + ", 25, 20, 5))" 
	
	with open("data/processed/Multiclass_Approach/model_results/nn/" +_name + ".txt", "wb") as fp:
		pickle.dump(mlp_res_current, fp)


# Create a DF from it:	
nn_results = pd.DataFrame({'5_fold_cv_results': NN_net_all,
								'first layer deppness': range(10, 60, 5)})
	
# -----------------------------------------------------------------------------
# Get Predicitons for the Test_Set and a .csv File ready to submitt -----------
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
		 pandas file  with
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


# Exmaple to for the function based on the model with the best CV-Score!
mlp = MLPClassifier(hidden_layer_sizes=(55, 25, 20))

subfile = predict_test_data(model = mlp, 
			    	        train_data = df_train, 
				            test_data  = df_test, 
				            features   = feature_names, 
				            scaled = True)

# save as ".csv" we can upload on Baidu
subfile.to_csv("submissions/sub_NN_Scaled_Features_55_First_Layer.csv", index=None, header=None)
