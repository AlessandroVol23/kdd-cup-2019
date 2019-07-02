# -*- coding: utf-8 -*-
"""
Script for LGBM Ranking 
Based on Dennys Script in:
	"kdd-cup-2019/notebooks/0.1_DS_lightgbm_lambdaranking.ipynb"
"""
# Laod needed Packages:
import pandas as pd
import numpy as np
import sklearn
import lightgbm as lgb
import os
import pickle
import json

# Check Working Directory:
if os.getcwd()[-12:] != "kdd-cup-2019":
	raise ValueError("Error with WorkingDirectory \n \
			[1] either WD isn't set correctly to 'kdd-cup-2019' \n \
			[2] or WD was renamed")

#%% Define needed Functions:
def load_data(path):
    """
    load the kdd cup data and preprocess it, for fitting an LGBM Ranking to it
	- sort by SIDs
	- drop duplicted possible transport modes 
	- do a reindexing
	- add a column that is always 0, except for the selected target, 
	  then it is 3 [alternativly we can set it to "1" o.s.]
	               ["0" lowest relevance to highest relevance]
	               --> just marks what was the selected 
				       row within a unique SID
	  
    Params:
        path (str) : path to the pickle file

    Return:
		DF (pandas) : Processed DF ready to train LGBM Ranking
    """
    # path to the raw pickle_file
    data = pd.read_pickle(path)

	# sort them by the sid value, so all coherently SIDs are together
    data.sort_values(by = "sid", inplace = True)
    
    # drop same transport_mode in every session (sid)
    data.drop_duplicates(["sid", "transport_mode"], inplace = True)
    
    # reindexing
    data.reset_index(drop=True, inplace=True)
    
    # assign new column target to the data
    if "click_mode" in data:
        data = data.assign(target = data.apply(lambda x: 3 if x.click_mode == x.transport_mode else 0, axis=1))
    
    return data

def create_query_file(df_r):
    """
	Based on the Data, get the amount of rows that belong together into a list
	(all rows that belong to the same query/ Session)
	[needed for LightGBM w/ lambda rank]

    Params:
		- df_r (Pandas) : dataset we want to extract the coherently rows 
		                  for the LGBM Ranking

    Return:
		numpy-array: counts of rows are assign to one sid
		             e.g. [4, 5, 12] --> first has rows 0-4, 
					                     second 5-10, 
										 third 11-23, ...
    """
	# Copy the data before the transformation
    df = df_r.copy()
	
	# Sort it by SID & reset the Index
    df = df.sort_values("sid")
    df = df.reset_index(drop = True)
	
	# Add a column "row" indicating the row in the ordered DF and count it up by 1
    df     = df.assign(row = df.index)
    df.row = df.row + 1
	
	# Get the rows, that all belong together and extract the amount of rows for them:
	# [519740    4   --> row 1 - 4 is SID 519740
	#  519751    9]  --> row 5 - 9 is SID 519751
    sid_rows           = pd.DataFrame(df.groupby("sid").last()["row"])
    sid_diff           = sid_rows.assign(difference = sid_rows.diff())
    sid_diff.iloc[0,1] = sid_diff.iloc[0,0]
	 
	# Return the amout of rows, that belong to one SID
    return sid_diff.difference.values

def get_max_from_diff(array, array_from, array_to):
    """
    get the max value from an array in an given interval of it
	Needed to extract the predicted TransModes from the array that is returned
	by the LGBM  Model

    Params: 
		- array (numpy) : An array we want to 

		- array_from (int) : Where to start

        - array_to (int) : Where to end 
		                   Attention: array_to = 10 
						              --> up to 9. Element as last element is ignored!
    Return:
		max value from the given interval in the array

    """

    array_diff = np.array(array[array_from:array_to])

    return array_diff[np.argmax(array_diff)]
"""
array = y_all_transport_mode
array_from = int(start)
array_to   = int(end)
"""
def get_max_from_diff_index(array, array_from, array_to):
    """
    Get the index of the maximum value of an array within a certain interval

    Params:
		- array (numpy) : Array we want to extract the index of the max value 
		
		- array_from (int) : index to begin
       
	    - array_to (int) : Where to end 
		                   Attention: array_to = 10 
						              --> up to 9. Element as last element is ignored!
    Return:
		index from the max value from the given interval in the array

    """
    array_diff = np.array(array[array_from:array_to])
        
    return np.argmax(array_diff)

"""
x = curr_test_set
y_all_transport_mode = predicted_scores
query_array = query_test
"""
def create_y_best_transport_mode(x, y_all_transport_mode, query_array):
    """
    Given the data used for the LGBM to calcualte the Scores of the single 
	Transportation Modes, extract for each SID the Transportation mode with
	the highest scores [assigned by the model]

    Params:
		- x (pandas) : the data we have used for the prediction/ calculation of 
		               the scores for the transportation modes for the SIDs               

        - y_all_transport_mode (numpy array) : The Scores our model predicted for 
		                                       the possible Transportmodes of all
											   SIDs in "x"!

        - query_array (numpy array) : Array with all integers that say which rows
		                              belong to the same SID
									  e.g. [4, 5, 12] --> first has rows 0 - 4, 
					                                      second has rows 5 - 10, 
										                  third has rows  11 - 23, ...
    Return:
		- numpy array : best possible transport mode for one sid
    """
    
    # Initial Parameters:
    start = 0
    end   = 0
    
	# Define Empyt lists to save SID and best Transportmode in
    y_best_transport_mode_list = []
    sid_list                   = []
    
    # Itterate through query array:
    for q in query_array:
        
        # Get the indexes of the data that belong to the same SID 
		# --> [start - end] same SID
        end = start + q
        
        # Get the Transport Mode with the highest scores in the predicted scores:
		# 	get the highest index of the current prediciton value:
        tmp1 = get_max_from_diff_index(y_all_transport_mode, int(start), int(end))
		
		#   count it up, so we can extract it from the whole DF:
        tmp0 = int(start + tmp1)
		
		# Extract the transport mode in the row with the highest predicted value
        tmp = x.iloc[tmp0,:]["transport_mode"]
		
		# Add this to the 'y_best_transport_mode_list' also add the SID for which
		# we extracted the max value to the corresponding list
        y_best_transport_mode_list.append(tmp)		
        sid_list.append(x.iloc[int(start + tmp1),:]["sid"])
        
        # set new start, so we can keep going to itterate
        start = end
            
    return pd.DataFrame(data={'sid':pd.Series(sid_list).values, 'y':pd.Series(y_best_transport_mode_list).values})

def cv_lgbm_ranking(data, features, CV, folder):
	"""
	Function to CrossValidate a lgbm-ranking model.
	Based on the CV, we can split the data to train and validation set.
	With features we select the explainable variables we want to use!
	
	Params:
		- data (pandas) : dataframe we want to use for CV
		                  the dataframe shall contain a:
								- "Respond" Column
						        - "sid"     Column 
								- "target"  Column & 
								   all variables in 'features' Argument
								   
		- featues (list) : list of strings, that shall be used as features
		                   Should always contain "transport_mode", else the 
						   Model will see features, but can not tell which is 
						   which, which will end up in terrible results!
		
		- CV (list of lists) : list with at least 2 lists.
		                       the sublists are filled with SIDs [must be in data]
							   first sublist contains the SIDs for the 1. Valdidation Set 
							   [Trained on 2.,3.,..., N. List and evaluated on 1.]
							   
   	   - folder (string)  : the folder to save the results & predictons in
	                        [on top of 'models/ranking/lgbm/']
							folder must be existent else error!
							   
   Return:
	    - save a.csv file with all predicitons of the fitted modells
		  on the corresponding CV-fold in "models/ranking/lgbm/" + folder
	
	    - save a .csv with all CV Scores in 
		  "models/ranking/lgbm/"+ 'folder'
	   - 
	"""
	# [1] Check Inputs:
	print("Check Inputs\n")
	# [1-1] Data contains "Respone", "sid" & all names passed in 'features'
	for _feat in features:
		if _feat not in data.columns.values:
			raise ValueError(_feat + "is not a columname in data")
			
	if "Response" not in data.columns.values:
		raise ValueError("data needs a 'Repsonse' Column") 
		
	if "sid" not in data.columns.values:
		raise ValueError("data needs a 'sid' Column")  
		
	if "target" not in data.columns.values:
		raise ValueError("data needs a 'target' Column")  
		
	if "click_mode" not in data.columns.values:
		raise ValueError("data needs a 'click_mode' Column")  
	
	# [1-2] CV should be of length two at least
	if len(CV) < 2: raise ValueError("CV has length < 2")
	
	# [1-3] Check whether the folder to save results is existent
	#       if it exists we count number of summaries [.csv-files] and assign 
	#       the number
	if not os.path.isdir("models/ranking/lgbm/" + str(folder)):
		raise ValueError(str(folder) + "is not existent in 'models/ranking/lgbm/'")
	else:
		numb = 0
		for _files_ in os.listdir("models/ranking/lgbm/" + str(folder)):
			if ".csv" in _files_:
				numb = numb + 1	
							
	# [2] Start doing the CV:
	# [2-1] lists to save the results of the k-fold CV for all performance measures:
	F1          = []
	Accuracy    = []
	Precision   = []
	Conf_Matrix = []	
	
	# [2-2] Loop over the different Test/Train Splits & 
	#       initalize a DF to save the predictions needed for stacking!
	print("Start CV: \n")
	res = pd.DataFrame(columns = ['M_prob_0', 'M_prob_1', 'M_prob_2', 'M_prob_3',
							      'M_prob_4', 'M_prob_5', 'M_prob_6', 'M_prob_7', 
								  'M_prob_8', 'M_prob_9', 'M_prob_10', 'M_prob_11', 
								  'fold', "SID"])
	
	for i in range(len(CV)):
	
		# Print the current Process:
		print("CV number: " + str(i + 1) + " / " + str(len(CV)))
		
		# Extract Test_Set based on the current CV:
		curr_test_set = data.loc[data["sid"].isin(CV[i]), :]
		
		# Extract SIDs we use for training & select correponding training points!
		train_sids = []
		for j in range(len(CV)):
			if j != i:
				train_sids = train_sids + CV[j]
				
		curr_train_set = data.loc[data["sid"].isin(train_sids), :]
		
		# Create Query Files [amount of rows, that belong to one SID]
		print("Extract the the rows that belong to a single Query for train | test \n")
		
		# Before we create query files, we sort the curr test/ train sets, as 
		# the query files use sorted DFs to create these values
		curr_test_set  = curr_test_set.sort_values("sid")
		curr_test_set  = curr_test_set.reset_index(drop = True)
		curr_train_set = curr_train_set.sort_values("sid")
		curr_train_set = curr_train_set.reset_index(drop = True)
		
		# now create the query files!
		query_train = create_query_file(curr_train_set)
		query_test  = create_query_file(curr_test_set)		
		
		print("Select features train | test \n")
		train_features = curr_train_set[features]
		train_response = curr_train_set["target"]
		test_features  = curr_test_set[features]
		test_response  = curr_test_set.drop_duplicates("sid").click_mode
	    
	    # set the parameters for the LGBM-Ranking Model
		print("Configure Parameters \n")
		params                  = {}
		params['learning_rate'] = 0.003
		params['boosting_type'] = 'gbdt'
		params['objective']     = 'lambdarank'
	    
	    # Transform the TrainData into an LGBM-Dataset & Train the LGBM Model
		print("Fit LightGBM Model \n")
		lgb_train = lgb.Dataset(train_features, train_response, group = query_train)
		gbm       = lgb.train(params, lgb_train)
	    
	    # predict the Scores for each TransMode in the TestSet
		print("LightGBM prediction on testset \n")
		predicted_scores = gbm.predict(test_features)
		
		# Based on the predicted Scores for each TransportMode, extract the 
		# predicted class, we need for.
		predicted_classes = create_y_best_transport_mode(curr_test_set,
												         predicted_scores, 
														 query_test)

		# Add Scores to the corresponding lists:
		F1.append(sklearn.metrics.f1_score(test_response, 
									       predicted_classes["y"],
										   average="weighted"))
		
		Accuracy.append(sklearn.metrics.accuracy_score(test_response, 
									                   predicted_classes["y"]))
		
		Precision.append(sklearn.metrics.recall_score(test_response, 
									                  predicted_classes["y"],
									                  average = "weighted"))
		
		Conf_Matrix.append(sklearn.metrics.confusion_matrix(test_response, 
									                        predicted_classes["y"]))
		
		# Create a List, with the prediction scores for each SID [0 if not avaible]!
		start = 0
		for _end in query_test:
			
			_end = int(_end)
			_end = start + _end
			
			# Extract the possible TransportModes and the corresponding SID
			poss_trans_modes = curr_test_set.iloc[start:_end,:]["transport_mode"].values
			SID              = curr_test_set.iloc[start:_end,:]["sid"].values[0]
			
			# Define "0" list and colnames, we will add to the prediction file later!
			cols_to_add = list(np.repeat(0, 14))
			
			# Fill in the predicted Scores for the possible Transport Modes:
			for index, poss_modes in enumerate(poss_trans_modes):
				cols_to_add[poss_modes] = predicted_scores[start + index]
			
			# Add the current fold
			cols_to_add[12] = (i + 1)
			cols_to_add[13] = SID
			
			# Bind the List to the result DF!
			res.loc[len(res)] = cols_to_add
			
			start = _end
	
	# Save the predicted Class Probabilities:	
	res.to_csv("models/ranking/lgbm/" + str(folder) + "/CV_Predicitons" + str(numb) + ".csv")

	print("CV done\n")
	# [3] Save the Results:
	# [3-1] Extract ParameterSettings
	_params = json.dumps(gbm.params)
	
	# [3-2] Create BasicShape for the Result .csv
	dict_to_pd = {'model_type' : "LamdaRank", 'parameters' : _params,
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
			
	pd_res.to_csv("models/ranking/lgbm/" + str(folder) + "/Summary" + str(numb) + ".csv")

#%% Run the CV
if __name__ == "__main__":
	
	# Load the Data:
	print("Load the Data\n")
	with open('data/processed/features/multiclass_1.pickle', 'rb') as ff:
		feature_names = pickle.load(ff)
		
	feature_names.append("transport_mode")
	
	with open("data/processed/split_test_train/5-fold/SIDs.pickle", "rb") as fp:
		CV = pickle.load(fp)
	
	data       = load_data("data/processed/Ranking/train_all_row.pickle")
	
	# TO BE DELETED
	data_ = pd.read_pickle("C:/Users/kuche_000/Desktop/train_all_processed.pickle")
	data_  = data_.sort_values("sid")
	data_  = data_.reset_index(drop = True)
	data = data_.head(1000)
	
	cv_lgbm_ranking(data     = data, 
				    features = feature_names, 
				    CV       = CV,
				    folder   = "1")