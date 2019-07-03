import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import sys
import click
from time import gmtime, strftime
import os
import utils
import pickle
import sklearn
from sklearn.metrics import f1_score


# Check Working Directory & throw error if wrong:
if "kdd-cup-2019" not in ''.join(list(os.getcwd())[-12:]):
	raise ValueError("Either you are running the script from the wrong directory \
				      and not from '../kdd-cup-2019' or \
					  you renamed the repository")

#%% Load the SIDs for CV
print("load the splits needed for CV")
with open("data/processed/Test_Train_Splits/5-fold/SIDs.pickle", "rb") as fp:
	SIDs = pickle.load(fp)	 

#%% Define Functions 
print("Define Functions")

def submit_result(submit, result, model_name):
    print('Saving submit...')
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    submit['recommend_mode'] = result
    submit.to_csv(
        'submissions/multiclass_lgb/{}_result_{}.csv'.format(model_name, now_time), index=False)

    print('Saved submit at {}'.format(model_name))
 
   
def save_preds(sids, preds, path):
    df = pd.DataFrame(preds, index=sids, columns=['p0','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11'])
    df.to_csv(path)
    
# Function for downsampling
def downsample(df, mode, amount):
    df_just_mode = df.loc[df.click_mode == mode]  
    
    df_mode_target = df_just_mode.sample(amount, replace=True)
    df = df.loc[df.click_mode != mode]
    df = pd.concat([df_mode_target, df], axis=0)
    
    return df

# Function for upsampling
def upsample(df, mode, amount):
    df_just_mode =df.loc[df.click_mode == mode]  
    
    df_mode_target = df_just_mode.sample(amount, replace=True)
    df = df.loc[df.click_mode != mode]
    df = pd.concat([df_mode_target, df], axis=0)
    return df
 
# Function needed for early stopping in LGBM Training:
def eval_f(y_pred, train_data):
    y_true = train_data.label
    y_pred = y_pred.reshape((12, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True

# Function to get CV-Scores for lgbm multiclass 
def lgbm_train(df_train, features, CV, folder, 
			   lr = 0.05, num_leaves = 60, downsample_mode = 99, 
			   downsample_amount = 99, upsample_mode_1 = 99, 
			   upsample_1_amount = 99, upsample_mode_2 = 99, 
			   upsample_2_amount = 99, upsample_mode_3 = 99, 
			   upsample_3_amount = 99, upsample_mode_4 = 99, 
			   upsample_4_amount = 99, upsample_mode_5 = 99,
			   upsample_5_amount = 99):
	"""
	Function to for CrossValidating the LGBM Multiclass Models
	
	Args:
		- df_train (pandas) : Data we use for Cross Validation
							  the dataframe shall contain a:
										- "Respond" Column
										- "sid"     Column & 
									       all variables in 'features' Argument
										   
		- features (list)  : List of strings, indicating which columns in 
		                     df_train shall be used as features
							 
		- CV (list of lists) : list of lists filled with sids used for 
		                       Test and Trainsplit
		
		- folder (string) : Name of the folder in "models/lgb_multi",
		                    where the model results shall be saved
					        IF not existent --> Error
								 
	    - lr (float > 0)       : learning rate of the model
		- num_leaves (int > 0) : How many leaves shall the model use
		- downsample_mode (int) : the mode we want to downsample 
		                          (99 if we don't want downsampling)
		- downsample_amount (int) : to how many shall the mode be downsampled
		
		GENERAL INFO FOR "upsample_X_amount" / "upsample_mode_X"
		
		- upsample_mode_X (int)    : if 99, no upsampling, will happen, 
		                             if we pass an integer, the corresponding 
									 mode willbe upsampled by upsample_X_amount
									 
		- upsample_X_amount (int)  : to how many shall the integer be upsampled?
		                             [will be ignored if corresponding 
									 'upsample_mode_X' is not 99]
									 
	 Return:
		 - 
 	"""
	 
	# [1] Check Inputs:
		# [1-1] Data contains "Respond", "sid" & all names passed in 'features'
	for _feat in features:
		if _feat not in df_train.columns.values:
			raise ValueError(_feat + "is not a columname in df_train")
			
	if "Response" not in df_train.columns.values:
		raise ValueError("df_train needs a 'Repsonse' Column") 
		
	if "sid" not in df_train.columns.values:
		raise ValueError("df_train needs a 'sid' Column")  
		
		# [1-2] CV should be of length two at least
	if len(CV) < 2: raise ValueError("CV has length < 2")
	
		# [1-3] Check whether the folder to save results is existent
		#       if it exists we count number of summaries [.csv-files] and  
		#       assign the number
	if not os.path.isdir("models/lgb_multi/" + str(folder)):
		raise ValueError(str(folder) + "is not existent in 'models/lgb_multi/'")
	else:
		numb = 0
		for _files_ in os.listdir("models\lgb_multi/" + str(folder)):
			if ".csv" in _files_:
				numb = numb + 1	

	# [2] Start CV
	print("Start to train light gbm model")

	# [2-1] Copy the original TrainData 
	#       [as we mitght change it with up/down sampling]
	data = df_train.copy()

	# [2-2] Set the Parameters for the LGBM-Model
	#       [some avaible as Argument for the function]
	lgb_paras = {
        'objective': 'multiclass',
        'metrics': 'multiclass',
        'learning_rate':  lr,
        'num_leaves': num_leaves,
        'lambda_l1': 0.01,
        'lambda_l2': 10,
        'num_class': 12,
        'seed': 2019,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4
    }
	
	# [2-3] Define empty lists to save results in & start the CrossValidation 
	F1           = []
	Accuracy     = []
	Precision    = []
	Conf_Matrix  = []
	res          = pd.DataFrame()
	
    # Loop over the different Splits
	for i in range(len(CV)):
        
        # Print current process
		print(str(i + 1) + " / " + str(len(CV)))

        # Extract the Test_Set based on the current SID:
		val_x     = data.loc[data["sid"].isin(CV[i]), features].values
		val_y     = data.loc[data["sid"].isin(CV[i]), "Response"].values
		val_index = data.loc[data["sid"].isin(CV[i]), "sid"]

        # Extract the SIDs we use for training, and select correponding train points!
		train_sids = []
		for j in range(len(CV)):
			if j != i:
				train_sids = train_sids + CV[j]
                
		df_train_split = data.loc[data["sid"].isin(train_sids), :]
        
		# Downsampling: If downsample_mode != 99, we upsample the 'downsample_mode'
		#               to 'downsample_amount'
		if downsample_mode != 99:
			print("Downsample mode {} on {}".format(downsample_mode, downsample_amount))
			df_train_split = downsample(df_train_split, downsample_mode, downsample_amount)
		else:
			print("No Downsampling")
        
		# Upsampling: For each 'upsample_mode_X' that is != 00, we upsample the
		#             mode corrsponing mode to the passed integer by "upsample_X_amount"
		# [1]
		if upsample_mode_1 != 99:
			print("Upsample mode 1 is {} on {}".format(upsample_mode_1, upsample_1_amount))
			df_train_split = upsample(df_train_split, upsample_mode_1, upsample_1_amount)
		else:
			print("Mode 1, no upsampling")
        # [2]  
		if upsample_mode_2 != 99:
			print("upsample_mode_2  is {} on {}".format(upsample_mode_2, upsample_2_amount))
			df_train_split = upsample(df_train_split, upsample_mode_2, upsample_2_amount)
		else:
			print("Mode 2, no upsampling")
        # [3] 
		if upsample_mode_3 != 99:
			print("upsample_mode_3  is {} on {}".format(upsample_mode_3, upsample_3_amount))
			df_train_split = upsample(df_train_split, upsample_mode_3, upsample_3_amount)
		else:
			print("Mode 3, no upsampling")
        # [4]   
		if upsample_mode_4 != 99:
			print("upsample_mode_4  is {} on {}".format(upsample_mode_4, upsample_4_amount))
			df_train_split = upsample(df_train_split, upsample_mode_4, upsample_4_amount)
		else:
			print("Mode 4, no upsampling")
        # [5] 
		if upsample_mode_5 != 99:
			print("upsample_mode_5  is {} on {}".format(upsample_mode_5, upsample_5_amount))
			df_train_split = upsample(df_train_split, upsample_mode_5, upsample_5_amount)
		else:
			print("Mode 5, no upsampling")
            
		# Get Information of the Distribution of the click modes in the current
		# train set
		print('Current Splits in the current TrainSet')
		print(df_train_split.groupby('Response').count()['sid'])
		          
		# Extract the features again, as the sampling might have changed the 
		# original distribution
		tr_x = df_train_split[features].values
		tr_y = df_train_split['Response'].values

		# Convert it into a format fitting for LGBM Multiclass
		train_set = lgb.Dataset(tr_x, tr_y)
		val_set   = lgb.Dataset(val_x, val_y)
		
		# Now we train the model, save it, evaluate its performance and save 
		# its predicitons [used for stacking later on]
        # Train the model on the current split
		lgb_model = lgb.train(lgb_paras, train_set,
                              valid_sets = [val_set], 
							  early_stopping_rounds = 50,
                              num_boost_round = 40000,
							  verbose_eval = 50, 
							  feval = eval_f)

        # Predict on best iteration of this split for the out of fold observations
		# get the predicted probabilites and use them to extract the single prediciton
		val_probs = lgb_model.predict(val_x, num_iteration = lgb_model.best_iteration)
		val_pred  = np.argmax(val_probs, axis = 1)

        # Add Scores to the corresponding lists:
		F1.append(sklearn.metrics.f1_score(val_y, val_pred,
										   average="weighted"))
		
		Accuracy.append(sklearn.metrics.accuracy_score(val_y, val_pred))
		
		Precision.append(sklearn.metrics.recall_score(val_y, val_pred,
									                  average = "weighted"))
		
		Conf_Matrix.append(sklearn.metrics.confusion_matrix(val_y, val_pred))

		
		# Put the models predicitons into an PD-DF [for stacking]
		colstoadd = np.append(val_probs, np.repeat((i+1), 
						      len(val_probs)).reshape(len(val_probs), 1), 1)
		 
		colnames = ["M_prob_" + str(_i) for _i in range(12)]
		colnames.append("fold")
		
		res = pd.concat([res, pd.DataFrame(colstoadd,
								  columns = colnames, 
								  index = val_index)])
	
	print("CV done\n")
	
	# Now save the Predictied Class Scores
	res.to_csv("models/lgb_multi/" + folder + "/Predicitons_" + str(numb) + ".csv")
	
	# Save the Performance Measures of the model!
	# Extract ParameterSettings + add down and upsampling settings
	_params = lgb_model.params
	_params["downsample"] = [downsample_mode, downsample_amount]
	_params["upsample_mode_1"] = [upsample_mode_1, upsample_1_amount]
	_params["upsample_mode_2"] = [upsample_mode_2, upsample_2_amount]
	_params["upsample_mode_3"] = [upsample_mode_3, upsample_3_amount]
	_params["upsample_mode_4"] = [upsample_mode_4, upsample_4_amount]
	_params["upsample_mode_5"] = [upsample_mode_5, upsample_5_amount]
	
	# [3-2] Create BasicShape for the Result .csv
	dict_to_pd = {'model_type' : "LGBM-Multi", 'parameters' : _params,
			      'features' : " - ".join(features), "Name": numb}
	
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
			
	pd_res.to_csv("models/lgb_multi/" + folder + "/Summary_" + str(numb) + ".csv")
	
	print("---DONE---")

"""	
"Try it out"
df_train = pd.read_pickle("data/processed/multiclass/train_all_first.pickle")
df_train = df_train.iloc[:10000]

with open ("data/processed/features/multiclass_1.pickle", 'rb') as fp:
	features = pickle.load(fp)

CV = SIDs

folder = "1"

lgbm_train(df_train = df_train, 
		   features = features, 
		   CV = SIDs, 
		   folder = "1", 
		   lr = 0.05, 
		   num_leaves = 60, 
		   downsample_mode = False, downsample_amount = 99, 
		   upsample_mode_1 = 0, upsample_1_amount = 1000, 
		   upsample_mode_2 = 99, upsample_2_amount = 99, 
		   upsample_mode_3 = 99, upsample_3_amount = 99, 
		   upsample_mode_4 = 99, upsample_4_amount = 99, 
		   upsample_mode_5 = 99, upsample_5_amount = 99)
"""
df_train = pd.read_pickle("data/processed/multiclass/train_all_first.pickle")

with open ("data/processed/features/multiclass_1.pickle", 'rb') as fp:
	features = pickle.load(fp)


for _lr in [0.001, 0.05, 0.1]:
	for _num_leaves in [10, 25, 50, 100]:
		
		lgbm_train(df_train = df_train, 
		   features = features, 
		   CV = SIDs, 
		   folder = "1", 
		   lr = 0.05, 
		   num_leaves = 60, 
		   downsample_mode = 99, downsample_amount = 99, 
		   upsample_mode_1 = 99, upsample_1_amount = 99, 
		   upsample_mode_2 = 99, upsample_2_amount = 99, 
		   upsample_mode_3 = 99, upsample_3_amount = 99, 
		   upsample_mode_4 = 99, upsample_4_amount = 99, 
		   upsample_mode_5 = 99, upsample_5_amount = 99)
		
		lgbm_train(df_train = df_train, 
		   features = features, 
		   CV = SIDs, 
		   folder = "1", 
		   lr = 0.05, 
		   num_leaves = 60, 
		   downsample_mode = 2, downsample_amount = 70000, 
		   upsample_mode_1 = 4, upsample_1_amount = 20000, 
		   upsample_mode_2 = 5, upsample_2_amount = 20000, 
		   upsample_mode_3 = 8, upsample_3_amount = 3000, 
		   upsample_mode_4 = 10, upsample_4_amount = 20000, 
		   upsample_mode_5 = 99, upsample_5_amount = 99)