# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:02:05 2019

@author: kuche_000

Script to add Featues to the Multiclass Preprocessed Test and Train DF,
always based on the SID
"""
import pandas as pd
import numpy  as np
from sklearn.decomposition import TruncatedSVD

#%% Read in Data
# [1] Load the Train & Testdata for multiclass (where we want to add features!)
df_train_multi     = pd.read_pickle("data/processed/Multiclass_Approach/data_trans/MulticlassTrainSet")
df_test_multi      = pd.read_pickle("data/processed/Multiclass_Approach/data_trans/MulticlassTestSet")

# Fill the "na" entrances
df_train_multi = df_train_multi.fillna(0)
df_test_multi  = df_test_multi.fillna(0)

# [2] load the data we use to add the features:
df_train_clean = pd.read_pickle("data/processed/Multiclass_Approach/data_trans/train_no_double_TransModes_per_sid")
df_test_clean  = pd.read_pickle("data/processed/Multiclass_Approach/data_trans/test_no_double_TransModes_per_sid")

#%% Add temporal Features! [req-time, req-weekend, ...]
def time_features(dataf, type = 'req'):
    """
	Add 5 new time features
	
	Args:
		dataf (pandas) = dataframe we want to add temporal features to 
		                 [DF needs an "req_time" / "plan_time" column]
	    type (string)  = the time_column we want to use for feature extraction
		                 ["req" or "plan", as "click_time" is only avaible for Train]
						 
	 Return:
		 dataf (pandas) = Same DF, but with 5 extra columns:
							 - req_date
							 - req_month
							 - req_hour
							 - req_weekend (1 if yes)
							 - req_night_bi
							 - req_day_bi
    """

    if type not in ['req', 'plan']:
	    raise ValueError("Wrong 'time' type: Should be 'req' or 'plan'")
	
    if type + "_time" not in list(dataf):
        raise ValueError(type + "_time" + "is no column name in the passed DF")
		
		
	# Change Type of selected 'type'-column
    dataf[type + '_time'] = pd.to_datetime(dataf[type + '_time'])


	# Add new features, based on the ColumnType:
	# [1]  Add the Hour, the Day and the Month:
    dataf[type  + '_date'] = dataf[type + '_time'].dt.strftime('%m-%d')
    dataf[[type + '_month',type + '_day']] = dataf[type + '_date'].str.split("-",expand=True,).astype(int)
    dataf = dataf.drop(type + '_date', axis=1)
    dataf[type + '_hour'] = dataf[type + '_time'].dt.hour
	
	# [2] Add a Binary, indicating whether its weekend or not! [ERROR]
    dataf[type + '_weekend'] = dataf[type + '_time'].dt.day_name().apply(lambda x: 1 if x in ["Friday", "Saturday"] else 0)
	
	# Add binary, NIGHT / EVENING / DAY Column
    dataf[type + '_night_bi'] = dataf[type + '_hour'].apply(lambda x: 1 if x <= 7 else 0)
    dataf[type + '_day_bi']   = dataf[type + '_hour'].apply(lambda x: 1 if x in range(8,18) else 0)
    dataf[type + '_evening_bi']   = dataf[type + '_hour'].apply(lambda x: 1 if x > 18 else 0)

    return dataf


# Add Time Features to df_train_/ df_test_multi:
# [1] Add a TimeColumn to the multiclasstrain DFs
df_train_multi["req_time"] = df_train_clean.drop_duplicates("sid")["req_time"].values
df_test_multi["req_time"]  = df_test_clean.drop_duplicates("sid")["req_time"].values	

# [2] Use the function to extract features from the time column:
df_train_multi = time_features(df_train_multi, type = 'req')
df_test_multi  = time_features(df_test_multi,  type = 'req')

#%% Add PIDs, and the corresponding 65 PID specific binary features:
# [1] Add the raw PIDs to df_train_/df_test_multi:
df_train_multi["pid"] = df_train_clean.drop_duplicates("sid")["pid"].values
df_test_multi["pid"]  = df_test_clean.drop_duplicates("sid")["pid"].values

# subset the PID with "-100" for the cases, where it is missing (easier to process then)
df_train_multi["pid"] = df_train_multi["pid"].fillna(0)
df_test_multi["pid"]  = df_test_multi["pid"].fillna(0)


# [2] Read in the PID specific features and add them:
#     Second row, for the missing ones, all features are 0!
PID_specific_features = pd.read_csv("data/raw/data_set_phase1/profiles.csv")
to_add                = pd.Series(list(np.repeat([0, 0], [1, 66])), 
								  list(PID_specific_features))
PID_specific_features = PID_specific_features.append(to_add, ignore_index = True)

# Add to all 'pid' in df_test_/df_train_multi the PID specific features!
df_train_multi = pd.merge(df_train_multi, PID_specific_features, on = "pid", how = "left")
df_test_multi  = pd.merge(df_test_multi,  PID_specific_features, on = "pid", how = "left")

#%% Add decomposed PID Features 
def gen_profile_feas(data, k):
	"""
	Add decomposed PID-specific features to data
	
	Args:
		- data (pandas) : DF we want to add the k-dimensional main components
                          of the decomposed PID-specific features 
						  Needs a Column named "pid" [with the Personal IDs] 
                          [these PID features are loaded automaitcally from
                           'data/raw/data_set_phase1/profiles.csv']
		- k (integer)   : Amount of dimensions we decompose our data to

	Res:
		- data (pandas) : same but with 'k' extra columns:
						   "svd_fea_1", ..., "svd_fea_k"
						   corresponding values of the k main comoponents
	"""
	# Inputcheck for data:
	if "pid" not in list(data):
		raise ValueError("Data doesn't have a 'pid' column")
			
	# read in the profile data and add a "0"-Row
	profile_data = pd.read_csv("data/raw/data_set_phase1/profiles.csv")
	to_add       = pd.Series(list(np.repeat([0, 0], [1, 66])), 
				   	         list(PID_specific_features))
	profile_data = profile_data.append(to_add, ignore_index = True)

	# subset all PID specific features:
	x = profile_data.drop(['pid'], axis=1).values

	# check whether k is meaningful	[smaller  #binary_cols in profiles]
	if k >= x.shape[1]: raise ValueError("k needs to be smaller than ", str(x.shape[1]))
	
	# linear dimensionality reduction by means of truncated 
	# singular value decomposition (SVD)
	svd = TruncatedSVD(n_components = k, n_iter = 20, random_state = 2019)
	svd_x = svd.fit_transform(x)
	
	# Save the decomposed PID features in DF
	svd_feas = pd.DataFrame(svd_x)
	svd_feas.columns = ['svd_fea_{}'.format(i) for i in range(k)]
	
	# add pid to svd_feas, so we can merge it to the original DF!
	svd_feas['pid'] = profile_data['pid'].values
	
    # data['pid'] = data['pid'].fillna(-1) # no na subsetted above
	
	data = pd.merge(data, svd_feas, on='pid', how='left')
	return data

# Add the 20 main components of PIDs to our Test and Traindata
df_train_multi = gen_profile_feas(data = df_train_multi, k  = 20)
df_test_multi  = gen_profile_feas(data = df_test_multi,  k  = 20)


#%% Add Distance to closest Subway Station!
#!!! Need to get external data!!!
###############################################################################
###################### TO BE ##################################################
###################### DONE ###################################################
###############################################################################
#%% Save the modified DFs!
if df_train_multi.isnull().values.any():
	raise ValueError("df_train_multi should not contain NAs")

if df_test_multi.isnull().values.any():
	raise ValueError("df_test_multi should not contain NAs")
df_train_multi.to_pickle("data/processed/Ready_to_Train_Test/Multiclass/Train_Set")
df_test_multi.to_pickle("data/processed/Ready_to_Train_Test/Multiclass/Test_Set")