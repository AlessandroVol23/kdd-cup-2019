# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:48:56 2019

@author: kuche_000

Add decomposed PID features
"""
import pandas as pd
import numpy  as np
from sklearn.decomposition import TruncatedSVD

# Load DF
df_train = pd.read_pickle("data/processed_raw/train_raw_first.pickle")
df_test  = pd.read_pickle("data/processed_raw/test_raw_first.pickle")

# Fill the "na" entrances
df_train = df_train.fillna(0)
df_test  = df_test.fillna(0)


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
							 list(profile_data))
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
df_train = gen_profile_feas(data = df_train, k = 20)
df_test  = gen_profile_feas(data = df_test,  k = 20)

df_train.to_pickle("data/processed_raw/with_SVD_20/Train_Set")
df_test.to_pickle("data/processed_raw/with_SVD_20/Test_Set")
