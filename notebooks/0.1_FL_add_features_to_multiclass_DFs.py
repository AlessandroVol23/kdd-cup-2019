# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:02:05 2019

@author: kuche_000

Script to add Featues to the Multiclass Preprocessed Test and Train DF,
always based on the SID
"""
import pandas as pd
import numpy  as np

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
							 - req__night_bi
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
df_train_multi["pid"] = df_train_multi["pid"].fillna(-100)
df_test_multi["pid"]  = df_test_multi["pid"].fillna(-100)


# [2] Read in the PID specific features and add them:
#     Second row, for the missing ones, all features are 0!
PID_specific_features = pd.read_csv("data/raw/data_set_phase1/profiles.csv")
to_add                = pd.Series(list(np.repeat([-100, 0], [1, 66])), 
								  list(PID_specific_features))
PID_specific_features = PID_specific_features.append(to_add, ignore_index = True)

# Add to all 'pid' in df_test_/df_train_multi the PID specific features!
df_train_multi = pd.merge(df_train_multi, PID_specific_features, on = "pid", how = "inner")
df_test_multi = pd.merge(df_test_multi, PID_specific_features, on = "pid", how = "inner")


#%% Save the modified DFs!
df_train_multi.to_pickle("data/processed/Ready_to_Train_Test/Multiclass/Train_Set")
df_test_multi.to_pickle("data/processed/Ready_to_Train_Test/Multiclass/Test_Set")



#%% Script to use the PIDs as one-hot-encoded feature -------------------------
# -----------------------------------------------------------------Experimental
# APPROACH TO ADD ONE HOT ENCODED PIDs [memory error last time...]
def one_hot_encode(df_all, column_to_encode):
	
    # One hot encoding of the colum_to_encode
    one_hot = pd.get_dummies(df_all[column_to_encode])
	
	# Add meaningful names:
    one_hot.columns = [column_to_encode + "_" + str(col) for col in one_hot.columns]
	
	# drop old 'column_to_encode'
    df_all = df_all.drop(columns=column_to_encode, axis=0)
	
	# join the one hot encoding
    df_all = df_all.join(one_hot)
	
    return(df_all)
	
df_train_multi_one_hot_pid = one_hot_encode(df_train_multi, "pid")
df_test_multi_one_hot_pid = one_hot_encode(df_test_multi, "pid")

df_train_multi_one_hot_pid.to_pickle("data/processed/Ready_to_Train_Test/Multiclass/Train_Set_onehotPID")
df_test_multi_one_hot_pid.to_pickle("data/processed/Ready_to_Train_Test/Multiclass/Test_Set_onehotPID")