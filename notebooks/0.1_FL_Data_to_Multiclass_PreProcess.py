# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:34:45 2019

@author: kuche_000


# Describtion of the Script: --------------------------------------------------
File to convert the preprocessed DF [#15, Sandro] {'data/processed/15_to_one_df'}
{merged the different DFs to a big one for Test and Train}
into a DF that can be used for Multiclass-Classification

# Expected Data Structure
Data:
	- Each Session-ID {=one request} is listed in as many rows, as it has 
	  possible transportation modes 
	    [if a request has 5 possible transportation modes, its 5 rows, each
		row for one transportation mode & its specific information]
	- There can be as many additional features inside, but these will be added
	  to the multiclass_DF later on then!
	
	- Test and Train have the same structure. 
	  Only Differenc: Test_Set has no Colum "Click_Mode" & "Click_Time"
	                                        [= Response]   [= Time clicked]
					  --> "Click_Time" can not be used as feature!

- First Step:
	- For some Sessions, the same transportation mode is offered multiple times
	- In these Cases the ones offered later will be removed:
	  [!did some invetigation, ones offered later worse in price or/and time!]
		e.g. [SID, ..., transport_mode, Distance, ...] 
		      123, ...,     1,             1200
	          123, ...,     11,             1100
	          123, ...,     1,             1300
			  
		  --> [SID, ..., transport_mode, Distance, ...] 
		      123, ...,     1,             1200
	          123, ...,     11,             1100

	>>> Results are saved as "test_no_double_TransModes_per_sid" {analog for train}
	    in the "data/preprocessed/Multiclass_Approach"
		[Folder Structure checked at beginning of Code]
		
- Second Step:
	- Transform the data in a way, that each row corresponds to one session
    - Each Column corresponds to the transportation mode specific 
	  Distance/ Time/ Cost [3 attributes * 11 possible Transport Modes = 33]
	  --> Return a DF with each Session in a row and 33 Columns:
	   e.g. [SID, ..., transport_mode, Distance, ...] 
	         123, ...,     1,             1200
	         123, ...,     11,            1100
		 
		  --> [SID, Distance_1, ..., Distance_11, Cost_1, ..., Cost_11, Time_1, ..., Time_11]
		       123,    1200   , ...,     1100   , ... {analog}
	  
    - Only Difference between Test and Train:
		- Test will not hold a "Respond" Column, Train will hold one!
"""
# SET YOUR WORKING DIRECTORY TO: "kkd-cip-2019", to avoid folder problems...

# Load the needed Packages ----------------------------------------------------
import pandas as pd
import os


# INPUT CHECKING --------------------------------------------------------------
# Folder Structure:
# check folder structure to avoird errors, raise error, for unsuited folder structures
if not os.path.isdir("data/processed/15_to_one_df"):
	raise Exception("there is not folder structure 'data/processed/15_to_one_df', where the processed data shall lie")

if not os.path.isdir("data/processed/Multiclass_Approach"):
	raise Exception("there is not folder structure 'data/processed/Multiclass_Approach' where we'll save the results of this script")

if not os.path.isdir("data/processed/Multiclass_Approach/SubDFs"):
	raise Exception("there is not folder structure 'data/processed/Multiclass_Approach/SubDFs' where we'll save temporary results")


# LOADING DATA ----------------------------------------------------------------
# Load the preprocessed DF [#15, Sandro]:
# Choose Train- or Test-Set or both
df_train = pd.read_pickle("data/processed/15_to_one_df/df_train.pickle")
df_test  = pd.read_pickle("data/processed/15_to_one_df/df_test.pickle")


# Start the Preprocessing -----------------------------------------------------
# (1) ------------------------------------------------------------------------- 
"""
Clean the DF:
	- each session should only offer one possibility for each transportation
	- if there are twice possibilities for the same transportation mode 
	  [e.g. twice possible connections with transportation mode "1"] 
	  only keep the first one!
	- checked couple of these secenarios, the ones offerd later, always 
	  worse in time or/and money... [reasonable sorted already]
"""

# Remove Rows, that offer an second alternative of a TransMode in the same SID:
df_train_clean = df_train.drop_duplicates(["sid", "transport_mode"])
df_test_clean  = df_test.drop_duplicates(["sid", "transport_mode"])

# Save the created Results
df_train_clean.to_pickle("data/processed/Multiclass_Approach/train_no_double_TransModes_per_sid")
df_test_clean.to_pickle("data/processed/Multiclass_Approach/test_no_double_TransModes_per_sid")


# (2) -------------------------------------------------------------------------
"""
Create DF to train Multiclass with:
	- Each row one Session-ID
	- to each Session ID the transport specific cost/ time/ distance in a row
	  [cost_1, ..., cost_11 -> cost for each transportation mode in an own col]
    - Non offered transport modes "NaN" [can be replaced later on]
	- resulting DF will have 34 Columns [33 from TransMode Specific Propertys 1 + SID]
"""

# Function to extract Information fromk data_clean [created above^]
def create_multiclass_df(data_clean, indeces, DF_name):
	"""
	Function to extraxt the transport specific distances/ costs/ times.
	The Resulting DF(s) will be saved in:
		 "data/processed/Multiclass_Approach/SubDFs/DF_" + DF_name
    --> Might be useful to do it in multiple itterations!
	
	Args:
	    - data_clean (pandas) : the DF we want to extract the information from
		                        needs the columns: - eta
								                   - price
												   - sid
												   - transport_mode
				                all other columns can easily be added later
								 
		- indeces (list)      : list of integers with all session IDs we want to 
		                        extract [need to be in "df_clean"]
						   
		- DF_name (str)       : Addtion to the Name the DF shall be safed in
		                       [e.g. "Test_2" --> "DF_" + "Test_2" --> "DF_Test_2"]
						   
    Return:
		- DF (pandas)    : Dataframe with a column for each transpotationmode
		                   specific Distance/Time/Cost:
						   [Distance_1, ..., Distance_11, Time_1, ...,Time_11,
						   Cost_1, ..., Cost_11, SID] 
						   
						   Folder where it will be saved:
						   "data/processed/Multiclass_Approach/SubDFs/DF_" + DF_name
	"""
	
	# Input Check -------------------------------------------------------------
	# [1] Check for needed Colnames:
	_needed_col_names = ["sid", "eta", "distance_plan", "price"]
	for _needed in _needed_col_names:
		if not _needed in list(data_clean):
			raise Exception(_needed + " is missing as colum in data_clean Argument")
			
			
	# [2] Check whether all passed indices are in data_clean
	# can be done faster!
	# for index in indeces:
	# 	if not index in data_clean["sid"].unique():
	# 		raise Exception(str(index) + " is not as SID in data_clean")
	
	
	# Create the DF, fill it with values and return it ------------------------
	# Create empty DF:
	df_ = pd.DataFrame(columns = ["Distance_1", "Distance_2", "Distance_3",
				  "Distance_4", "Distance_5", "Distance_6", "Distance_7", 
				  "Distance_8", "Distance_9", "Distance_10", "Distance_11",
				  "Time_1", "Time_2", "Time_3", "Time_4", "Time_5", "Time_6", 
				  "Time_7", "Time_8", "Time_9", "Time_10", "Time_11",
				  "Cost_1", "Cost_2", "Cost_3", "Cost_4", "Cost_5", "Cost_6",
				   "Cost_7", "Cost_8", "Cost_9", "Cost_10", "Cost_11"])
	
	# Fill the DF, by looping over all indeces, so that each index has its own 
	# row and 11 columns for time/ distance/ cost each!
	
	# Initalize Counter to give feedback during transformation!
	counter = 0
	
	# Start looping over Indices
	for index in indeces:
		
		# print the process:
		counter += 1
		if counter % 100 == 0: print(str(counter) + " / " + str(len(indeces)))
		
		# get all rows with the same Index (same session):
		# and extract the possible transportation modes:
		current_sess         = data_clean.loc[data_clean["sid"] == index]
		possible_trans_modes = current_sess["transport_mode"].values
			
		# Create Dict to add, we fill with the TransMode specific attributes and 
		# add it to the DF
		values_to_add = {}
		
		# Add all possible transportmodes specific times/ costs/ distances to df_
		for i in possible_trans_modes:
			
			values_to_add["Distance_" + str(i)] = current_sess.loc[current_sess["transport_mode"] == i]["distance_plan"].values[0]
			values_to_add["Cost_"     + str(i)] = current_sess.loc[current_sess["transport_mode"] == i]["price"].values[0]
			values_to_add["Time_"     + str(i)] = current_sess.loc[current_sess["transport_mode"] == i]["eta"].values[0]
		
		df_ = df_.append(values_to_add, ignore_index = True)
		
		
	# Add the SessionIDs  & save the Preprocessed DF --------------------------
	df_["SID"] = indeces		
	df_.to_pickle("data/processed/Multiclass_Approach/SubDFs/DF_" + DF_name)

# # E.G. for a couple of indeces from df_test_clean
# create_multiclass_df(df_test_clean, [1112456, 1413458, 1243160] , "TEEEEST")
# test_ = pd.read_pickle("data/processed/Multiclass_Approach/SubDFs/DF_TEEEEST")
	
	
# Use Function, to transform "df_train_clean" / "df_test_clean"----------------

# Loop over all SessionIDs:
# Save the results partly, and conecate parts later! --> faster to run!
# Select the Indices we want to extract!
df_to_extract   = df_test_clean                    # df_test_clean or df_train_clean
unique_sessions = df_to_extract["sid"].unique()    # which SIDs
save_name       = "Test_"                          # "Test_" or "Train_"

# run over the function to extract results:
j = 0
for i in range(25000, len(unique_sessions), 25000):
	
	# Print the current Range we're working on:
	print("from: "+ str(j) + " to " + str(i))
	
	# Create the Name it shall be saved in:
	DFNAME = save_name + str(i / 25000)
	
	# Do the Trasnforming
	create_multiclass_df(df_to_extract,
					     unique_sessions[j:i],
					     DFNAME)
	
	# update Indices
	j = i

# Add the last part that wasn't coverd by the loop
DFNAME = save_name + str(round(len(unique_sessions) / 25000)) + ".0"

create_multiclass_df(df_to_extract,
					 unique_sessions[j:len(unique_sessions)],
					 DFNAME)


# Conecate the single DFs to a big one-----------------------------------------
# Select Test or Train, that shall be extracted + amount of DFs
_which_   = "Test"        # "Test" or "Train"
amount_df = 4             # biggest number of the _which_ _X.0

# Create an basis DF, where we'll concate all other DFs
df_all = pd.read_pickle("data/processed/Multiclass_Approach/SubDFs/DF_" + _which_ + "_1.0")

for i in range(2, amount_df + 1):
	
	# Create Filename, based on the current number!
	file_number = str(i)
	file_name   = "DF_" + _which_ + "_" + file_number + ".0"
	
	# Read in the current Results:
	df_current = pd.read_pickle("data/processed/Multiclass_Approach/SubDFs/" + file_name)
	
	# Bind it to the big DF!
	df_all = pd.concat([df_all, df_current])


# Add the Response if its the Trainset:
if _which_ == "Train":
	df_all["Response"] = df_train_clean.drop_duplicates("sid")["click_mode"].values

# Save the Files:
df_all.to_pickle("data/processed/Multiclass_Approach/Multiclass" + _which_ + "Set")