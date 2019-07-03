# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:01:23 2019

@author: kuche_000

Read out sumamries of the results
"""
# Load all Packages needed
import pandas as pd
import numpy as np
import os

# Check Working Directory:
if "kdd-cup-2019" not in os.getcwd():
	raise ValueError("Your Working Directory is not set correctly")

def extract_summaries(path, key):
	"""
	function to extract the summaries of the files in "path", that 
	are a "summary".csv file!
	
	Args:
		- path (string) : path where the summary files are
		- key (string)  : what do all summary names have in common, 
		                  so we can detect them
		
	Return:
		- pandas DF with the joint summaries
	"""
	
	folder_files = os.listdir(path)
	
	for _file in folder_files:
		
		if key in _file:
			curr_res = pd.read_csv(path + "/" + _file)
			
			if "all_res" in locals():
				all_res = all_res.append(curr_res)
			else:
				all_res = curr_res
				
	return(all_res)

# Example of how to extract results and merge them:
res_Xgboost = extract_summaries("models/Multivariate Approach/merged_dfs/xgboost",
				                key = "Summary")

res_knn = extract_summaries("models/Multivariate Approach/merged_dfs/knn",
				                key = "Summary")

all_results = res_Xgboost.append(res_knn, ignore_index=True)

stacked_res = extract_summaries("models/stacking", key = "Summary")
