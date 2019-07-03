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

def extract_summaries(paths, key):
	"""
	function to extract the summaries of the files in "path", that 
	are a "summary".csv file!
	
	Args:
		- paths (list)  :   list with paths (string values) to wjere the summary 
		                    files are [can contain multiple paths]
		- key (string)  : what do all summary names have in common, 
		                  so we can detect them
		
	Return:
		- pandas DF with the joint summaries
	"""
	
	for _path in paths:
		
		folder_files = os.listdir(_path)
		
		for _file in folder_files:
			
			if key in _file:
				curr_res = pd.read_csv(_path + "/" + _file)
				
				if "all_res" not in locals():
					all_res = curr_res
				else:
					all_res = all_res.append(curr_res)
				
	return(all_res)


# Example of how to use the function:
paths = ["models/Multivariate Approach/1/MLP",
		 "models/Multivariate Approach/1/KNN", 
		 "models/Multivariate Approach/1/RF", 
		 "models/Multivariate Approach/1/XGBoost",
		 "models/lgb_multi/1"]

results = extract_summaries(paths = paths, key = "Summary")

boxplot = results.boxplot(column=['F1_mean'], by = "model_type")
