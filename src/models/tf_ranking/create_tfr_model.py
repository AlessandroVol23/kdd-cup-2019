# -*- coding: utf-8 -*-
"""
File to do CrossValidation w/ the Tensorflow Ranking Approach
"""
# Load needed Packages:
import tensorflow as tf
import tensorflow_ranking as tfr
import wget
import numpy as np
import pandas as pd
import itertools
import json
from sklearn.datasets import dump_svmlight_file
import pickle
import os
import sklearn.metrics
from sklearn.metrics import f1_score

tf.enable_eager_execution()
tf.executing_eagerly()

# Check Working Directory:
if os.getcwd()[-12:] != "kdd-cup-2019":
	raise ValueError("Error with WorkingDirectory \n \
			[1] either WD isn't set correctly to 'kdd-cup-2019' \n \
			[2] or WD was renamed")

#%% Define Functions
# [0] Function to convert regular data to the lib-SVM format!
def create_svm_file(df, features_X, path):
    """
    Convert a pandas DF into a lib-SVM format!
	
	Args:
		- df (pd DF) : Pandas DF - needs to have the columns, that are passed
		               in the features_X argument!
					   
	    - features_X (list) : list of strings with the names of the attributes
	                          in df we want to use as feature!
	  
	    - path (string)     : path (incl. DF_Name) to save the lib-SVM at!
		                     [on top of "data/processed/Ranking/tf_ranking/"]
							 e.g. "CV/1.txt"
		
	Return: 
		- save the LIB-SVM DF
		- return the true responses of the passed df 
		  (used to calc the F1-Score later on!)
    """
	# [1] Check Input:
	# 	- all feature names in df, so we can select the corresponding cols?
    for _feat in features_X:
	    if _feat not in df.columns.values:
		    raise ValueError(_feat + "is not a columname in df")
			
    # 	- features_X must start with "transport_mode" 
	#     (needed to create valid predicitons)
    if features_X[0] != "transport_mode":
	    raise ValueError("'features_X' must start with 'transport_mode'")
				
	# 	- is the path, already defined?
	#     split folder and file name and check for the existence of the folder
    folder = path.split("/")
    folder = folder[:(len(folder) - 1)]
    folder = "".join(folder)
    if not os.path.isdir("data/processed/Ranking/tf_ranking/" + str(folder)):
	    raise ValueError(str(folder) + " is not existent in 'data/processed/Ranking/tf_ranking/'")
	
	
	# [2] Clean the DF and get it ready!
	#	- Sort the SIDs
    df.sort_values("sid", inplace = True)
	#	- drop rows, that have the same trans_mode multiple times for a single sid!
    df = df.drop_duplicates(['sid', 'transport_mode'], keep='first')

    # [3] Create ranking target
    # 	- if click_mode we mark the target with "1" and the irrelevant ones as "0"
    if 'click_mode' in df.columns:
        print("Build LTR labels")
        # T1 for target <--> 0 else
        df = df.assign(target=df.apply(lambda x: 1 if x.click_mode == x.transport_mode else 0, axis=1))
    else:
        # If test set every entry gets zeri for a label
        print("Assign label 0 for test set")
        df = df.assign(target=0)

	# [4] Spit the DF into Target & Feature + extract the SIDs
	# 	- we pass these to svm-converter to create a libsvm DF! 
    X = df[features_X]
    y = df["target"]
    query_id = df.sid
    path = "data/processed/Ranking/tf_ranking/" + str(path)

    # [5] Save the SVM_File on top of: "data/processed/Ranking/tf_ranking"
    print("Dump file")
    dump_svmlight_file(X=X, y=y, f=path, query_id=query_id, zero_based=False)
	
	# [6] Return the Values of the true click_modes [needed for metrics!]
    return np.array(df.drop_duplicates(["sid"]).click_mode)

# [1] Function for InputData to read them in in a serial way
def input_fn(path):
  """
  Function to load the LIBSVM Data from the path & tranform the feas into a dictionary [0]
  and the response [relevance of the current option] into a tf.Tensor [1]
  
  Response: Size: - as many rows as we have BATCH_SIZE
                  - as many cols as we have LIST_SIZE  
            Values: - the (shuffled) Responsevalues (the higher the better!) 
                      ["-1" was padded, and represents an option, that is not avaible]
  Features: Size: - Dict w/ as many 'keys' as NUM_FEATURES, each key holds an array 
                    of length LIST_SIZE and as many entrances as BATCH_SIZE
                    
  Note:           - if run like this 'input_fn(_TEST_DATA_PATH)' it will return a tuple
                    of "features" as dictionary and "responses" as list!
                    
                    For Training the Model this function is passed as anonymus function and 
                    will pass all datapoints in a serial way to the TF-NN!
                    (called via: "lambda: input_fn(_TEST_DATA_PATH)")
  """
  train_dataset = tf.data.Dataset.from_generator(
      tfr.data.libsvm_generator(path, _NUM_FEATURES, _LIST_SIZE),
      output_types=(
          {str(k): tf.float32 for k in range(1,_NUM_FEATURES+1)},
          tf.float32
      ),
      output_shapes=(
          {str(k): tf.TensorShape([_LIST_SIZE, 1])
            for k in range(1,_NUM_FEATURES+1)},
          tf.TensorShape([_LIST_SIZE])
      )
  )

  train_dataset = train_dataset.shuffle(1000).repeat().batch(_BATCH_SIZE)
  return train_dataset.make_one_shot_iterator().get_next()

# [2] Scoring Function
# compute a relevance score for a (set of) query-document pair(s). 
# The TF-Ranking model will use training data to learn this function.
# Function takes the features of a single example (i.e., query-document pair)
# and produces a relevance score.
def example_feature_columns():
  """Returns the example feature columns."""
  feature_names = [
      "%d" % (i + 1) for i in range(0, _NUM_FEATURES)
  ]
  return {
      name: tf.feature_column.numeric_column(
          name, shape=(1,), default_value=0.0) for name in feature_names
  }

def make_score_fn():
  """Returns a scoring function to build `EstimatorSpec`."""

  def _score_fn(context_features, group_features, mode, params, config):
    """Defines the network to score a documents."""
    del params
    del config
    # Define input layer.
    example_input = [
        tf.layers.flatten(group_features[name])
        for name in sorted(example_feature_columns())
    ]
    input_layer = tf.concat(example_input, 1)

    cur_layer = input_layer
    for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
      cur_layer = tf.layers.dense(
          cur_layer,
          units=layer_width,
          activation="tanh")

    logits = tf.layers.dense(cur_layer, units=1)
    return logits

  return _score_fn


# [3] Evaluation Metrics
# Information Retrieval evalution metricis implemented in the TF Ranking library.
# !!! Could be cool to implement a F1-Score !!!
def eval_metric_fns():
  """Returns a dict from name to metric functions.

  This can be customized as follows. 
  Care must be taken when handling padded lists.

  def _auc(labels, predictions, features):
    is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
    clean_labels   = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
    clean_pred     = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
    return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
  metric_fns["auc"] = _auc

  Returns:
    A dict mapping from metric name to a metric function with above signature.
  """
  metric_fns = {}
  metric_fns.update({
      "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
          tfr.metrics.RankingMetricKey.NDCG, topn=topn)
      for topn in [1, 3, 5, 10]
  })

  return metric_fns


# [4] Put all together
# put all of the components above together and 
# create an Estimator that can be used to train and evaluate a model.
def get_estimator(hparams):
  """Create a ranking estimator.

  Args:
    hparams: (tf.contrib.training.HParams) a hyperparameters object.

  Returns:
    tf.learn `Estimator`.
  """
  def _train_op_fn(loss):
    """Defines train op used in ranking head."""
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=hparams.learning_rate,
        optimizer="Adagrad")

  ranking_head = tfr.head.create_ranking_head(
      loss_fn=tfr.losses.make_loss_fn(_LOSS),
      eval_metric_fns=eval_metric_fns(),
      train_op_fn=_train_op_fn)

  return tf.estimator.Estimator(
      model_fn=tfr.model.make_groupwise_ranking_fn(
          group_score_fn=make_score_fn(),
          group_size=1,
          transform_fn=None,
          ranking_head=ranking_head),
      params=hparams)

# [5] Input Function needed for prediciting!	  
def input_fn_no_shuff2(path):
  """
  Function to load the LIBSVM Data from the path
  Again features as dict and response as list!
  
  Return:
      iterator object, that we can itterate over
      Will always bring back the original order, as we have set a seed
  """
  train_dataset = tf.data.Dataset.from_generator(
      tfr.data.libsvm_generator(path, _NUM_FEATURES, _LIST_SIZE, seed = 1312),
      output_types=(
          {str(k): tf.float32 for k in range(1,_NUM_FEATURES+1)},
          tf.float32
      ),
      output_shapes=(
          {str(k): tf.TensorShape([_LIST_SIZE, 1])
            for k in range(1,_NUM_FEATURES+1)},
          tf.TensorShape([_LIST_SIZE])
      )
  )
  return train_dataset.make_one_shot_iterator()

#%% Set the global params
# Define a loss function [refer to the tensorflow_ranking.losses module.]
_LOSS = "pairwise_logistic_loss"

# For simplicity we fix the amount of response classes to a mximumvalue called 'list_size'.
# 'list_size' is the maximum number of documents per query in the dataset. 
# [for KDD it's 12, as we can maximal have 12 different offers for a single query]
#   * If a query has fewer documents, its Tensor will be padded appropriately.
#   * If a query has more documents, we shuffle its list of documents and trim the list down to the prescribed list_size.
_LIST_SIZE = 12

# The total number of features per query-document pair 
_NUM_FEATURES = 127 # [should be equal to len(feature_names)]

# Parameters to the scoring function.
_BATCH_SIZE = 50                 # [size of the batches for training]
_HIDDEN_LAYER_DIMS = ["256", "128", "64"] # [Layer Dimensions in the TF-NN]

# Set the Learning Rate of the TFR-Model & the amount of steps for the training!
_LEARNING_RATE = 0.05
_STEPS         = 1001 


# Select the path for feature names and for the SIDs for CV!
_FEAT_PATH  = 'data/processed/features/multiclass_1.pickle'
_CV_PATH    = "data/processed/split_test_train/5-fold/SIDs.pickle"
_TRAIN_DATA = "C:/Users/kuche_000/Desktop/RankingData/train_all_row.pickle" # --> will be adjusted asa the branch is merged!

#%% Cross Validate a TF-Network!
if __name__ == "__main__":
	
	# [1] Load all necessary data for the CV
	print("--- Load the Data needed for CV ---\n")
	
	# [1-1] Load the names, we want to use as features for our model &
	#       add "transport_mode" as first feature name, as the functions expects this!
	with open(_FEAT_PATH, 'rb') as ff:
			feature_names = pickle.load(ff)
	feature_names.insert(0, "transport_mode")
	
	# [1-2] Load the SIDs, we use to split our data for CV!
	with open(_CV_PATH, "rb") as fp:
		CV = pickle.load(fp)
	
	# [1-3] Load the data we use for Training
	df = pd.read_pickle(_TRAIN_DATA)
	
	# [1-4] Check whether the folder to save results in is existent
	#       if it exists we count number of .csv-files and assign the number
	if not os.path.isdir("models/ranking/tfr"):
		raise ValueError("'models/ranking/tfr' is not existent")
	else:
		numb = 0
		for _files_ in os.listdir("models/ranking/tfr/"):
			if ".csv" in _files_:
				numb = numb + 1	
	
	# [2] Initalize DFs and lists to save results/ metrics/ predicitons... of CV
	print("--- Initalize Lists to save CV-Results ---\n")
	# [2-1] Intialize Lists and DFs to save results/ metrics/ predicitons...
	# [2-1-1] lists to save the metric results of the k-fold CV:
	F1          = []
	Accuracy    = []
	Precision   = []
	Conf_Matrix = []	
	
	# [2-1-2] Initalize a DF to save the predictions scores [for stacking]
	res = pd.DataFrame(columns = ['M_prob_0', 'M_prob_1', 'M_prob_2', 'M_prob_3',
							      'M_prob_4', 'M_prob_5', 'M_prob_6', 'M_prob_7', 
								  'M_prob_8', 'M_prob_9', 'M_prob_10', 'M_prob_11', 
								  'fold', "SID"])
	
	# [3] Start the CV, by looping over the SID lists in "CV"
	print("--- Start the CV ---\n")
	for i in range(len(CV)):
		
		# [3-0-1] Print the current Process:
		print("CV number: " + str(i + 1) + " / " + str(len(CV)) + "\n")
		
		# [3-1] Split the data into test- and validation-set [based on SIDs in CV]
		# [3-1-1] Extract Test_Set based on the current CV:
		curr_test_set  = df.loc[df["sid"].isin(CV[i]), :]
		
		# Get the SIDs of the TestSet. Needed for the PredScores [stacking]
		curr_test_sids = curr_test_set.sort_values(["sid"]).sid.unique()
		
		# [3-1-2] Extract SIDs we use for training & slice the original DF by it!
		train_sids = []
		for j in range(len(CV)):
			if j != i:
				train_sids = train_sids + CV[j]
			
		# 	Extract Train_Set based on the "train_sids"
		curr_train_set = df.loc[df["sid"].isin(train_sids), :]
		
		
		# [3-2] Convert Train & Validation-Set into lib-SVM:
		# Convert the current train/ validation set into lib-SVM, the format 
		# needed to train a TF-Ranking Modell in this case.
		# Also get the true responses, needed for evaluation later on!
		
		# LIB-SVM Data is saved in:  
		#	- "data/processed/Ranking/tf_ranking/CV/train.txt" or
		#	- "data/processed/Ranking/tf_ranking/CV/vali.txt"
		# --> DON'T CHANGE THESE PATHs, as we need it for training/ prediciting!!
		true_resp_vali = create_svm_file(df = curr_test_set,
								    features_X = feature_names,
								    path =  "CV/vali.txt")
		
		true_resp_train = create_svm_file(df = curr_train_set,
							        features_X = feature_names,
							        path =  "CV/train.txt")
		
	    # [3-3] Train the Model:
		# [3-3-1] Set Parameters and define the learner	
		hparams = tf.contrib.training.HParams(learning_rate= _LEARNING_RATE)
		ranker  = get_estimator(hparams)
		
		# [3-3-2] Train the Model on the lib-SVM data
		# 'create_svm_file' function should have saved data in libsvm in corresponding paths!
		ranker.train(input_fn=lambda: input_fn("data/processed/Ranking/tf_ranking/CV/train.txt"), 
	 			     steps=_STEPS) # loss is decreasing, so it seems to learn something (?!)
	
		# [3-4] Get predicitons on the validation set
		# Validation set should be in: 
		#	- "data/processed/Ranking/tf_ranking/CV/vali.txt"  
		#      (saved to this path by the 'create_svm_file' function)
		preds = ranker.predict(input_fn=lambda: input_fn_no_shuff2("data/processed/Ranking/tf_ranking/CV/vali.txt").get_next())
		
		# [3-4-1] Predicitons are saved rowise
		# --> Loop over the predictions [itterator-object] and save the scores!
		j = 0
		pred_scores = []
		for i_ in preds:
			if j > 12*curr_test_set.sid.nunique():
				break
			j+=1
			pred_scores.append(i_) # [len(pred_scores) == 12*curr_test_set.sid.nunique()]
		
		# [3-4-2] Create a single Prediciton from the list of prediciton Scores!
		# which pred_scores belongs to which 'trans_mode'?
		# Use the 'trans_mode' with the highest score as prediciton!
		
		# [3-4-2-1] Reload Data - we've used for prediciton. [exactly same as seed!]
		data2 = input_fn_no_shuff2("data/processed/Ranking/tf_ranking/CV/vali.txt")
		
		# [3-4-2-2] loop over all predscores, that belong to one SID [always 12]
		# and choose prediciton based on highest prediciton score!
		predicted_class = []
		for itt in range(curr_test_set.sid.nunique()):
		    lower = int(itt * 12)
		    upper = int((itt+1)*12)
		    pos = np.argmax(np.array(pred_scores)[lower:upper])
		    lol = data2.get_next()
			
			# Important to have "transport_mode" as "1" feature else this would make no sense!
		    predicted_class.append(np.array(lol[0]["1"][pos]))
			
			# [3-4-2-3] Add the predicted Scores to the prediction csv for stacking
		    cols_to_add = list(np.repeat(0, 14))
			
			#	get the transport_modes that were only padded and not avaible for query
		    padded_modes = np.where(lol[1] < 0)
			
			#	for the possible modes, get the scores!
		    for _mode in range(12):
			    if _mode not in padded_modes[0]:
				    _trans_mode = lol[0]["1"][_mode]
				    cols_to_add[_trans_mode] = np.array(pred_scores)[lower:upper][_mode][0]
					
		    cols_to_add[12] = (i + 1)
		    cols_to_add[13] = curr_test_sids[itt]
			
			# Bind the List to the result DF & save it!
		    res.loc[len(res)] = cols_to_add
		
		# [3-3-2-4] Save the file with the tfr scores of the single SIDs!
		res.to_csv("models/ranking/tfr/Predicitons_" + str(numb) + ".csv")
			
		# [3-4] Get the Metrics of the current fold & add them to corresponding lists
		F1.append(sklearn.metrics.f1_score(true_resp_vali,  np.array(predicted_class),
										   average="weighted"))
		
		Accuracy.append(sklearn.metrics.accuracy_score(true_resp_vali,  np.array(predicted_class)))
		
		Precision.append(sklearn.metrics.recall_score(true_resp_vali,  np.array(predicted_class),
									                  average = "weighted"))
		
		Conf_Matrix.append(sklearn.metrics.confusion_matrix(true_resp_vali,  np.array(predicted_class)))
		
	
		print("F1-Score for the " + str(i + 1) + "th fold: " + str(F1[i]))
		print("\n")
		
	print("CV done\n")
	# [4] Save the Results:
	# [4-1] Extract ParameterSettings --> Needs to be adjusted if we add more parameters!
	_params = json.dumps({"loss"  : _LOSS,
						 "batch_size" : _BATCH_SIZE,
						 "Layers" : _HIDDEN_LAYER_DIMS,
						 "LR"     : _LEARNING_RATE,
						 "Steps"  : _STEPS})
	
	# [4-2] Create BasicShape for the Result .csv
	dict_to_pd = {'model_type' : "TFR", 'parameters' : _params,
			      'features' : " - ".join(feature_names), "Number": numb}
	
	# [4-3] Add CV-Scores to the Dict:
	for index in range(len(F1)):
		dict_to_pd["F1_" + str(index + 1)]        = F1[index]
		dict_to_pd["Acc_" + str(index + 1)]       = Accuracy[index]
		dict_to_pd["Precision_" + str(index + 1)] = Precision[index]
		dict_to_pd["Conf_" + str(index + 1)]      = Conf_Matrix[index]
	
	# [4-4] Add mean of the scores [except for Confmatrix]
	dict_to_pd["F1_mean"]   = np.mean(F1)
	dict_to_pd["Acc_mean"]  = np.mean(Accuracy)
	dict_to_pd["Prec_mean"] = np.mean(Precision)
	
	# [4-5] Transform it to pandas, order the columns and save it: 
	pd_res = pd.DataFrame([dict_to_pd])
			
	pd_res.to_csv("models/ranking/tfr/Summary_" + str(numb) + ".csv")