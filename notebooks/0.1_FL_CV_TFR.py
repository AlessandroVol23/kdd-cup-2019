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
_HIDDEN_LAYER_DIMS = ["25", "10", "12"] # [Layer Dimensions in the TF-NN]

# Set the Learning Rate of the TFR-Model:
_LEARNING_RATE = 0.05
#%% Example for a single itteration on the first 1.000 observations!
# Load Data and only keep the first 1000 rows!
df = pd.read_pickle("data/processed/Ranking/train_all_row.pickle")
df = df.head(1000)
df = df.fillna(-1)

# [1] Convert the Data to LIB-SVM Format (saved in CV/lol2.txt), and get the true responses
true_resp = create_svm_file(df = df,
							features_X = ["transport_mode", "price", "eta", "o_long", "o_lat"],
							path =  "CV/lol2.txt")

# [2] Set Parameters and define the estimator as "ranker"
hparams = tf.contrib.training.HParams(learning_rate=0.05)
ranker  = get_estimator(hparams)

# [3] Train the Model on the libSVM data stored in "_TEST_DATA_PATH"
ranker.train(input_fn=lambda: input_fn("data/processed/Ranking/tf_ranking/CV/lol2.txt"), 
			 steps=501) # loss is decreasing, so it seems to learn something (?!)

# [4] Evaluate the Model
ranker.evaluate(input_fn=lambda: input_fn("data/processed/Ranking/tf_ranking/CV/lol2.txt"), 
				steps=10) 

# [5] Get the predicitons and save their row-wise scores!
preds = ranker.predict(input_fn=lambda: input_fn_no_shuff2("data/processed/Ranking/tf_ranking/CV/lol2.txt").get_next())

j = 0
pred_scores = []
for i in preds:
    if j == 10000:
        break
    j+=1
    pred_scores.append(i)
	

# [6] Extract the Predicitons from the list of scores!
data2 = input_fn_no_shuff2("data/processed/Ranking/tf_ranking/CV/lol.txt")

predicted_class = []
for i in range(22):
    lower = int(i * 12)
    upper = int((i+1)*12)
    pos = np.argmax(np.array(pred_scores)[lower:upper])
    lol = data2.get_next()
    predicted_class.append(np.array(lol[0]["1"][pos]))
    
print(predicted_class)
print(len(predicted_class))

# [7] Calculate F1-Score
predicted_class = np.array(predicted_class)
f1_score(true_resp, predicted_class, average = "weighted")