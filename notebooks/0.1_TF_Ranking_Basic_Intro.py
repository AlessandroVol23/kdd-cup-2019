# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:03:53 2019

@author: kuche_000

Get Started with TF Ranking
"""
# Import needed Packages:
import tensorflow as tf
import tensorflow_ranking as tfr

tf.enable_eager_execution()
tf.executing_eagerly()

# (1) Define all flexible Parameters-------------------------------------------

# Get the [dummy] dataset  and save its paths! [Train- & Test-Set]
# ['https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/data/train.txt']
# Store the paths to files containing training and test instances.
# !!! Data in LibSVM format and content of each file is sorted by query ID !!!
_TRAIN_DATA_PATH = "C:/Users/kuche_000/Desktop/KDD - Own/ranking-master/tensorflow_ranking/examples/data/train.txt"
_TEST_DATA_PATH = "C:/Users/kuche_000/Desktop/KDD - Own/ranking-master/tensorflow_ranking/examples/data/test.txt"

# Define a loss function. 
# complete list of available functions or add own function:
# refer to the tensorflow_ranking.losses module.
_LOSS = "pairwise_logistic_loss"

# A training instance is represented by a Tensor that contains features from a
# list of documents associated with a single query. 
# For simplicity, we fix the shape of these Tensors to a maximum 
# list size ["list_size"], the maximum number of documents per query in the dataset.

# In this demo, we take the following approach:
#   * If a query has fewer documents, its Tensor will be padded
#     appropriately.
#   * If a query has more documents, we shuffle its list of
#     documents and trim the list down to the prescribed list_size.
_LIST_SIZE = 4

# The total number of features per query-document pair.
# We set this number to the number of features in the MSLR-Web30K
# dataset.
_NUM_FEATURES = 136

# Parameters to the scoring function
_BATCH_SIZE = 12
_HIDDEN_LAYER_DIMS = ["5", "12"]

# (2) Input Pipeline-----------------------------------------------------------
# input pipeline that reads your dataset and produces a tensorflow.data.Dataset object
# parameterize function w( a path argument so it can be used to read in train & test set!

# LibSVM parser that is included in the tensorflow_ranking.data module to 
# generate a Dataset from a given file

def input_fn(path):
	"""
	Function to read in a libSVM DF an generate a tf.data.dataset-object!
	
	Args:
		path (string) : Gives the path, where the (libSVM-)DF is lying
	"""
	train_dataset = tf.data.Dataset.from_generator(
      tfr.data.libsvm_generator(path, _NUM_FEATURES, _LIST_SIZE),
	  
	  # Define the Types of the Input:
      output_types=(
          {str(k): tf.float32 for k in range(1,_NUM_FEATURES+1)},
          tf.float32
      ),
	  
	  # Define the Shapes of the Dataset!
      output_shapes=(
          {str(k): tf.TensorShape([_LIST_SIZE, 1])
            for k in range(1,_NUM_FEATURES+1)},
          tf.TensorShape([_LIST_SIZE])
      )
	  )
	  
	train_dataset = train_dataset.shuffle(1000).repeat().batch(_BATCH_SIZE)
	return train_dataset.make_one_shot_iterator().get_next()

t2 = input_fn(_TRAIN_DATA_PATH)

# Scoring Function ------------------------------------------------------------
# compute a relevance score for a (set of) query-document pair(s). 
# The TF-Ranking model will use training data to learn this function.

# Function to convert features to 'tf.feature_column' [so they have right type!]
def example_feature_columns():
  """Returns the example feature columns in the correct type
     Right now only numeric features!
  """
  # Create Names ["1",..., "_NUM_FEATURES"]
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

# (3) Evaluation Metric--------------------------------------------------------
# 
def eval_metric_fns():
  """Returns a dict from name to metric functions.

  This can be customized as follows. Care must be taken when handling padded
  lists.

  def _auc(labels, predictions, features):
    is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
    clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
    clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
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

# PUT IT ALL TOGETHER ---------------------------------------------------------
# ready to put all of the components above together and create an Estimator that 
# can be used to train and evaluate a model
  
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

# Let us instantiate and initialize the Estimator we defined above  
hparams = tf.contrib.training.HParams(learning_rate=0.05)
ranker  = get_estimator(hparams)

# input_fn(PATH) is the function to read in data!
ranker.train(input_fn = lambda: input_fn(_TRAIN_DATA_PATH), steps = 25)
lol = ranker.evaluate(input_fn=lambda: input_fn(_TEST_DATA_PATH), steps=15)

# Extract Results
lol.values()
lol.keys()
lol["labels_mean"] # 0.77777780 w/ 10
                   # 0.77800936 w/ 15
lol["loss"]        # 0.89 with the trainset2 [only 1 Query ID Pair] 

ranker.model_dir
