import tensorflow as tf
import tensorflow_ranking as tfr
import pandas as pd
from sklearn.metrics import f1_score

tf.enable_eager_execution()
tf.executing_eagerly()

# Store the paths to files containing training and test instances.
# As noted above, we will assume the data is in the LibSVM format
# and that the content of each file is sorted by query ID.

_TRAIN_DATA_PATH = "../data/interim/df_train_test.libsvm"
_TEST_DATA_PATH = "../data/interim/df_train_test.libsvm"

# Define a loss function. To find a complete list of available
# loss functions or to learn how to add your own custom function
# please refer to the tensorflow_ranking.losses module.
_LOSS = "pairwise_logistic_loss"
_LOSS = "sigmoid_cross_entropy_loss"

# In the TF-Ranking framework, a training instance is represented
# by a Tensor that contains features from a list of documents
# associated with a single query. For simplicity, we fix the shape
# of these Tensors to a maximum list size and call it "list_size,"
# the maximum number of documents per query in the dataset.
# In this demo, we take the following approach:
#   * If a query has fewer documents, its Tensor will be padded
#     appropriately.
#   * If a query has more documents, we shuffle its list of
#     documents and trim the list down to the prescribed list_size.
_LIST_SIZE = 100

# The total number of features per query-document pair.
# We set this number to the number of features in the MSLR-Web30K
# dataset.
_NUM_FEATURES = 4

# Parameters to the scoring function.
_BATCH_SIZE = 100
_HIDDEN_LAYER_DIMS = ["20", "10"]


# _OUT_DIR = "../models/tfranking/"

def input_fn(path):
    train_dataset = tf.data.Dataset.from_generator(
        tfr.data.libsvm_generator(path, _NUM_FEATURES, _LIST_SIZE),
        output_types=(
            {str(k): tf.float32 for k in range(1, _NUM_FEATURES + 1)},
            tf.float32
        ),
        output_shapes=(
            {str(k): tf.TensorShape([_LIST_SIZE, 1])
             for k in range(1, _NUM_FEATURES + 1)},
            tf.TensorShape([_LIST_SIZE])
        )
    )

    train_dataset = train_dataset.shuffle(1000).repeat().batch(_BATCH_SIZE)
    return train_dataset.make_one_shot_iterator().get_next()


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
        print("MAKE SCORE FUNCTION:")
        print(example_input)
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


def ltr_to_submission(df, ranker, path):
    preds = ranker.predict(input_fn=lambda: input_fn(_TRAIN_DATA_PATH))
    import itertools
    import numpy as np
    # Not sure how to get all preds because it runs infinit
    # So I take all till list size
    preds_slice = itertools.islice(preds, len(df_train_test))
    count = 0
    a = np.zeros((len(df_train_test), _LIST_SIZE))

    for i in preds_slice:
        a[count] = i
        count += 1

    test_X = df[[
        'sid',
        'transport_mode',
        'distance_plan',
        'eta',
        'price'
    ]]

    # Assign prediction vals to df
    # Tried with a or a sum for all features
    test_X = test_X.assign(yhat=a[:, 0])

    df_end = pd.DataFrame(columns=['yhat'], index=df.sid.unique())

    df_end = test_X.sort_values(['sid', 'yhat'], ascending=False).groupby('sid').first()[[
        'yhat', 'transport_mode'
    ]]

    return df_end


def main():
    df_train_train = pd.read_pickle("data/interim/df_train_test.pickle")
    df_train_test = pd.read_pickle("data/interim/df_train_test.pickle")

    hparams = tf.contrib.training.HParams(learning_rate=0.05)
    ranker = get_estimator(hparams)

    ranker.train(input_fn=lambda: input_fn(_TRAIN_DATA_PATH), steps=100)

    df_preds = ltr_to_submission(df_train_train, ranker, '../data/interim/df_train_train.libsvm')

    f1_score(df_train_train.groupby("sid").first()['click_mode'], df_preds.transport_mode, average='weighted')
