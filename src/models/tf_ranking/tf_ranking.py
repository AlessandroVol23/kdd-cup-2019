import itertools

import click
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_ranking as tfr
from sklearn.metrics import f1_score

import logging

logger = logging.getLogger(__name__)

tf.enable_eager_execution()
tf.executing_eagerly()

# Store the paths to files containing training and test instances.
# As noted above, we will assume the data is in the LibSVM format
# and that the content of each file is sorted by query ID.

_TRAIN_DATA_PATH = ''
_TEST_DATA_PATH = ''

# Define a loss function. To find a complete list of available
# loss functions or to learn how to add your own custom function
# please refer to the tensorflow_ranking.losses module.
_LOSS = "pairwise_logistic_loss"
# _LOSS = "sigmoid_cross_entropy_loss"

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
_NUM_FEATURES = 90

# Parameters to the scoring function.
_BATCH_SIZE = 1000
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

    # Do I have to define how to handle features in here?
    def _score_fn(context_features, group_features, mode, params, config):
        """Defines the network to score a documents."""
        del params
        del config
        # Define input layer.
        # Where does he get the group features from?
        example_input = [
            tf.layers.flatten(group_features[name])
            for name in sorted(example_feature_columns())
        ]
        # Example input are all features as numeric columns
        # The names are 0 -> _NUM_FEATURES
        # Input features
        input_layer = tf.concat(example_input, 1)

        cur_layer = input_layer
        for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
            cur_layer = tf.layers.dense(
                cur_layer,
                units=layer_width,
                activation="tanh")

        # Last Layer
        logits = tf.layers.dense(cur_layer, units=1)
        return logits

    return _score_fn


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
        # Model fn gets features and labels from input_fn
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            # group_size: An integer denoting the number of examples in `group_score_fn`
            # How many number of examples are in our group_score_fn?
            group_size=1,
            transform_fn=None,
            ranking_head=ranking_head),
        params=hparams)


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


def ltr_to_submission(df, features, ranker, path):
    features = features + ['sid']

    preds = ranker.predict(input_fn=lambda: input_fn(path))
    import itertools
    import numpy as np
    # Not sure how to get all preds because it runs infinit
    # So I take all till list size
    preds_slice = itertools.islice(preds, len(df))
    count = 0
    a = np.zeros((len(df), _LIST_SIZE))

    for i in preds_slice:
        a[count] = i
        count += 1

    test_X = df[features]

    test_X = test_X.assign(yhat=a[:, 0])

    df_end = pd.DataFrame(columns=['yhat'], index=df.sid.unique())

    df_end = test_X.sort_values(['sid', 'yhat'], ascending=False).groupby('sid').first()[[
        'yhat', 'transport_mode'
    ]]

    from sklearn.metrics import f1_score
    score = f1_score(df.groupby("sid").first()['click_mode'], df_end.transport_mode, average='weighted')
    logger.info('F1 Score is: %s', str(score))

    return df_end


@click.command()
@click.argument("path_train_svm", type=click.Path(exists=True))
@click.argument("path_val_svm", type=click.Path(exists=True))
@click.argument("path_feature_file", type=click.Path(exists=True))
@click.argument("path_train_df", type=click.Path(exists=True))
@click.argument("path_val_df", type=click.Path(exists=True))
def main(path_train_svm, path_val_svm, path_feature_file, path_train_df, path_val_df):
    logger.info("MAIN: Read in data")
    df_train_train = pd.read_pickle(path_train_df)
    df_train_test = pd.read_pickle(path_val_df)

    _TRAIN_DATA_PATH = path_train_svm
    _TEST_DATA_PA
    TH = path_val_svm

    with open(path_feature_file) as f:
        features = f.read().splitlines()

    _NUM_FEATURES = len(features)

    logger.info("MAIN: Create Ranker")
    hparams = tf.contrib.training.HParams(learning_rate=0.05)
    ranker = get_estimator(hparams)

    logger.info("MAIN: Train Ranker")
    ranker.train(input_fn=lambda: input_fn(_TRAIN_DATA_PATH), steps=10)

    logger.info("MAIN: LTR to submission")
    df_preds = ltr_to_submission(df_train_test, features, ranker, _TEST_DATA_PATH)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
