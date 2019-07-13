import logging
import click
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# Initialize logger
logger = logging.getLogger(__name__)


def get_X_y(df):
    """
    :param df: DataFrame to extract X and y. y has to be in the last column
    :return: X and y
    """
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def create_k_fold(X, y):
    """

    :param X: Feature matrix
    :param y: Label Vector
    :return: KFold Object
    """
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    kf.get_n_splits(X, y)
    return kf


def execute_k_fold(kf, X, y, target_col):
    """
    This function trains the model on each split and prints the f1 score.
    :param kf: KFold Object
    :return: void
    """
    count_iter = 0

    with(
            open("data/interim/task2/" + target_col + "_" + str(datetime.now().strftime('%m-%d-%H-%M-%S')) + ".txt",
                 "w+")) as fo:
        for train_index, test_index in kf.split(X, y):
            count_iter += 1
            logger.info('Split Nr. {}'.format(count_iter))
            logger.debug("Train size: {}, Test size: {}".format(len(train_index), len(test_index)))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            rf = RandomForestClassifier(n_estimators=10)
            rf.fit(X_train, y_train)
            y_preds = rf.predict(X_test)
            f1 = f1_score(y_test, y_preds, average='weighted')
            logger.info("F1 Score at split {} is {:.2f} %".format(count_iter, f1 * 100))
            fo.writelines("F1 Score at split {} is {:.2f} %\n".format(count_iter, f1 * 100))
            logger.info("====================================================================")


@click.command()
@click.argument('target_col')
def main(target_col):
    """
    :param column: Which column to take as a label
    """
    logger.info("Start.")
    logger.info('Target col is {}'.format(target_col))
    logger.info("Read in df")
    df = pd.read_pickle('data/interim/task2/df_pp.pickle')

    logger.info("Read in label_matrix")
    label_matrix = pd.read_pickle('data/interim/label_matrix.pickle')

    logger.info("Join label_matrix and df")
    df_w_labels = pd.merge(df, label_matrix, on='pid')

    logger.info("Read in features")
    features = pd.read_pickle('data/interim/task2/features.pickle')
    # Add target column to features
    features = features + [target_col]

    logger.info("Get x and y")
    X, y = get_X_y(df_w_labels[features])

    logger.info("Create KFold object")
    kf = create_k_fold(X, y)

    logger.info("Execute and train model")
    execute_k_fold(kf, X, y, target_col)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
