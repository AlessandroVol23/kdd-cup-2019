import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
import pickle
import click

def create_svm_file(df, features_X, path):
    """
    This function saves the dataframe as lib svm file.
    :df: Pandas DataFrame to convert into libsvm file.
    :features_X: Feature list to take from df. These features have to exist in df
    :path: Path to save the file.
    :returns X, y
    """
    df.sort_values("sid", inplace=True)

    # Create ranking target
    # Check if click_mode is in df_columns -> If not it is the test set
    if 'click_mode' in df.columns:
        print("Build LTR labels")
        # Target 1 for right label / 100 if not the right label
        df = df.assign(target=df.apply(lambda x: 1 if x.click_mode == x.transport_mode else 100, axis=1))
    else:
        # If test set every entry gets zeri for a label
        print("Assign label 0 for test set")
        df = df.assign(target=0)

    X = df[features_X]
    y = df["target"]
    query_id = df.sid

    # Dump SVM file
    print("Dump file")
    dump_svmlight_file(X=X, y=y, f=path, query_id=query_id, zero_based=False)
    return X, y


@click.command()
@click.argument("path_train_file", type=click.Path(exists=True))
@click.argument("path_test_file", type=click.Path(exists=True))
@click.argument("path_feature_file", type=click.Path(exists=True))
@click.argument("output_path_train")
@click.argument("output_path_test")
def main(path_train_file, path_test_file, path_feature_file, output_path_train, output_path_test):
    import os
    print(os.getcwd())
    df_train = pd.read_pickle(path_train_file)
    df_test = pd.read_pickle(path_test_file)

    print("Loaded df_train with shape: ", df_train.shape)
    print("Loaded df_test with shape: ", df_test.shape)

    with open(path_feature_file) as f:
        features = f.read().splitlines()

    print("Create for df_train")
    train_X, train_y = create_svm_file(df_train, features, output_path_train)
    print("create for df test")
    test_X, test_y = create_svm_file(df_test, features, output_path_test)


if __name__ == "__main__":
    main()
