import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
import pickle
import click

def create_svm_file(df, features_X, path):
    """
    This function saves the dataframe as lib svm file.
    """
    df.sort_values("sid", inplace=True)

    # Create ranking target
    if 'click_mode' in df.columns:
        print("Build LTR labels")
        df = df.assign(target=df.apply(lambda x: 1 if x.click_mode == x.transport_mode else 0, axis=1))
    else:
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
@click.argument("output_path_train")
@click.argument("output_path_test")
def main(output_path_train, output_path_test):
    import os
    print(os.getcwd())
    df_train = pd.read_pickle('../../../data/interim/Train_extern_git_feat_pid_-1.pickle')
    df_test = pd.read_pickle('../../../data/interim/Test_extern_git_feat_pid_-1.pickle')

    print("Loaded df_train with shape: ", df_train.shape)
    print("Loaded df_test with shape: ", df_test.shape)

    with open('../../../data/interim/features_pid_all.pickle', 'rb') as fp:
        features = pickle.load(fp)

    print("Create for df_train")
    train_X, train_y = create_svm_file(df_train, features, output_path_train)
    print("create for df test")
    test_X, test_y = create_svm_file(df_test, features, output_path_test)

if __name__ == "__main__":
    main()