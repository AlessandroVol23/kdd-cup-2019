import pandas as pd
import numpy as np


def create_query_file(df_r):
    df = df_r.copy()

    # Sort values by sid to get all rows right
    df = df.sort_values("sid")

    # Reset index beginning from zero and assign the index as a new column "row"
    df = df.reset_index(drop=True)
    df = df.assign(row=df.index)

    # Count row +1 because it started with zero
    df.row = df.row+1

    # Get the last row of every sid
    sid_rows = pd.DataFrame(df.groupby("sid").last()["row"])

    # Calculate the difference between the rows to create the query file
    sid_diff = sid_rows.assign(difference=sid_rows.diff())

    # First value of difference is zero -> Take first row value
    sid_diff.iloc[0, 1] = sid_diff.iloc[0, 0]
    query = sid_diff.difference.values

    return query


def save_model(model, file_path):
    """
        This function saves the model in file_path as a pickle file.
        model: Model to save
        file_path: Path (relative or absolute) to save the model.
        It is just allowed to save models in the model folder!
        returns: nothing
    """
    import pickle
    import os
    if "models" not in file_path:
        raise ValueError("It is just allowed to save models in the models folder!")
    print("Save model...")
    path = os.path.join(file_path)
    pickle.dump(model, open(path, 'wb'))
    print("Model was saved at {}".format(path))
    print("Don't forget to add your model in DVC with dvc and git workflow!")
    
    
def load_model(file_path):
    import pickle
    print("Load model...")
    loaded_model = pickle.load(open(file_path, 'rb'))
    print("Model was loaded!")
    return loaded_model