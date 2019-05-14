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
