import pandas as pd
import numpy as np


# Euclidean Distance Calculation
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# Get coordinates of origins/destinations and generate arrays
def get_points(df, x, y):
    f1 = df[x].values
    f2 = df[y].values
    point_array = np.array(list(zip(f1, f2)))
    return point_array


# Set district center as centroids
def set_centroids(x, y):
    df_beijing_districts = pd.read_csv("../data/external/districts/beijing_districts.csv")

    x1 = df_beijing_districts[x].values
    y1 = df_beijing_districts[y].values
    centroids = np.array(list(zip(x1, y1)))
    return centroids


# Locate existing column in data frame and add new column filled with NaN right behind
def preprocess_districts(df, ex_column, new_column):
    add_after_this_column = df.columns.get_loc(ex_column)
    new_column_position = add_after_this_column + 1

    df.insert(new_column_position, new_column, np.nan, True)
    return df


# Assigning each point to its closest district
def assign_points(df, column, points, centroids):

    # Cluster array filled with 0
    clusters = np.zeros(len(points))

    for p in range(len(points)):

        if p % 100 == 0:
            print("Processing row {}".format(str(p)), end="\r")

        distances = dist(points[p], centroids)
        cluster = np.argmin(distances)
        clusters[p] = cluster
        df.loc[df.index == p, column] = column + str(cluster)

    return df


def one_hot(df, column):
    df = df.join(pd.get_dummies(df[column]))
    return df
