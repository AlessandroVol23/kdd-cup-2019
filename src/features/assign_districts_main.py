
from assign_district_functions import dist, get_points, set_centroids, preprocess_districts, assign_points, one_hot


def assign_districts(df_train):

    # Getting coordinates of Starting/Destination Points
    P = get_points(df_train, 'o_lat', 'o_long')
    Q = get_points(df_train, 'd_lat', 'd_long')

    # Setting Centroids
    C = set_centroids('o_lat', 'o_long')

    # Add o_district, d_district filled with NaN values after columns o_lat, d_lat
    df_train = preprocess_districts(df_train, 'o_lat', 'o_district')
    df_train = preprocess_districts(df_train, 'd_lat', 'd_district')

    # Assigning each point to its closest origin/destination district
    df_train = assign_points(df_train, 'o_district', P, C)
    df_train = assign_points(df_train, 'd_district', Q, C)

    # One-Hot-Encoded assigned districts
    df_train = one_hot(df_train, 'o_district')
    df_train = one_hot(df_train, 'd_district')

    df_train.to_pickle("../data/external/districts/train_all_first_districts.pickle")

    return df_train
