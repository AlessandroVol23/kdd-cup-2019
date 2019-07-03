import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import pickle
import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import Point


# Initialize logger
logger = logging.getLogger(__name__)


def preprocess_coordinates(df):
    '''
    Generates following 5 columns
    * o_long
    * o_lat
    * d_long
    * d_lat

    And deletes these:
    * o
    * d

    +2 columns
    '''
    df[["o_long", "o_lat"]] = df.o.str.split(",", 1, expand=True).astype(float)
    df.drop("o", axis=1, inplace=True)

    df[["d_long", "d_lat"]] = df.d.str.split(",", 1, expand=True).astype(float)
    df.drop("d", axis=1, inplace=True)
    return df


def time_features(dataf, type='req'):
    '''
    Creates 5 new time column on the dataframe. This can be used with:
    * df_train_queries
    * df_test_queries
    * df_train_plans
    * df_test_plans
    * df_train_clicks
    * df_test_clicks

    Just specify the dataframe variable and the type of column you want to change: req, click or plan
    '''

    if type not in ['req', 'click', 'plan']:
        logger.error("Wrong time type, it should be ['req', 'click', 'plan']. Try again.")

    # Change the selected Time Column to datetime [easier to extract stuff]
    dataf[type + '_time'] = pd.to_datetime(dataf[type + '_time'])

    # Add new features, based on the ColumnType:

    # [1] Add the Hour, the Day and the Month of the time type
    dataf[type + '_date'] = dataf[type + '_time'].dt.strftime('%m-%d')
    # dataf[[type + '_month',type + '_day']] = dataf[type + '_date'].str.split("-",expand=True,).astype(int)
    # dataf = dataf.drop(type + '_date', axis=1)
    dataf[type + '_hour'] = dataf[type + '_time'].dt.hour

    # [2] Add a Binary, indicating whether its weekend or not! [ERROR]
    dataf[type + '_weekend'] = dataf[type + '_time'].dt.day_name().apply(
        lambda x: 1 if x in ["Friday", "Saturday"] else 0)

    # [3] Add binary, NIGHT / EVENING / DAY Column
    dataf[type + '_night'] = dataf[type + '_hour'].apply(lambda x: 1 if x <= 7 else 0)
    dataf[type + '_day'] = dataf[type + '_hour'].apply(lambda x: 1 if x in range(8, 18) else 0)
    dataf[type + '_evening'] = dataf[type + '_hour'].apply(lambda x: 1 if x > 18 else 0)

    # Print some Info
    # logger.debug('5 new columns created: month, day, weekend, hour, and hour_bin')

    return dataf


def add_public_holidays(dataf):
    '''
    Creates 1 new time column on the dataframe, a binary column stating if the date is a public holiday or not.
    '''
    df_holidays = pd.read_csv("data/external/CN_holidays_summary.csv", index_col=False)
    if 'req_date' not in dataf:
        try:
            dataf['req_time'] = pd.to_datetime(dataf['req_time'])
            dataf['req_date'] = dataf['req_time'].dt.strftime('%m-%d')
        except:
            logger.error("Can't create 'req_date' column. Make sure to run the `time_features` function first.")
            sys.exit(1)

    dataf['is_holiday'] = dataf['req_date'].apply(lambda x: int(x in df_holidays.values))
    dataf.drop('req_date', 1)
    logger.debug("Feature added: is_holiday, binary column stating if a date is a holiday or not")
    return dataf


def add_dist_nearest_subway(dataf):
    '''
    #Creates 1 new column with the distance to the nearest subway station (from subways.csv)
    '''

    def extract_Points_df(df, lat_column, long_column, crs={'init', 'epsg:4326'}):
        df_copy = df.copy()
        geometry = [Point(xy) for xy in zip(df_copy[long_column], df_copy[lat_column])]
        Points = gpd.GeoDataFrame(df_copy, crs=crs, geometry=geometry)
        return Points

    df_subways = pd.read_csv("data/external/subways.csv", index_col=False).round(2)

    if 'o_lat' not in dataf or 'o_long' not in dataf:
        logger.error(
            "The dataframe doesn't have the coordinates in the correct format. They need to be 'o_lat' and 'o_long'.")

    gdf_subways = extract_Points_df(df_subways, lat_column="o_lat", long_column="o_long")
    gdf_dataf = extract_Points_df(dataf, lat_column="o_lat", long_column="o_long")

    pts3 = gdf_subways.geometry.unary_union

    # https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe
    def near(point, pts=pts3):
        if (near.i + 1) % 100 == 0:
            print("Processing row {}".format(str(near.i + 1)), end="\r")
        near.i += 1
        # find the nearest point and return the corresponding Place value
        nearest = gdf_subways.geometry == nearest_points(point, pts)[1]
        return "%.3f" % (gdf_subways[nearest].geometry.get_values()[0].distance(point) * 10.0)

    near.i = 0
    gdf_dataf['dist_nearest_sub'] = gdf_dataf.apply(lambda row: near(row.geometry, pts3), axis=1)
    gdf_dataf = gdf_dataf.drop('geometry', 1)
    dataf = pd.DataFrame(gdf_dataf)
    dataf["dist_nearest_sub"] = pd.to_numeric(dataf["dist_nearest_sub"])
    print("\n")

    return dataf


def add_weather_features(dataf, type='req'):
    '''
    This function adds 4 new columns about the weather

    * maximum temperature in that day
    * minimum temperature in that day
    * type of weather: ['q', 'dy', 'dyq', 'qdy', 'xq', 'xydy']
    * wind
    '''
    with open("data/external/weather.json", 'r') as f:
        weather_dict = json.load(f)

    if 'req_date' not in dataf:
        dataf[type + '_time'] = pd.to_datetime(dataf[type + '_time'])
        dataf[type + '_date'] = dataf['req_time'].dt.strftime('%m-%d')

    dataf['max_temp'] = dataf['req_date'].apply(lambda r: weather_dict[r]['max_temp'])
    dataf['min_temp'] = dataf['req_date'].apply(lambda r: weather_dict[r]['min_temp'])
    dataf['weather'] = dataf['req_date'].apply(lambda r: weather_dict[r]['weather'])
    dataf['wind'] = dataf['req_date'].apply(lambda r: weather_dict[r]['wind'])

    dataf["max_temp"] = pd.to_numeric(dataf["max_temp"])
    dataf["min_temp"] = pd.to_numeric(dataf["min_temp"])
    dataf["wind"] = pd.to_numeric(dataf["wind"])

    dataf = dataf.join(pd.get_dummies(dataf["weather"]))
    dataf = dataf.drop('weather', axis=1)

    weather_only_in_train = ['xydy', 'xq']
    for el in weather_only_in_train:
        if el not in dataf:
            dataf[el] = 0

    weather_cols = ['dy', 'dyq', 'q', 'qdy', 'xq', 'xydy']
    dataf = dataf.rename(lambda x: 'weather_' + x if x in weather_cols else x, axis=1)

    return dataf


def assign_districts(df_train):
    # Euclidean Distance Calculation
    def dist(a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)

    # Get coordinates of all points and generate one single arrays
    def get_points(df, x, y):
        f1 = df[x].values
        f2 = df[y].values
        point_array = np.array(list(zip(f1, f2)))
        return point_array

    # Set each district center as centroids and generate one single centroid array
    def set_centroids(x, y):

        if os.path.isfile("../data/external/districts/beijing_districts.csv"):
            df_beijing_districts = pd.read_csv("../data/external/districts/beijing_districts.csv")
        else:
            sys.exit(-1)

        x1 = df_beijing_districts[x].values
        y1 = df_beijing_districts[y].values
        centroids = np.array(list(zip(x1, y1)))
        return centroids

    # Locate existing column (ex_column) in data frame and add new_column filled with NaN right behind it
    def preprocess_districts(df, ex_column, new_column):
        add_after_this_column = df.columns.get_loc(ex_column)
        new_column_position = add_after_this_column + 1

        df.insert(new_column_position, new_column, np.nan, True)
        return df

    # Assign each point of points to its closest district by calculating the distance to each value in centroid
    # Result in column
    def assign_points(df, column, points, centroids):

        # Cluster array filled with 0
        clusters = np.zeros(len(points))

        for p in range(len(points)):

            if p % 100 == 0:
                print("Processing row {}".format(str(p)), end="\r")

            # Calculate distance to all centroids, choose the smallest to assign cluster
            distances = dist(points[p], centroids)
            cluster = np.argmin(distances)
            clusters[p] = cluster
            df.loc[df.index == p, column] = column + str(cluster)

        return df

    def one_hot(df, column):
        df = df.join(pd.get_dummies(df[column]))
        return df

    # Generate one array with districts of unique data frame
    def get_district(df):
        array = df['o_district'].values
        split_array = []
        districts = []

        for each in array:
            split = each.split(', ')
            split_array.append(split)

        for nr in range(len(split_array)):
            z = split_array[nr][0].split('o_district_')

            if z[1] not in districts:
                districts.append(z[1])

        districts.sort()
        return districts

    def calculate_distances(df, points, centroids):
        # Add new column filled with NaN for each district
        districts = get_district(df)

        for district in districts:
            df.loc[:, 'o_distance_' + district] = np.nan

        for p in range(len(points)):

            if p % 100 == 0:
                print("Processing row {}".format(str(p)), end="\r")

            for i in districts:
                i = int(i)
                center = centroids[i]

                distance = dist(P[p], center)
                df.loc[p, 'o_distance_' + str(i)] = distance
        return df

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

    # Calculate distances to each districts
    df_train = calculate_distances(df_train, P, C)

    if not os.path.isdir("../data/external"):
        os.mkdir("../data/external")
    if not os.path.isdir("../data/external/districts"):
        os.mkdir("../data/external/districts")

    df_train.to_pickle("../data/external/districts/train_all_first_districts.pickle")

    return df_train


def add_features(df):
    print("Adding coordinate features")
    df = preprocess_coordinates(df)
    print("Adding subway features")
    df = add_dist_nearest_subway(df)
    df = pd.DataFrame(df)
    df = df.drop(['o_long', 'o_lat', 'd_long', 'd_lat'], axis=1)
    print("Adding time features")
    df = time_features(df)
    print("Adding public holidays features")
    df = add_public_holidays(df)
    print("Adding weather features")
    df = add_weather_features(df)
    print("Assign districts")
    df = assign_districts(df)
    return df


def main():
    
    print("Reading dataframes")
    df_train = pd.read_csv("data/raw/data_set_phase1/train_queries.csv")
    # df_test = pd.read_csv("data/raw/data_set_phase1/test_queries.csv")
    
    print("Adding features in df_train")
    df_train = add_features(df_train)

    print("Added all features in df_train")
    df_train.to_pickle("data/external_features/external.pickle")

    ''' DEPRECATED
    print("Adding features in df_test")
    df_test = add_features(df_test)

    print("Added all features in df_test")
    df_test.to_pickle("data/external_features/test_external.pickle")
    '''
    
    return


if __name__ == "__main__":
    main()

