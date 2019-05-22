import sys
import pandas as pd
import logging
from shapely.ops import nearest_points
from shapely.geometry import Point
import geopandas as gpd

# Initialize logger
logger = logging.getLogger(__name__)

def time_features(dataf, type = 'req'):
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
    #dataf[[type + '_month',type + '_day']] = dataf[type + '_date'].str.split("-",expand=True,).astype(int)
    #dataf = dataf.drop(type + '_date', axis=1)
    dataf[type + '_hour'] = dataf[type + '_time'].dt.hour
    
    # [2] Add a Binary, indicating whether its weekend or not! [ERROR]
    dataf[type + '_weekend'] = dataf[type + '_time'].dt.day_name().apply(lambda x: 1 if x in ["Friday", "Saturday"] else 0)
    
    # [3] Add binary, NIGHT / EVENING / DAY Column
    dataf[type + '_night'] = dataf[type + '_hour'].apply(lambda x: 1 if x <= 7 else 0 )
    dataf[type + '_day']   = dataf[type + '_hour'].apply(lambda x: 1 if x in range(8,18) else 0)
    dataf[type + '_evening']   = dataf[type + '_hour'].apply(lambda x: 1 if x > 18 else 0)

   # Print some Info
   # logger.debug('5 new columns created: month, day, weekend, hour, and hour_bin')

    return dataf

def add_public_holidays(dataf):
    '''
    Creates 1 new time column on the dataframe, a binary column stating if the date is a public holiday or not.
    '''
    df_holidays = pd.read_csv("../data/external/CN_holidays_summary.csv", index_col=False)
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
    Creates 1 new column with the distance to the nearest subway station (from subways.csv)
    '''
    def extract_Points_df(df, lat_column, long_column, crs={'init', 'epsg:4326'}):
        df_copy = df.copy()
        geometry = [Point(xy) for xy in zip(df_copy[long_column], df_copy[lat_column])]
        Points = gpd.GeoDataFrame(df_copy, crs=crs, geometry=geometry)
        return Points

    df_subways = pd.read_csv("../data/external/subways.csv", index_col=False).round(2)

    if 'o_lat' not in dataf or 'o_long' not in dataf:
        logger.error("The dataframe doesn't have the coordinates in the correct format. They need to be 'o_lat' and 'o_long'.")

    gdf_subways = extract_Points_df(df_subways, lat_column="o_lat", long_column="o_long")
    gdf_dataf = extract_Points_df(dataf, lat_column="o_lat", long_column="o_long")

    pts3 = gdf_subways.geometry.unary_union
    
    # https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe
    def near(point, pts=pts3):
        print("Processing row {}".format(str(near.i + 1)), end="\r")
        near.i += 1
        # find the nearest point and return the corresponding Place value
        nearest = gdf_subways.geometry == nearest_points(point, pts)[1]
        return "%.3f" % (gdf_subways[nearest].geometry.get_values()[0].distance(point)*10.0)
    
    near.i = 0
    gdf_dataf['dist_nearest_sub'] = gdf_dataf.apply(lambda row: near(row.geometry, pts3), axis=1)
    gdf_dataf = gdf_dataf.drop('geometry', 1)

    return gdf_dataf
