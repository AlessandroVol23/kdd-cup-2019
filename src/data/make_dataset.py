# -*- coding: utf-8 -*-
import json
import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from geopy.distance import distance
from pandas.io.json import json_normalize

# Initialize logger
logger = logging.getLogger(__name__)


def read_in_data(absolute_raw_data_path, all_datasets, train_set_only, test_set_only):
    """
        This function reads in all the data sets.
        returns: df_train_queries, df_train_plans, df_train_clicks, df_test_queries, df_test_queries, df_test_plans, df_profiles
    """
    # TODO: Unit test if shape is original shape

    # TODO: Check if path is absolute

    # Concat paths

    raw_dir = os.path.join(absolute_raw_data_path)
    logger.info("Raw folder path is: %s", raw_dir)

    data_set_path = raw_dir
    # data_set_path = os.path.join(raw_dir, "data_set_phase1")

    if all_datasets:
        df_profiles = pd.read_csv(os.path.join(data_set_path, "profiles.csv"))
        df_clicks = pd.read_csv(os.path.join(
            data_set_path, "train_clicks.csv"))
        df_train_plans = pd.read_csv(
            os.path.join(data_set_path, "train_plans.csv"))
        df_train_queries = pd.read_csv(
            os.path.join(data_set_path, "train_queries.csv"))
        df_test_queries = pd.read_csv(
            os.path.join(data_set_path, "test_queries.csv"))
        df_test_plans = pd.read_csv(
            os.path.join(data_set_path, "test_plans.csv"))
        return (
            df_profiles,
            df_clicks,
            df_train_plans,
            df_train_queries,
            df_test_queries,
            df_test_plans,
        )
    else if train_set_only:
        df_profiles = pd.read_csv(os.path.join(data_set_path, "profiles.csv"))
        df_clicks = pd.read_csv(os.path.join(
            data_set_path, "train_clicks.csv"))
        df_train_plans = pd.read_csv(
            os.path.join(data_set_path, "train_plans.csv"))
        df_train_queries = pd.read_csv(
            os.path.join(data_set_path, "train_queries.csv"))

        return (
            df_profiles, df_clicks, df_train_plans, df_train_queries
        )
    else if test_set_only:
        df_profiles = pd.read_csv(os.path.join(data_set_path, "profiles.csv"))
        df_clicks = pd.read_csv(os.path.join(
            data_set_path, "train_clicks.csv"))
        df_test_queries = pd.read_csv(
            os.path.join(data_set_path, "test_queries.csv"))
        df_test_plans = pd.read_csv(
            os.path.join(data_set_path, "test_plans.csv"))

        return (
            df_profiles, df_clicks, df_test_queries, df_test_plans
        )


def concat_data(df_train_plans, df_train_queries, df_test_queries, df_test_plans):
    """
        This function concats all train and test sets together to one set.
        returns: df_clicks, df_plans, df_queries
    """
    df_plans = df_train_plans.append(df_test_plans)
    logger.debug("df_plans shape %s", df_plans.shape)

    df_queries = df_train_queries.append(df_test_queries)
    logger.debug("df_queries.shape %s", df_queries.shape)

    return df_plans, df_queries


def clean_data(df_clicks, df_plans, df_queries):
    """
        This function cleans the dataset from null an na values
        returns: This function returns a clean dataset with handled na / null values
    """


def preprocess_queries(df_queries):
    df = df_queries.copy()

    df[["o_long", "o_lat"]] = df.o.str.split(",", 1, expand=True).astype(float)
    df.drop("o", axis=1, inplace=True)

    df[["d_long", "d_lat"]] = df.d.str.split(",", 1, expand=True).astype(float)
    df.drop("d", axis=1, inplace=True)
    return df


def unstack_plans(df_plans):
    df = df_plans.copy()

    df.plans = df.apply(
        lambda x: json.loads(
            '{"plans":'
            + x.plans
            + ',"sid":"'
            + str(x.sid)
            + '"'
            + ',"plan_time":'
            + '"'
            + str(x.plan_time)
            + '"}'
        ),
        axis=1,
    )

    df_unstacked = json_normalize(
        df.plans.values, "plans", ["sid", "plan_time"])
    df_unstacked.rename({"distance": "distance_plan"}, axis=1, inplace=True)
    return df_unstacked


def calculate_distance(df):
    df = df.assign(
        distance_query=(
            df.apply(
                lambda x: (distance([x.o_lat, x.o_long],
                                    [x.d_lat, x.d_long]).km),
                axis=1,
            )
        )
    )

    return df


def calc_distance(df):
    return df.apply(
        lambda x: (distance([x.o_lat, x.o_long], [x.d_lat, x.d_long]).km), axis=1
    )


def distance_function(coords_1, coords_2):
    """
        Function to calculate distance in km between two coordinates
        coords_1: (lat, long)
        coords_2: (lat, long)
        returns: distance in km
    """

    return distance(coords_1, coords_2).km


def preprocess_datatypes(df_plans, df_clicks, df_queries):

    df_plans.sid = df_plans.sid.astype(int)
    df_clicks.sid = df_clicks.sid.astype(int)
    df_queries.sid = df_queries.sid.astype(int)

    return df_plans, df_clicks, df_queries


def join_data_sets(df_plans, df_clicks, df_queries):
    """
        This function joins all datasets together.
    """

    df = pd.merge(df_clicks, df_plans, on="sid")
    df = pd.merge(df, df_queries, on="sid")

    return df


def preprocess_datatypes_after_join(df):
    """
        In this function I change the datatypes after the join.
    """
    df.price = pd.to_numeric(df.price)
    return df


def fill_missing_price(df, median=True, mean=False):
    """
        This function fills all missing values in price with the median value. 
    """
    df.loc[df.price.isna(), "price"] = df.price.median()
    return df


@click.command()
@click.argument("absolute_path_raw_folder")
@click.argument("output_file")
def main(absolute_path_raw_folder, output_file, all_datasets=True, train_set_only=False, test_set_only=False):
    """ 
        Script to construct dataset.
        Script looks in absolut_path_raw_folder for profiles.csv, train_clicks.csv, train_plans.csv, train_queries.csv, test_queries.csv, test_plans.csv. File will be saved as .pickle file.
        absolute_path_raw_folder: Path to raw data folder. Typically /home/xxx/repo/data/raw
        output_file: Path to output file. Typically /home/xxx/repo/data/processed/df_features.pickle
    """
    logger.info("making final data set from raw data")
    # Should we concat all data sets? -> Parameter if we concat it or not

    logger.info("Filepath to raw folder is %s", absolute_path_raw_folder)
    logger.info("Filepath to processed folder is %s", output_file)

    # Read in data
    logger.info("Start reading in data...")

    if all_datasets:
        df_profiles, df_clicks, df_train_plans, df_train_queries, df_test_queries, df_test_plans = read_in_data(
            absolute_path_raw_folder, all_datasets, train_set_only, test_set_only
        )
    else if train_set_only:
        df_profiles, df_clicks, df_plans, df_queries = read_in_data(
            absolute_path_raw_folder, all_datasets, train_set_only, test_set_only
        )
    else if test_set_only:
        df_profiles, df_clicks, df_plans, df_queries = read_in_data(
            absolute_path_raw_folder, all_datasets, train_set_only, test_set_only
        )

    # Concat data
    logger.info("Start concating data...")
    if all_datasets:
        df_plans, df_queries = concat_data(
            df_train_plans, df_train_queries, df_test_queries, df_test_plans
        )
    # TODO: Unit test to check if shape is the same afterwards

    # Preprocess lat / long
    logger.info("Start preprocessing datatypes...")
    df_queries_pp = preprocess_queries(df_queries)

    # Calculate distance from origin to destination
    logger.info("Calculate distance from origin to destination")
    # df_queries_pp = calculate_distance(df_queries_pp)
    df_queries_pp["distance_query"] = calc_distance(df_queries_pp)

    # Unstack JSON plans
    logger.info("Start unstacking json plans")
    df_plans_pp = unstack_plans(df_plans)

    # Preprocess datatypes
    logger.info("Preprocess datatypes")
    df_plans_pp, df_clicks, df_queries_pp = preprocess_datatypes(
        df_plans_pp, df_clicks, df_queries_pp
    )

    # Join datasets
    logger.info("Join datasets")
    df_joined = join_data_sets(df_plans_pp, df_clicks, df_queries_pp)

    # Process datatypes after join
    logger.info("Preprocess datatypes after join")
    df_joined = preprocess_datatypes_after_join(df_joined)

    # Handle price column
    logger.info("Fill NA values in price with median")
    df_joined = fill_missing_price(df_joined)

    # Save processed dataset
    logger.info("Save preprocessed data in %s", output_file)
    df_joined.to_pickle(os.path.join(output_file))

    logger.info("Preprocessing DONE!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
