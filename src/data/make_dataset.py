# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import os

# Initialize logger
logger = logging.getLogger(__name__)


def read_in_data(relative_raw_data_path):
    """
        This function reads in all the data sets.
        returns: df_train_queries, df_train_plans, df_train_clicks, df_test_queries, df_test_queries, df_test_plans, df_profiles
    """
    # TODO: Unit test if shape is original shape

    # Concat paths
    dirname = os.path.dirname(__file__)
    print(dirname)
    return null
    filename = os.path.join(dirname, relative_raw_data_path)

    df_profiles = pd.read_csv("../data/raw/data_set_phase1/profiles.csv")
    df_train_clicks = pd.read_csv(
        "../data/raw/data_set_phase1/train_clicks.csv")
    df_train_plans = pd.read_csv("../data/raw/data_set_phase1/train_plans.csv")
    df_train_queries = pd.read_csv(
        "../data/raw/data_set_phase1/train_queries.csv")

    df_test_queries = pd.read_csv(
        "../data/raw/data_set_phase1/test_queries.csv")
    df_test_plans = pd.read_csv("../data/raw/data_set_phase1/test_plans.csv")
    df_test_clicks = pd.read_csv("../data/raw/data_set_phase1/test_clicks.csv")
    return df_profiles, df_train_clicks, df_train_plans, df_train_queries, df_test_queries, df_test_plans, df_test_clicks


def concat_data(df_train_clicks, df_train_plans, df_train_queries, df_test_queries, df_test_plans, df_test_clicks):
    """
        This function concats all train and test sets together to one set.
        returns: df_clicks, df_plans, df_queries
    """
    df_clicks = df_train_clicks.append(df_test_clicks)
    df_plans = df_train_plans.append(df_test_plans)
    df_queries = df_train_queries.append(df_test_queries)

    return df_clicks, df_plans, df_queries


def clean_data(df_clicks, df_plans, df_queries):
    """
        This function cleans the dataset from null an na values
        returns: This function returns a clean dataset with handled na / null values
    """


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final data set from raw data')
    # Should we concat all data sets? -> Parameter if we concat it or not
    # TODO: Parameter to concat all data
    # TODO: Parameter to use profiles as features?

    # Read in data
    df_profiles, df_train_clicks, df_train_plans, df_train_queries, df_test_queries, df_test_plans, df_test_clicks = read_in_data()

    # TODO: Concat data
    df_clicks, df_plans, df_queries = concat_data(
        df_train_clicks, df_train_plans, df_train_queries, df_test_queries, df_test_plans, df_test_clicks)
    # TODO: Unit test to check if shape is the same afterwards

    # TODO: Clean data


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
