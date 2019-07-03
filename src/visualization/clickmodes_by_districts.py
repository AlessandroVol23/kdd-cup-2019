import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gf


def get_click_modes(df_districts):
    click_modes = df_districts.click_mode.unique()
    click_modes.sort()
    return click_modes


def get_districts(df_districts):
    get_districts = df_districts.o_district.unique()
    get_districts.sort()
    districts = []

    for each in get_districts:
        districts.append(each)

    return districts


def get_district_number(df_districts):
    array = df_districts['o_district'].values
    split_array = []
    districts = []

    for each in array:
        split = each.split(', ')
        split_array.append(split)

    for district in range(len(split_array)):
        z = split_array[district][0].split('o_district_')
        if z[1] not in districts:
            districts.append(z[1])

    districts.sort()
    return districts


# point_type is either 'origin' or 'destination'
def plot_click_modes_by_districts(df_districts, click_modes, districts, point_type):

    beijing_districts = pd.read_csv("../data/external/districts/beijing_districts.csv")

    for cm in click_modes:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w', 'orange', 'brown', 'olive', 'gold', 'pink', 'aqua', 'lime', 'beige',
                  'indigo']
        fig, ax = plt.subplots(figsize=(20, 20))
        plt.title(f'Click Mode {cm} by Districts')

        # Excerpt data frame by click mode
        new_df = df_districts[df_districts.click_mode == cm]

        # Change color for every district
        for each in districts:
            if point_type == 'origin':
                df = new_df[new_df.o_district == each]
                ax.scatter(df.o_lat, df.o_long, s=10, color=colors[each], alpha=0.5)
            elif point_type == 'destination':
                df = new_df[new_df.d_district == each]
                ax.scatter(df.d_lat, df.d_long, s=10, color=colors[each], alpha=0.5)

        # plot district centroids
        ax.scatter(beijing_districts.o_lat, beijing_districts.o_long, color='k', marker='*', s=300, alpha=1)

        # plot district name next to point
        for dis in range(len(beijing_districts)):
            name = beijing_districts['District'].iloc[dis]
            x = beijing_districts['o_lat'].iloc[dis] + 0.03
            y = beijing_districts['o_long'].iloc[dis]

            ax.annotate(name, xy=(x, y), size=10, weight='bold', color='white', backgroundcolor='k')

        # save each figure in directory
        fig.savefig(
            f'../data/external/districts/beijing_districts_clickmode_{cm}.png')
    return
