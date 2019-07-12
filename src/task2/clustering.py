from sklearn.cluster import KMeans, OPTICS
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


'''KMeans CLustering with Euclidean Distance
    - requires a df with PIDs
    - choose n_cluster
    - add new column with cluster_id'''


def kmeans_clustering(df, n_cluster):

    # Standardize the data
    df_std = StandardScaler().fit_transform(df)

    # Run local implementation of kmeans
    km = KMeans(n_clusters=n_cluster)
    km.fit(df_std)
    labels = km.labels_

    df[f'cluster{n_cluster}_id'] = np.nan
    col_name = f'cluster{n_cluster}_id'

    # Fill new created column with cluster assignments
    for each in range(len(df)):
        x = labels[each]
        df.loc[df.index == each, col_name] = str(x)

    # Create an array containing n arrays filled with PIDs of n cluster
    n = df.col_name.unique()
    cluster = []

    for each in n:
        cluster_no = str(each)
        array = np.array(df.pid.loc[df[col_name] == cluster_no])
        cluster.append(array)

    return df, cluster


'''Hierarchical Clustering with using linkage() and ward method
    - requires a df with PIDs
    - choose n_cluster to truncate dendrogram
    - add new column with cluster_id'''


def hierc_clustering(df, n_cluster):

    # Generate the linkage matrix
    cluster_matrix = linkage(df, 'ward')

    # Calculate full Dendrogram and saves it
    figure = plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('PIDs')
    plt.ylabel('Distance')
    dendrogram(
        cluster_matrix,
        leaf_rotation=90,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
    )
    plt.show()

    figure.savefig('data/task2/dendrogram.jpg')

    labels = fcluster(cluster_matrix, n_cluster, criterion='maxclust')
    col_name = f'cluster{n_cluster}_id'
    df[col_name] = np.nan

    for each in range(len(df)):
        x = labels[each]
        df.loc[df.index == each, col_name] = str(x)

    # Create an array containing n arrays filled with PIDs of n cluster
    n = df.col_name.unique()
    cluster = []

    for each in n:
        cluster_no = str(each)
        array = np.array(df.pid.loc[df[col_name] == cluster_no])
        cluster.append(array)

    return df, cluster


'''Optics Clustering: automatically generates n clusters
    - choose min_sample size
    - choose min_cluster_size
    - add new column with cluster_id'''


def optics_clustering(df, min_sample, min_cluster_size):

    clust = OPTICS(min_samples=min_sample, xi=.05, min_cluster_size=min_cluster_size)

    # Run the fit
    clust.fit(df)
    labels = clust.labels_[clust.ordering_]

    # Generate new column with cluster_id
    df['cluster_id'] = np.nan

    for each in range(len(df)):
        x = labels[each]
        df.loc[df.index == each, 'cluster_id'] = str(x)

    # Create an array containing n arrays filled with PIDs of n cluster
    n = df.col_name.unique()
    cluster = []

    for each in n:
        cluster_no = str(each)
        array = np.array(df.pid.loc[df['cluster_id'] == cluster_no])
        cluster.append(array)

    return df, cluster
