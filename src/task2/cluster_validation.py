from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import numpy as np


'''Silhouette Analysis used to determine the degree of separation between clusters. For each sample:
    * 0 –> the sample is very close to the neighboring clusters.
    * 1 –> the sample is far away from the neighboring clusters.
    * -1 –> the sample is assigned to the wrong clusters.
    
    labels needed from KMeans Clustering'''


def silhouette_score(df, labels):

    df_std = StandardScaler().fit_transform(df)

    # Get silhouette samples
    silhouette_vals = silhouette_samples(df_std, labels)

    # Silhouette Score
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()

    # Get the average silhouette score
    avg_score = np.mean(silhouette_vals)

    return avg_score


'''Cophenetic Correlation Coefficient: compare the actual pairwise distances of all samples to those implied 
--> The closer to 1, the better the clustering preserves the original distances

    cluster_matrix from hierarchical clustering is needed'''


def ccc_validation(df, cluster_matrix):
    c, coph_dists = cophenet(cluster_matrix, pdist(df))
    return c, coph_dists
