import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from itertools import combinations
import Levenshtein

def levenshtein_metric(X, min_samples=0):
    """ Returns array of average levenshtein scores of each cluster by averaging the levenshtein scores of all pairwise strings in each cluster.

    :param X: Pandas DataFrame of clusters with their respective strings
    :param min_samples: Minimum number of elements in a cluster for the cluster to be counted for calculating the Levenshtein score.
    :return: Array of average Levenshtein scores of each cluster

    """

    keepcols = [c for c in X.columns if len(X[c].dropna()) >= min_samples]
    X = X[keepcols]

    return np.array([np.mean(np.array([Levenshtein.distance(comb[0], comb[1]) for comb in combinations(X[i].dropna(), 2) if type(comb[0]) == str and type(comb[1]) == str])) for i in X.columns])

def silhouette_metric(X, labels):
    """ Returns silhouette score of featurized Pandas DataFrame

    :param X: Pandas DataFrame of featurized data
    :param labels: The labeled cluster assigned to each string (array)
    :return: Silhouette Score (Floating point value between 0 and 1)

    """
            
    return silhouette_score(X=X, labels=labels)