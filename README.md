### bldg_point_clustering

**PyPi Package:** <https://pypi.org/project/bldg-point-clustering/>

**Docs:** <https://bldg-point-clustering.readthedocs.io/en/latest/>

## Introduction

A Python 3.5+ wrapper for clustering building point labels
using KMeans, DBScan, and Agglomerative clustering.

## Installation

Using pip for Python 3.5+ run:

```bash
$ pip install bldg_point_clustering
```

## Quick Start

Instantiate Featurizer object and get featurized Pandas DataFrame.

Instantiate Cluster object and pass in featurized
DataFrame to. Then, call a clustering method with the
appropriate parameters.

Use the plot3D function in the Plotter to create a
3D plot of metrics returned by any of the clustering trials.

## Example Usage

Running one iteration of the KMeans algorithm.

```python
import pandas as pd
import numpy as np
from bldg_point_clustering.cluster import Cluster
from bldg_point_clustering.featurizer import Featurizer

filename = "GBSF"

df = pd.read_csv("./datasets/" + filename + ".csv")

first_column = df.iloc[:, 0]

f = Featurizer(filename, corpus=first_column)

featurized_df = f.bag_of_words()

c = Cluster(df, featurized_df)

clustered_df = c.kmeans(n_clusters=300, plot=True, to_csv=True)

metrics = c.get_metrics_df()

avg_levenshtein_score = np.mean(c.get_levenshtein_scores())
```

Running several iterations of the KMeans algorithm.

```python
from bldg_point_clustering.plotter import plot_3D

c.kmeans_trials()

metrics = c.get_metrics_df()

plot_3D(metrics, "n_clusters", "Avg Levenshtein Score", "Silhouette Score")
```
This process is similar for DBScan and Agglomerative. 