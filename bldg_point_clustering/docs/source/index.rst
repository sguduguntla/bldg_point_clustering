.. bldg_point_clustering documentation master file, created by
   sphinx-quickstart on Wed Dec  4 16:50:14 2019.

Welcome to bldg_point_clustering's documentation!
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Introduction
============

A Python 3.5+ wrapper for clustering building point labels
using KMeans, DBScan, and Agglomerative clustering.

Installation
============

Using pip for Python 3.5+ run:

.. code-block:: console

   $ pip install bldg_point_clustering

Quick Start
===========

Instantiate Featurizer object and get featurized Pandas DataFrame.

Instantiate Cluster object and pass in featurized
DataFrame to. Then, call a clustering method with the
appropriate parameters.

Use the plot3D function in the Plotter to create a
3D plot of metrics returned by any of the clustering trials.

Example Usage
=============

Running one iteration of the KMeans algorithm:

.. code-block:: python

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

Running several iterations of the KMeans algorithm:

.. code-block:: python

    from bldg_point_clustering.plotter import plot_3D

    c.kmeans_trials()

    metrics = c.get_metrics_df()

    plot_3D(metrics, "n_clusters", "Avg Levenshtein Score", "Silhouette Score")

This process is similar for DBScan and Agglomerative.

Featurizer
==========
.. autoclass:: bldg_point_clustering.featurizer.featurizer.Featurizer
   :members:
   :undoc-members:

Cluster
=======
.. autoclass:: bldg_point_clustering.cluster.cluster.Cluster
   :members:
   :undoc-members:

Plotter
=======
.. automodule:: bldg_point_clustering.plotter.plotter
   :members:
   :undoc-members:

Metrics
=======
.. automodule:: bldg_point_clustering.metrics.metrics
   :members:
   :undoc-members:

Index
=====

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`

.. toctree::
   :maxdepth: 2
   :caption: Contents:

