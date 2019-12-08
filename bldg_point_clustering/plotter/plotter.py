import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_samples
import plotly.express as px

def plot_3D(df, x, y, z):
    """ Runs one iteration of the agglomerative/hierarchical clustering algorithm 

    :param df: A dataframe of columns of numerical data
    :param x: The column of the dataframe to go on the x-axis (Column Name -> String)
    :param y: The column of the dataframe to go on the y-axis (Column Name -> String)
    :param z: The column of the dataframe to go on the z-axis (Column Name -> String)
    :return: 3D Plot of x, y, and z (using Plotly Express)

    """
    return px.scatter_3d(df, x=x, y=y, z=z)

def plot_silhouettes(X, labels):
    """ Finds and plots silhouette samples for each label

    :param X: A dataframe of the featurized data
    :param labels: The labeled cluster assigned to each string (array)

    """
    # Get silhouette samples
    silhouette_vals = silhouette_samples(X, labels)
    fig, (ax1) = plt.subplots(1)
    fig.set_size_inches(18, 7)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02)
    
    plt.show()

def plot_inertia(X, n_clusters=2):
    """ Finds and plots silhouette samples for each label

    :param X: A dataframe of the featurized data
    :param n_clusters: The number of clusters to plot the inertia (sum of squared error) for.

    """
    sse = []
    list_k = list(range(1, n_clusters))

    for k in list_k:
        km = KMeans(n_clusters=n_clusters)
        km.fit(X)
        sse.append(km.inertia_)

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance');

    plt.show()

def plot_kmeans(X, n_clusters=2):
    """ Plots the PCA (2 components) reduced data with kmeans clustering and n_clusters

    :param X: A dataframe of the featurized data
    :param n_clusters: The number of clusters to apply kmeans clustering and plot

    """

    reduced_data = PCA(n_components=2).fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters)
    clustering = kmeans.fit(reduced_data)
    labels = clustering.labels_
    centroids = clustering.cluster_centers_

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering (PCA-reduced data)\n'
            'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
    # Nicer Plot without boundaries
    fig, (ax2) = plt.subplots(1)
    fig.set_size_inches(18, 7)

    # Scatter plot of data colored with labels
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
    ax2.set_xlim([-10, 10])
    ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')

    plt.show()

def plot_dbscan(X, eps=0.2, min_samples=10):
    """ Plots the PCA (2 components) reduced data with dbscan clustering with epsilon (eps) and min_samples

    :param X: A dataframe of the featurized data
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.

    """

    reduced_data = PCA(n_components=2).fit_transform(X)

    ss = StandardScaler()
    reduced_data = ss.fit_transform(reduced_data)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clustering = dbscan.fit(reduced_data)
    labels = clustering.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    
    fig, (ax2) = plt.subplots(1)
    fig.set_size_inches(14, 7)

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = reduced_data[class_member_mask & core_samples_mask]
        print(len(xy[:, 0]))
        ax2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = reduced_data[class_member_mask & ~core_samples_mask]
        ax2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=10)
            
    plt.title('Estimated number of clusters: %d' % len(unique_labels))
    plt.show()

def plot_agglomerative(X, n_clusters=2):
    """ Plots the PCA (2 components) reduced data with agglomerative clustering with n_clusters

    :param X: A dataframe of the featurized data
    :param n_clusters: The number of clusters to apply dbscan clustering and plot

    """
    
    reduced_data = PCA(n_components=2).fit_transform(X)

    ss = StandardScaler()
    reduced_data = ss.fit_transform(reduced_data)

    agg = AgglomerativeClustering(n_clusters=n_clusters)
    clustering = agg.fit(reduced_data)
    labels = clustering.labels_
    y_pred = agg.fit_predict(reduced_data)

    fig, (ax2) = plt.subplots(1)
    fig.set_size_inches(18, 7)

    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
    #ax2.scatter(reduced_data[:,0], reduced_data[:,1],c=y_pred, cmap='Paired')
    plt.title("Agglomerative Clustering with " + str(n_clusters) + " clusters")
    plt.show()