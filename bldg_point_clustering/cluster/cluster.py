import numpy as np 
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from bldg_point_clustering.plotter import plot_kmeans, plot_dbscan, plot_agglomerative
from bldg_point_clustering.metrics import levenshtein_metric, silhouette_metric

class Cluster:

    """ Creates a Cluster object instance.

    :param df: The Pandas DataFrame of the original data.
    :param featurized_df: The Pandas DataFrame of the featurized data

    """

    def __init__(self, df, featurized_df):
        
        assert isinstance(df, pd.DataFrame), "Invalid dataframe."
        assert isinstance(featurized_df, pd.DataFrame), "Invalid featurized dataframe."

        self.df = df
        self.X = featurized_df
        self.__method = ""
        self.__cluster_instance = None
        self.__cluster_fit_instance = None 
        self.__clustered_df = None
        self.__metrics_df = None
    
    def kmeans(self, n_clusters=2, max_iter=300, plot=False, levenshtein_min_samples=50, to_csv=False): 
        """ Runs one iteration of the kmeans clustering algorithm 

        :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
        :param max_iter: Maximum number of iterations of the k-means algorithm for a single run.
        :param plot: Boolean value indicating whether to plot clusters.
        :param levenshtein_min_samples: Minimum number of elements in a cluster for the cluster to be counted for calculating the Levenshtein score.
        :param to_csv: Boolean value indicating whether to export clusters to a csv file.
        :return: Pandas DataFrame with a column for each cluster.

        """

        assert n_clusters > 0, "Must have 1 or more clusters"

        print("KMeans Clustering")
        print("----------------------")
        print("n clusters:", str(n_clusters))

        self.__cluster_instance = KMeans(n_clusters=n_clusters, max_iter=max_iter)

        if plot:
            plot_kmeans(X=self.X, n_clusters=n_clusters)
        
        self.__clustered_df = self.__accumulate_clusters_to_df()
        
        if to_csv:
            self.__clustered_df.to_csv("kmeans_" + str(n_clusters) + ".csv", index=False)
        
        metrics = self.__calculate_kmeans_metrics(n_clusters=n_clusters, levenshtein_min_samples=levenshtein_min_samples)
        
        if self.__method != "kmeans trials":
            self.__metrics_df = pd.DataFrame(columns=list(metrics.keys()))

        self.__metrics_df = self.__metrics_df.append(metrics, ignore_index=True)
        self.__method = "kmeans"

        return self.__clustered_df
        
    def dbscan(self, eps=2, min_samples=10, plot=False, levenshtein_min_samples=50, to_csv=False):
        """ Runs one iteration of the dbscan clustering algorithm 

        :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster.
        :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
        :param levenshtein_min_samples: Minimum number of elements in a cluster for the cluster to be counted for calculating the Levenshtein score.
        :param plot: Boolean value indicating whether to plot clusters.
        :param to_csv: Boolean value indicating whether to export clusters to a csv file.
        :return: Pandas DataFrame with a column for each cluster.

        """

        assert eps >= 0, "Must have eps >= 0"

        print("DBSCAN Clustering")
        print("----------------------")
        print("eps:", str(eps))
        print("min samples:", str(min_samples))

        self.__cluster_instance = DBSCAN(eps=eps, min_samples=min_samples)

        if plot:
            plot_dbscan(X=self.X, eps=eps, min_samples=min_samples)

        self.__clustered_df = self.__accumulate_clusters_to_df()

        if to_csv:
            self.__clustered_df.to_csv("dbscan_eps" + str(eps) + "_minsamples" + str(min_samples)+".csv", index=False)

        metrics = self.__calculate_dbscan_metrics(eps=eps, min_samples=min_samples, levenshtein_min_samples=levenshtein_min_samples)
        
        if self.__method != "dbscan trials":
            self.__metrics_df = pd.DataFrame(columns=list(metrics.keys()))

        self.__metrics_df = self.__metrics_df.append(metrics, ignore_index=True)
        self.__method = "dbscan"

        return self.__clustered_df
    
    def agglomerative(self, n_clusters=2, plot=False, levenshtein_min_samples=50, to_csv=False):
        """ Runs one iteration of the agglomerative/hierarchical clustering algorithm 

        :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
        :param plot: Boolean value indicating whether to plot clusters.
        :param levenshtein_min_samples: Minimum number of elements in a cluster for the cluster to be counted for calculating the Levenshtein score.
        :param to_csv: Boolean value indicating whether to export clusters to a csv file.
        :return: Pandas DataFrame with a column for each cluster.

        """

        assert n_clusters > 0, "Must have 1 or more clusters"

        print("Agglomerative Clustering")
        print("----------------------")
        print("n clusters:", str(n_clusters))
        
        self.__cluster_instance = AgglomerativeClustering(n_clusters=n_clusters)
        
        if plot:
            plot_agglomerative(X=self.X, n_clusters=n_clusters)

        self.__clustered_df = self.__accumulate_clusters_to_df()
        
        if to_csv:
            self.__clustered_df.to_csv("agglomerative_" + str(n_clusters) + ".csv", index=False)
        
        metrics = self.__calculate_agglomerative_metrics(n_clusters=n_clusters, levenshtein_min_samples=levenshtein_min_samples)
        
        if self.__method != "agglomerative trials":
            self.__metrics_df = pd.DataFrame(columns=list(metrics.keys()))
        
        self.__metrics_df = self.__metrics_df.append(metrics, ignore_index=True)
        self.__method = "agglomerative"        
        
        return self.__clustered_df
    
    def kmeans_trials(self, min_clusters=2, max_clusters=10, step=3, levenshtein_min_samples=50, plot=False):
        """ Runs multiple iterations/trials of the kmeans clustering algorithm starting from 'min_clusters'
        and adding 'step' each time until the number of clusters reaches 'max_clusters'. Calculates the metrics
        below at each iteration.

        :param min_clusters: The lowest number of clusters to start with
        :param max_clusters: The maximum number of clusters to end with
        :param step: The number of clusters to increment by for each iteration
        :param levenshtein_min_samples: Minimum number of elements in a cluster for the cluster to be counted for calculating the Levenshtein score.
        :param plot: Boolean value indicating whether to plot clusters.
        :return: Pandas DataFrame with all the metrics for each clustering iteration. Metrics include: Number of Clusters, Avg Levenshtein Score, STD Levenshtein Score, Min Levenshtein Score, Max Levenshtein Score, Silhouette Score, SSE, MSE, RMSE, Average Cluster Size, STD Cluster Size

        """

        print("Minimum # of Clusters:", min_clusters)
        print("Maximum # of Clusters:", max_clusters)
        print("Cluster Step:", step)
        print("Minimum samples in cluster for Levenshtein score:", levenshtein_min_samples)
        print("----------------------")
        
        try:
            n_clusters = min_clusters

            while n_clusters <= max_clusters:
                self.__clustered_df = self.kmeans(n_clusters=n_clusters, plot=plot)
                self.__method = "kmeans trials"

                n_clusters += step

                print("----------------------")
        
        except Exception as e:
            print("An error occurred:", e)

        return self.__metrics_df
    
    def dbscan_trials(self, min_eps=0.2, max_eps=1, eps_step=0.2, start_min_samples=10, max_min_samples=30, min_samples_step=5, levenshtein_min_samples=50, plot=False):
        """ Runs multiple iterations/trials of the dbscan clustering algorithm starting from 'min_eps'
        and adding 'eps_step' each time until the number of clusters reaches 'max_eps'. For each iteration of an eps value,
        several iterations will be run starting with 'start_min_samples' up to 'max_min_samples' incremented by 'min_samples_step' at each iteration.

        :param min_eps: The lowest maximum distance between two samples for one to be considered as in the neighborhood of the other.
        :param max_eps: The highest maximum distance between two samples for one to be considered as in the neighborhood of the other.
        :param eps_step: The amount of eps to increment by for each iteration
        :param start_min_samples: The lowest amount of the number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        :param max_min_samples: The highest amount of the number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        :param min_samples_step: The number of min_samples to increment by for each iteration
        :param levenshtein_min_samples: Minimum number of elements in a cluster for the cluster to be counted for calculating the Levenshtein score.
        :param plot: Boolean value indicating whether to plot clusters.
        
        :return: Pandas DataFrame with all the metrics for each clustering iteration. Metrics include: EPS, Min Samples, Avg Levenshtein Score, STD Levenshtein Score, Min Levenshtein Score, Max Levenshtein Score, Estimated # of Clusters, Estimated # of Noise/Outlier Points, Silhouette Score, Average Cluster Size, STD Cluster Size

        """

        print("Minimum eps:", min_eps)
        print("Maximum eps:", max_eps)
        print("eps step:", eps_step)
        print("Lowest minimum # of samples in cluster:", start_min_samples)
        print("Highest minimum # of samples in cluster:", max_min_samples)
        print("Minimum samples in cluster for Levenshtein score:", levenshtein_min_samples)
        print("----------------------")
        
        try:
            eps = min_eps

            while eps <= max_eps:
                num_samples = start_min_samples

                while num_samples <= max_min_samples:
                    self.__clustered_df = self.dbscan(eps=eps, min_samples=num_samples, plot=plot)
                    self.__method = "dbscan trials"

                    num_samples += min_samples_step

                eps += eps_step

                print("----------------------")
    
        except Exception as e:
            print("An error occurred:", e)
                
        return self.__metrics_df
    
    def agglomerative_trials(self, min_clusters=2, max_clusters=10, step=3, levenshtein_min_samples=50, plot=False):
        """ Runs multiple iterations/trials of the agglomerative/hierarchical clustering algorithm starting from 'min_clusters'
        and adding 'step' each time until the number of clusters reaches 'max_clusters'. Calculates the metrics
        below at each iteration.

        :param min_clusters: The lowest number of clusters to start with
        :param max_clusters: The maximum number of clusters to end with
        :param step: The number of clusters to increment by for each iteration
        :param levenshtein_min_samples: Minimum number of elements in a cluster for the cluster to be counted for calculating the Levenshtein score.
        :param plot: Boolean value indicating whether to plot clusters.
        :return: Pandas DataFrame with all the metrics for each clustering iteration. Metrics include: Number of Clusters, Avg Levenshtein Score, STD Levenshtein Score, Min Levenshtein Score, Max Levenshtein Score, Silhouette Score, Average Cluster Size, STD Cluster Size

        """
        
        print("Minimum # of Clusters:", min_clusters)
        print("Maximum # of Clusters:", max_clusters)
        print("Cluster Step:", step)
        print("Minimum samples in cluster for Levenshtein score:", levenshtein_min_samples)
        print("----------------------")
        
        try:
            n_clusters = min_clusters

            while n_clusters <= max_clusters:

                self.__clustered_df = self.agglomerative(n_clusters=n_clusters, plot=plot)
                self.__method = "agglomerative trials"

                n_clusters += step

                print("----------------------")
                
        except Exception as e:
            print("An error occurred:", e)
    
        return self.__metrics_df

    def get_levenshtein_scores(self, min_samples=50):
        """ Calculate levenshtein score of each cluster by averaging the levenshtein scores of all pairwise strings in each cluster and returns a
        score for each cluster. This will calculate levenshtein scores for the last run clustering algorithm.

        :param min_samples: Minimum number of elements in a cluster for the cluster to be counted for calculating the Levenshtein score.
        :return: Array of Levenshtein scores of each cluster respectively

        """

        if self.__method == "dbscan":
            return levenshtein_metric(self.__clustered_df.drop(columns=[-1]), min_samples=min_samples)
        
        return levenshtein_metric(self.__clustered_df, min_samples=min_samples)
    
    def get_metrics_df(self):
        """ Gets the Pandas dataframe containing the metrics of the last run clustering algorithm.

        :return: Pandas DataFrame containing the metrics of the last run clustering algorithm.

        """

        return self.__metrics_df
    def get_clustered_df(self):
        """ Gets the Pandas dataframe of the data sorted into its respective clusters after running
        one of the clustering algorithms.

        :return: Pandas DataFrame of each column representing a cluster.

        """

        return self.__clustered_df
    
    def get_cluster_instance(self):
        """ Gets the Sklearn Object of the previously called clustering algorithm.

        :return: Sklearn Object of the previously called clustering algorithm.

        """

        return self.__cluster_instance
    
    def get_cluster_fit_instance(self):
        """ Gets the Sklearn Object of the previously called clustering algorithm after fitting the data.

        :return: Sklearn Object of the previously called clustering algorithm after fitting the data.

        """

        return self.__cluster_fit_instance
    
    def __calculate_kmeans_metrics(self, n_clusters, levenshtein_min_samples):
        """ Calculates metrics for the last run kmeans clustering algorithm.

        :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
        :param levenshtein_min_samples: Minimum number of elements in a cluster for the cluster to be counted for calculating the Levenshtein score.
        :return: Dictionary of metrics

        """

        levenshtein_scores = self.get_levenshtein_scores(min_samples=levenshtein_min_samples)
        sse = self.__cluster_instance.inertia_
        mse = sse / self.X.shape[0]
        rmse = np.sqrt(mse)
        silhouette = silhouette_metric(X=self.X, labels=self.__cluster_instance.labels_)

        metrics = {
            "N-Clusters":n_clusters, 
            "Avg Levenshtein": np.mean(levenshtein_scores[~np.isnan(levenshtein_scores)]),
            "STD Levenshtein": np.std(levenshtein_scores[~np.isnan(levenshtein_scores)]),
            "Min Levenshtein": np.min(levenshtein_scores[~np.isnan(levenshtein_scores)]), 
            "Max Levenshtein": np.max(levenshtein_scores[~np.isnan(levenshtein_scores)]),
            "Avg Cluster Size": np.average(self.__clustered_df.count()),
            "STD Cluster Size": np.std(self.__clustered_df.count()),
            "Silhouette Score": silhouette,
            "SSE": sse, 
            "MSE": mse,
            "RMSE": rmse
        }

        return metrics

    def __calculate_dbscan_metrics(self, eps, min_samples, levenshtein_min_samples):
        """ Calculates metrics for the last run dbscan clustering algorithm.

        :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
        :param levenshtein_min_samples: Minimum number of elements in a cluster for the cluster to be counted for calculating the Levenshtein score.
        :return: Dictionary of metrics

        """

        clustered_df_without_outliers = self.__clustered_df.drop(columns=[-1])
        levenshtein_scores = self.get_levenshtein_scores(min_samples=min_samples + levenshtein_min_samples)
        n_clusters = len(set(self.__cluster_fit_instance.labels_)) - (1 if -1 in self.__cluster_fit_instance.labels_ else 0)
        n_noise = list(self.__cluster_fit_instance.labels_).count(-1)
        silhouette = silhouette_metric(X=self.X, labels=self.__cluster_instance.labels_)
        
        #print(levenshtein_scores, len(levenshtein_scores), n_clusters, np.mean(levenshtein_scores[~np.isnan(levenshtein_scores)]))

        metrics = {
            "EPS":eps,
            "Min Samples": min_samples,
            "Avg Levenshtein": np.mean(levenshtein_scores[~np.isnan(levenshtein_scores)]),
            "STD Levenshtein": np.std(levenshtein_scores[~np.isnan(levenshtein_scores)]),
            "Min Levenshtein": np.min(levenshtein_scores[~np.isnan(levenshtein_scores)]), 
            "Max Levenshtein": np.max(levenshtein_scores[~np.isnan(levenshtein_scores)]),
            "Estimated # of Clusters": n_clusters,
            "Estimated # of Noise Points": n_noise,
            "Avg Cluster Size": np.average(clustered_df_without_outliers.count()),
            "STD Cluster Size": np.std(clustered_df_without_outliers.count()),
            "Silhouette Score": silhouette
        }

        return metrics
    
    def __calculate_agglomerative_metrics(self, n_clusters, levenshtein_min_samples):
        """ Calculates metrics for the last run agglomerative clustering algorithm.

        :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
        :param levenshtein_min_samples: Minimum number of elements in a cluster for the cluster to be counted for calculating the Levenshtein score.
        :return: Dictionary of metrics

        """
        
        levenshtein_scores = self.get_levenshtein_scores(min_samples=levenshtein_min_samples)

        silhouette = silhouette_metric(X=self.X, labels=self.__cluster_instance.labels_)

        metrics = {
            "N-Clusters":n_clusters, 
            "Avg Levenshtein": np.mean(levenshtein_scores[~np.isnan(levenshtein_scores)]),
            "STD Levenshtein": np.std(levenshtein_scores[~np.isnan(levenshtein_scores)]),
            "Min Levenshtein": np.min(levenshtein_scores[~np.isnan(levenshtein_scores)]), 
            "Max Levenshtein": np.max(levenshtein_scores[~np.isnan(levenshtein_scores)]),
            "Avg Cluster Size": np.average(self.__clustered_df.count()),
            "STD Cluster Size": np.std(self.__clustered_df.count()),
            "Silhouette Score": silhouette
        }

        return metrics

    def __accumulate_clusters_to_df(self):
        """ Accumulates all the clusters found from the previously called clustering algorithm into one dataframe.

        :return: Pandas DataFrame of each column representing a cluster.

        """

        self.__cluster_fit_instance = self.__cluster_instance.fit(self.X)

        labels = self.__cluster_fit_instance.labels_

        clusters = dict.fromkeys(np.unique(labels), np.array([]))
    
        for i, label in enumerate(labels):
            clusters[label] = np.append(clusters[label], self.df.iloc[i].values[0])

        dfs = []
        
        for key in clusters:
            d = {}
            d[key] = clusters[key]
            
            dfs.append(pd.DataFrame(d))
            
        return pd.concat(dfs, axis=1)
