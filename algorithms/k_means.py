import numpy as np
import random


def k_means(datapoints : np.ndarray, k : int):
    """
    Runs K-mean clustering algorithm on given datapoints

    Args:
        datapoints (np.ndarray): Datapoints as numpy array with
            shape of (m,n) that m is number of datapoints and n
            is dimensions of each datapoint
        k (int): Number of clusters

    Returns:
        clusters (np.ndarray): Cluster index of each datapoint
        centers (np.ndarray): Center of clusters
    """
    if k > datapoints.shape[0] :
        raise Exception("K can not be bigger than number of datapoints")
    
    # initiate clusters 
    clusters = np.array([0 for i in range(datapoints.shape[0])])

    # datapoint random choice for centeroids
    centers = np.array(random.sample(datapoints.tolist(), k))

    # first iteration of kmeans
    before_assign = clusters.copy()
    clusters = assign(datapoints, centers)
    centers = update_centers(datapoints, clusters, k)
    
    # iterate util convergence
    while not is_converged(before_assign, clusters) :
        before_assign = clusters.copy()
        clusters = assign(datapoints, centers)
        centers = update_centers(datapoints, clusters, k)

    return clusters, centers


def is_converged(clusters_1 : np.ndarray, clusters_2 : np.ndarray):
    """
    Checks that 2 set of cluster indexes are identical or no

    Args:
        clusters_1 (np.ndarray): Numpy array of cluster indexes
        clusters_2 (np.ndarray): Numpy array of cluster indexes

    Returns:
        boolean
    """
    clusters_diff = (clusters_1 - clusters_2) ** 2

    return clusters_diff.sum() == 0


def assign(datapoints : np.ndarray, centers : np.ndarray):
    """
    Updates assigned cluster index to each datapoint

    Args: 
        datapoints (np.ndarray): Datapoints as numpy array with
                shape of (m,n) that m is number of datapoints and n
                is dimensions of each datapoint
        centers (np.ndarray): Ceneters of clusters with shape of (k,n) 
                that k is number of clusters and n is dimensions of each 
                datapoint

    Returns:
        Numpy array of cluster indexes with shape of (1,n)
    """
    distance_matrix = np.zeros((datapoints.shape[0], centers.shape[0]))
    
    for i in range(datapoints.shape[0]):
        datapoint_distance = ((datapoints[i] - centers) ** 2).sum(axis=1)
        distance_matrix[i] = datapoint_distance

    return distance_matrix.argmin(axis=1)


def update_centers(datapoints : np.ndarray, clusters : np.ndarray, k : int) :
    """
    Updates center of clusters

    Args:
        datapoints (np.ndarray): Datapoints as numpy array with
                shape of (m,n) that m is number of datapoints and n
                is dimensions of each datapoint
        centers (np.ndarray): Ceneters of clusters with shape of (k,n) 
                that k is number of clusters and n is dimensions of each 
                datapoint
        k (int): Number of clusters

    Returns:
        Numpy array of new cluster centers
    """
    centers = np.zeros((k, datapoints.shape[1]))
    for i in range(k):

        cluster_mask = clusters == i
        cluster_datapoints = datapoints[cluster_mask]
        centers[i] = cluster_datapoints.mean(axis=0)

    return centers


if __name__ == "__main__" : 

    data = [
        [1, 2, 1],
        [1, 3, 0],
        [3, 4, 2],
        [3, 3, 2],
        [10, 10, 7],
        [11, 11, 12],
        [10, 11, 9],
        [20, 23, 45],
        [23, 22, 33]
    ]

    print(k_means(np.array(data), 3))



