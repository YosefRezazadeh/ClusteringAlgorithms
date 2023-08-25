import numpy as np
import random

def k_means(datapoints : np.ndarray, k : int):

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

    print(datapoints)
    print(clusters)


def is_converged(clusters_1 : np.ndarray, clusters_2 : np.ndarray):

    clusters_diff = (clusters_1 - clusters_2) ** 2

    return clusters_diff.sum() == 0


def assign(datapoints : np.ndarray, centers : np.ndarray):

    distance_matrix = np.zeros((datapoints.shape[0], centers.shape[0]))
    
    for i in range(datapoints.shape[0]):
        datapoint_distance = ((datapoints[i] - centers) ** 2).sum(axis=1)
        distance_matrix[i] = datapoint_distance

    return distance_matrix.argmin(axis=1)

def update_centers(datapoints : np.ndarray, clusters : np.ndarray, k : int) :
    
    centers = np.zeros((k, datapoints.shape[1]))
    for i in range(k):

        cluster_mask = clusters == i
        cluster_datapoints = datapoints[cluster_mask]
        centers[i] = cluster_datapoints.mean(axis=0)

    return centers


if __name__ == "__main__" : 

    data = [
        [1, 2],
        [1, 3],
        [3, 4],
        [3, 3],
        [10, 10],
        [11, 11],
        [10, 11],
        [20, 23],
        [23, 22]
    ]

    k_means(np.array(data), 3)



