import numpy as np


def mean_shift_clustering(datapoints : np.ndarray, bandwidth : float):

    density_centers = []
    for i in range(datapoints.shape[0]):
        density_centers.append(mean_shift(datapoints[i], datapoints, bandwidth).tolist())

    # density_centers = np.array(density_centers)
    clusters = []
    cluster_density_centers = []

    cluster_id = -1
    for i in range(len(density_centers)):
        if not density_centers[i] in cluster_density_centers:
            cluster_id += 1
            clusters.append(cluster_id)
            cluster_density_centers.append(density_centers[i])
        else:
            clusters.append(cluster_id)

    return np.array(clusters), np.array(density_centers)


def mean_shift(datapoint : np.ndarray, datapoints : np.ndarray, bandwidth : float):

    center = datapoint.copy()
    window_data, window_mean = get_window(datapoints, center, bandwidth)

    while not is_center_converged(center, window_mean):
        center = window_mean
        window_data, window_mean = get_window(datapoints, center, bandwidth)


    return window_mean


def get_window(datapoints : np.ndarray, center : np.ndarray, bandwidth : float):

    distances = ((datapoints - center) ** 2).sum(axis=1)
    in_window_datapoints_mask = distances <= bandwidth**2
    in_window_datapoints = datapoints[in_window_datapoints_mask]
    window_center = in_window_datapoints.mean(axis=0)

    return in_window_datapoints, window_center


def is_center_converged(center_1 : np.ndarray, center_2 : np.ndarray):

    center_distance = ((center_1 - center_2) ** 2).sum()

    return center_distance == 0


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

    print(mean_shift_clustering(np.array(data), 2))