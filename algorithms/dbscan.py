import numpy as np


def dbscan(datapoints: np.ndarray, epsilon: float, min_points: int):
    cluster = np.full((datapoints.shape[0], 1), -2)
    new_cluster_id = -1

    for i in range(datapoints.shape[0]):
        if cluster[i] == -2:
            # print("==============================")
            neighbors = get_neighbors(datapoints, datapoints[i], epsilon)
            # print(f"n = {neighbors}")
            if neighbors.shape[0] < min_points:
                cluster[i] = -1  # Noise
            else:
                new_cluster_id += 1
                cluster[i] = new_cluster_id

                for neighbor in neighbors:
                    neighbor_index = search_datapoint(datapoints, neighbor)
                    # print(neighbor, neighbor_index)
                    if cluster[neighbor_index] == -1:
                        cluster[neighbor_index] = new_cluster_id
                    if cluster[neighbor_index] >= 0:
                        continue
                    neighbors_of_neighbor = get_neighbors(datapoints, neighbor, epsilon)
                    cluster[neighbor_index] = new_cluster_id
                    if neighbors_of_neighbor.shape[0] > min_points:
                        neighbors = np.append(neighbors, neighbors_of_neighbor)

    return cluster[:, 0]


def get_neighbors(datapoints: np.ndarray, datapoint: np.ndarray, epsilon: float):
    distances = ((datapoints - datapoint) ** 2).sum(axis=1)
    in_window_datapoints_mask = distances <= epsilon ** 2
    in_window_datapoints = datapoints[in_window_datapoints_mask]
    in_window_datapoints = np.delete(in_window_datapoints, [search_datapoint(in_window_datapoints, datapoint)], axis=0)
    return in_window_datapoints


def search_datapoint(datapoints: np.ndarray, datapoint: np.ndarray):
    # print(f"Search for {datapoint}")
    if datapoints.shape[1] != datapoint.shape[0]:
        raise Exception("Dimension Mismatch")

    dims = datapoints.shape[1]
    result = np.full((1, datapoints.shape[0]), True)

    for i in range(dims):
        result = result & (datapoints[:, i] == datapoint[i])

    return np.where(result == True)[1][0]

