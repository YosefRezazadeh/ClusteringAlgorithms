import random

import numpy as np

import algorithms
from utils.cluster_draw import cluster_draw


def random_data_generator(number_of_datapoint: int, x_max: float, y_max: float):
    random_array = np.random.random((number_of_datapoint, 2))
    random_array[:, 0] *= x_max
    random_array[:, 1] *= y_max

    return random_array


if __name__ == "__main__":

    data = random_data_generator(500, 12, 12)
    clusters, centers = algorithms.k_means(data, 4)
    cluster_draw(data, clusters, "k_means.jpg")
    clusters, centers = algorithms.mean_shift_clustering(data, 2)
    cluster_draw(data, clusters, "mean_shift.jpg")
    clusters = algorithms.dbscan(data, 2, 3)
    cluster_draw(data, clusters, "dbscan.jpg")
