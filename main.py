import random

import numpy as np

import algorithms
from utils.cluster_draw import cluster_draw


def circle_distribution(radius):
    xs, ys = np.array([]), np.array([])
    for r in radius:
        x = np.arange(0, r, 0.01, dtype=float)
        y = np.sqrt(r ** 2 - x ** 2) + random.choice([0.5, -0.5]) * np.random.random(x.shape)

        xs = np.concatenate((xs, x, x, -1 * x, -1 * x))
        ys = np.concatenate((ys, y, -1 * y, y, -1 * y))

    circle_data = np.zeros((xs.shape[0], 2))
    circle_data[:, 0] = xs
    circle_data[:, 1] = ys

    return circle_data


if __name__ == "__main__":

    data = circle_distribution([10, 10.1, 10.2, 5, 5.1, 5.2])
    clusters, centers = algorithms.k_means(data, 8)
    cluster_draw(data, clusters, "k_means.jpg")
    clusters, centers = algorithms.mean_shift_clustering(data, 2.0)
    cluster_draw(data, clusters, "mean_shift.jpg")
