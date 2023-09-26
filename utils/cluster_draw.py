import matplotlib.pyplot as plt


def cluster_draw(data, clusters_id, file_name="image.jpg"):
    xs, ys = data[:, 0], data[:, 1]

    total_clusters = len(set(clusters_id.tolist()))
    cluster_id = 0
    while cluster_id != total_clusters:
        cluster_mask = clusters_id == cluster_id
        cluster_x = xs[cluster_mask]
        cluster_y = ys[cluster_mask]

        plt.scatter(cluster_x, cluster_y)
        cluster_id += 1

        # plt.show()
        plt.savefig(file_name)
