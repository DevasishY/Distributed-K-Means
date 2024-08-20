import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI

np.random.seed(20)
comm = MPI.COMM_WORLD
worker = comm.Get_rank()
size = comm.Get_size()


def distance(data_point, centroids):
    return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))


def labels(data, centroids):
    label = []
    for data_point in data:
        label.append(np.argmin(distance(data_point, centroids)))
    label = np.array(label)
    return label


def update(k, label, centroids, data):
    cluster_indices = []
    for i in range(k):
        cluster_indices.append(np.argwhere(label == i))

    cluster_centers = []
    number_points = np.zeros(k)

    for i, indices in enumerate(cluster_indices):
        if len(indices) == 0:
            cluster_centers.append(centroids[i])
            number_points[i] = 0
        else:
            cluster_centers.append(np.mean(data[indices], axis=0)[0])
            centroids = np.array(cluster_centers)
            number_points[i] = len(indices)

    return centroids, number_points


k = 4
if worker == 0:
    df = pd.read_csv("cluster_data.csv")
    data = df.to_numpy()  # or we can use df.values --> gives a 2d array
    m = data.shape[1]
    split = np.array_split(data, size)
else:
    data = None
    split = None

split_data = comm.scatter(split, root=0)

if worker == 0:
    centroids = data[np.random.choice(len(data), k, replace=False)]
else:
    centroids = None

centroid_rank = comm.bcast(centroids, root=0)

for _ in range(10):
    cluster_labels = labels(split_data, centroid_rank)
    updated_centroids, sizes_of_each_cluster = update(
        k, cluster_labels, centroid_rank, split_data
    )
    all_centroids = comm.gather(updated_centroids, root=0)
    all_sizes = comm.gather(sizes_of_each_cluster, root=0)

    if worker == 0:
        sum = np.zeros((k, m))
        final_centroids = np.zeros((k, m))
        sizes = np.zeros(k)
        for i in range(k):
            for sub in range(len(all_centroids)):
                sum[i] = sum[i] + all_centroids[sub][i] * all_sizes[sub][i]
                sizes[i] += all_sizes[sub][i]
                final_centroids[i] = sum[i] / sizes[i]
        centroid_rank = final_centroids  # update centroid_rank with final centroids

        # Broadcast updated centroids to all processes
    centroid_rank = comm.bcast(centroid_rank, root=0)

if worker == 0:
    cluster_labels = labels(data, final_centroids)
    plt.scatter(data[:, 0], data[:, 1], c=cluster_labels)
    plt.show()
