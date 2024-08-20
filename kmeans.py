import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("cluster_data.csv")
print(df.head())

data = df.to_numpy()


class Kmeansclustering:
    def __init__(self, k):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(datapoint, centroids):
        return np.sqrt(np.sum((centroids - datapoint) ** 2, axis=1))

    def fit(self, data, max_iterations=1000):
        self.centroids = np.random.uniform(
            np.amin(data, axis=0), np.amax(data, axis=0), size=(self.k, data.shape[1])
        )

        for _ in range(max_iterations):
            y = []  # it is label for each datapoint that belongs to which cluster.

            for data_point in data:
                distance = Kmeansclustering.euclidean_distance(
                    data_point, self.centroids
                )
                # it gives a list, each element in list is distnace of a data point to cluster
                # i.e [1,2,3] means our data point is 1 unit distnace from first centroid, 2 units distance from second centroid,
                # 3 units distnace from 3rd centroid.
                clust_num = np.argmin(distance)
                # it basically identifies at what index minimum is there, that index is treated as cluster that point belongs to.
                # [1,2,3] min is at index=0, so 0 is the cluster number it belongs to if we have three clusters 0,1,2.
                y.append(clust_num)

            y = np.array(y)
            # now y is having information what datapoint belongs to which cluster.
            # here indices of y array corresponds to datapoint at that point.

            # now we need to group the same cluster points.
            cluster_indices = []
            # to store a indices that corresponds to particular cluster ,
            # it is a list of list [[points belongs to cluster1],[points belongs to cluster 2]]
            # here we need incides of data points cause if we now inidices we can always accesss data depending on its corresponding indices.
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            # we need to find the mean of new clusters to adjust the centroids
            cluster_centers = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    # this happens if we have more centroids than clusters as some of centroids doesnt have any datapoint.
                    # then in such cases we donot change the centroids of these points
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(
                        np.mean(data[indices], axis=0)[0]
                    )  # this gives a list we to extract its values so we used [0]

            if np.max(self.centroids - np.array(cluster_centers)) < 1e-4:
                break
            else:
                self.centroids = np.array(cluster_centers)
        return y  # this can be treated as label, it tells which point belongs to which cluster


kmeans = Kmeansclustering(k=4)

labels = kmeans.fit(data)

plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(
    kmeans.centroids[:, 0],
    kmeans.centroids[:, 1],
    c=range(len(kmeans.centroids)),
    marker="*",
    s=200,
)
plt.show()
