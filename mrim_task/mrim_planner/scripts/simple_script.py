from sklearn.cluster import KMeans

import numpy as np

X = np.array([[1.3 , 2.1,  3], [1, 4, 7], [1, 0 ,3],

              [10, 2, 4], [10, 4, 5], [10, 0, -3.3]])

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

print(X)
print(kmeans)
print(kmeans.get_params())
print(kmeans.labels_)
print(kmeans.cluster_centers_)

