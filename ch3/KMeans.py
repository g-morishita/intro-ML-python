from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=1)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print(f"Cluster memberships:\n {kmeans.labels_}")
