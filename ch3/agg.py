from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from mglearn import discrete_scatter
import matplotlib.pyplot as plt

X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.savefig("agg_clustering.png")
