import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
from mglearn import cm3
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

algos = [KMeans(n_clusters=2), AgglomerativeClustering(), DBSCAN()]

random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={
                         'xticks': (), 'yticks': ()})

axes[0].scatter(X[:, 0], X[:, 1], c=random_clusters, cmap=cm3, s=60)
axes[0].set_title(
    f"Rondom assignment - ARI: {adjusted_rand_score(y, random_clusters): .2f})")

for ax, alg in zip(axes[1:], algos):
    clusters = alg.fit_predict(X_scaled)
    ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap=cm3, s=60)
    ax.set_title(
        f"{alg.__class__.__name__}- ARI: {adjusted_rand_score(y, clusters): .2f})")

plt.savefig("cls_comp.png")
