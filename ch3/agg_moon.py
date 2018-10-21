from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering
from pprint import pprint
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100)

agg = AgglomerativeClustering(n_clusters=3)
pred = agg.fit_predict(X)
score = pred == y
pprint(f"Score {score.sum() / len(score)}")
y0 = pred == 0
y1 = pred == 1

plt.plot(X[y0][:, 0], X[y0][:, 1], 'o', c="red")
plt.plot(X[y1][:, 0], X[y1][:, 1], 'o', c="blue")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("moon_agg.png")
