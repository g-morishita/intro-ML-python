from sklearn.datasets import fetch_lfw_people
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

ppl = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

mask = np.zeros_like(ppl.target, dtype=np.bool)
for target in np.unique(ppl.target):
    mask[np.where(ppl.target == target)[0][:50]] = 1

X_ppl = ppl.data[mask] / 255
y_ppl = ppl.target[mask]

pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_ppl)
X_pca = pca.transform(X_ppl)

dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print(f"Unique labels default: {np.unique(labels)}")

dbscan = DBSCAN(min_samples=3)
labels = dbscan.fit_predict(X_pca)
print(f"Unique labels lower min_samples: {np.unique(labels)}")

for eps in range(1, 15, 2):
    dbscan = DBSCAN(min_samples=3, eps=eps)
    labels = dbscan.fit_predict(X_pca)
    print(f"Clusters present: {np.unique(labels)}")
    print(f"Cluster size: {np.bincount(labels + 1)}")
