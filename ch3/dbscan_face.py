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

dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
print(f"Unique labels lower min_samples and higher eps: {np.unique(labels)}")

print(f"Number of points per cluster {np.bincount(labels + 1)}")

noise = X_ppl[labels == -1]
fig, axes = plt.subplots(3, 9, figsize=(12, 4), subplot_kw={
                         'xticks': (), 'yticks': ()})

img_shape = ppl.images[0].shape
for img, ax in zip(noise, axes.ravel()):
    ax.imshow(img.reshape(img_shape), vmin=0, vmax=1)
plt.savefig("noise_face.png")
