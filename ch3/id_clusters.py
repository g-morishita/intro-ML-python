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

dbscan = DBSCAN(eps=7, min_samples=3)
labels = dbscan.fit_predict(X_pca)
img_shape = ppl.images[0].shape

for cluster in range(max(labels) + 1):
    mask = labels == cluster
    n_images = np.sum(mask)
    fig, axes = plt.subplots(1, n_images, figsize=(
        n_images * 1.5, 4), subplot_kw={'xticks': (), 'yticks': ()})

    for img, label, ax in zip(X_ppl[mask], y_ppl[mask], axes):
        ax.imshow(img.reshape(img_shape), vmin=0, vmax=1)
        ax.set_title(ppl.target_names[label].split()[-1])

    plt.savefig(f"faces_{cluster}.png")
