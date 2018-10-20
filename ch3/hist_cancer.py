import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from mglearn import cm3

cancer = load_breast_cancer()
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color=cm3(0), alpha=0.5)
    ax[i].hist(benign[:, i], bins=bins, color=cm3(2), alpha=0.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignat", "benign"], loc="best")
fig.tight_layout()
fig.savefig("hist_cancer.png")
