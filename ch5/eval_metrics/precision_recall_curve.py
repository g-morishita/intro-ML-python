import numpy as np
from mglearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=(4000, 500), centers=2,
                  cluster_std=[7, 2], random_state=22)
X_train, X_test, y_tran, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=0.05).fit(X_train, y_tran)
precision, recall, threshold = precision_recall_curve(
    y_test, svc.decision_function(X_test))
close_zero = np.argmin(np.abs(threshold))
plt.plot(precision[close_zero], recall[close_zero], marker='o',
         markersize=10, label="threshold zero", fillstyle="none", c='k', mew=2)
plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.savefig("./ex.png")
