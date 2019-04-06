import numpy as np
from mglearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=(4000, 500), centers=2,
                  cluster_std=[7, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=0.05).fit(X_train, y_train)
precision, recall, threshold = precision_recall_curve(
    y_test, svc.decision_function(X_test))
close_zero = np.argmin(np.abs(threshold))
plt.plot(precision[close_zero], recall[close_zero], marker='o',
         markersize=10, label="threshold 0 svm", fillstyle="none", c='k', mew=2)
plt.plot(precision, recall, label="svm")
plt.xlabel("Precision")
plt.ylabel("Recall")

rf = RandomForestClassifier(
    n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)
precision_rf, recall_rf, threshold_rf = precision_recall_curve(
    y_test, rf.predict_proba(X_test)[:, 1])

close_zero_rf = np.argmin(np.abs(threshold_rf-0.5))
plt.plot(precision_rf, recall_rf, label="rf")
plt.plot(precision_rf[close_zero_rf], recall_rf[close_zero_rf], '^',
         c='k', markersize=10, label="threshold 0.5 rf", fillstyle="none", mew=2)
plt.legend(loc='best')
plt.savefig("./ex2.png")
