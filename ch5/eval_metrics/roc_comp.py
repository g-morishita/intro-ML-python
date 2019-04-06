import numpy as np
from mglearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=(4000, 500), centers=2,
                  cluster_std=[7, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=0.05).fit(X_train, y_train)
fpr, tpr, threshold = roc_curve(y_test, svc.decision_function(X_test))
close_zero = np.argmin(np.abs(threshold))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
         label='svc threshold 0', fillstyle="none", c='k', mew=2)
plt.plot(fpr, tpr, label="svm ROC curve")

rf = RandomForestClassifier(n_estimators=100, max_features=2, random_state=0)
rf.fit(X_train, y_train)
fpr_rf, tpr_rf, threshold_rf = roc_curve(
    y_test, rf.predict_proba(X_test)[:, 1])
close_zero = np.argmin(np.abs(threshold - 0.5))
plt.plot(fpr_rf, tpr_rf, label="rf ROC curve")
plt.plot(fpr_rf[close_zero], tpr_rf[close_zero], '^', markersize=10,
         label="rf threshold 0.5", fillstyle="none", c='k', mew=2)
plt.legend(loc="best")
plt.savefig("./ex3.png")
