import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=200, centers=2)
X_train, X_test, y_train, y_test = train_test_split(X, y)
svc = SVC(C=0.1).fit(X_train, y_train)
fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))

plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel('FPR')
plt.ylabel('TPR')
close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
         label='threshold zero', fillstyle="none", c='k', mew=2)
plt.legend(loc=4)
plt.savefig("./roc.png")
