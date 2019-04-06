from mglearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

X, y = make_blobs(n_samples=(400, 50), centers=2,
                  cluster_std=[7, 2], random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=0.05).fit(X_train, y_train)
print(
    f"before changing the decision function:\n{classification_report(y_test, svc.predict(X_test))}")

y_pred_lower_threshold = svc.decision_function(X_test) > -0.8

print(
    f"After changin the decision function:\n{classification_report(y_test, y_pred_lower_threshold)}")
