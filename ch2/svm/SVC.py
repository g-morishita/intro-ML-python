from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)

print(f"Accuracy on training set: {svc.score(X_train, y_train):.3f}")
print(f"Accuracy on test set: {svc.score(X_test, y_test):.3f}")

plt.plot(X_train.max(axis=0), 'o', label="max")
plt.plot(X_train.min(axis=0), '^', label="min")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")
plt.savefig("svc.png")
