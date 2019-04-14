from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(X, y)

for gamma in [1, 0.1, 0.01]:
    svc = SVC(gamma=gamma).fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    auc = roc_auc_score(y_test, svc.decision_function(X_test))
    fpr, tpr, threshold = roc_curve(y_test, svc.decision_function(X_test))
    print(f"gamma = {gamma:.2f}  accuracy = {accuracy:.2f}  AUC = {auc:.2f}")
    plt.plot(fpr, tpr, label=f"gamma={gamma:.3f}")

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("./svc_roc.png")
