import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()
train_X, test_X, train_y, test_y = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0)

for C, marker in zip([1, 100, 0.001], ['o', '^', 'v']):
    model = LogisticRegression(C=C, penalty="l1")
    model.fit(train_X, train_y)

    plt.plot(model.coef_.T, marker, label=f"C={C}")

plt.ylim(-5, 5)
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, range(cancer.data.shape[1]))
plt.title("Logistic Regression with L1 regularization")
plt.ylabel("Coefficient maginitude")
plt.xlabel("Coefficient index")
plt.legend(loc=3)
plt.savefig("logreg_l1.png", bbox_inches='tight')
