import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
train_X, test_X, train_y, test_y = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0)

logreg = LogisticRegression(C=1)  # default
logreg100 = LogisticRegression(C=100)
logreg0001 = LogisticRegression(C=0.001)

for model in [logreg, logreg100, logreg0001]:
    model.fit(train_X, train_y)
    print(f"C={model.C}, Training Score: {model.score(train_X, train_y):.3f}, Test Score: {model.score(test_X, test_y):.3f}")

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg0001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)

plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.savefig("logreg.png", bbox_inches='tight')
